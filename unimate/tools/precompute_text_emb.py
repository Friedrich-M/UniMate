"""Pre-encode a feature directory's text into an on-disk feature cache.

The mixture dataset encodes every unique caption and joint name when it is
constructed: a text encoder is loaded onto the GPU, run once, and thrown away —
on every training run, every sampling run, and once per worker. This tool does
that work once and writes the result next to the features, after which the
dataset loads the embeddings and never instantiates the encoder at all.

One file per source, each keyed the way the dataset looks it up:

  * ``captions.json``  -> ``caption_emb_cache.npz`` one entry per CLIP KEY (the
                          keys of captions.json), so a clip's features are a
                          direct lookup; two clips sharing a caption each get
                          their own entry
  * ``cond.npy``       -> ``joint_emb_cache.npz``  one entry per UNIQUE joint
                          name, taken from the cleaned vocabulary
                          ``clean_joint_names`` (see
                          ``dataset.model_joint_names``)

A caption key gets an embedding and a mask — right-padded to the file's
longest, the mask marking the real tokens. Joint names are stored pooled, one
vector per name, because that is all the model reads of them
(``joint_names_emb`` is one vector per joint).

Repeated caption strings are still encoded only once — the dedupe happens
before the forward pass, not in the file.

Captions are stored per token because ``text_cond='cross_attn'`` attends the
sequence; the pooled vector ``adaLN`` uses is the mean over the same rows,
derived by the data loader. The encoder is recorded in each file's metadata and
checked on load, so a cache built with one encoder is never read by a run
using another.

Usage (from the repo root)::

    # every dataset/features/* with the config's encoder
    python -m unimate.tools.precompute_text_emb --config configs/uniml3d_90frames.json

    # one input folder (positional or --input_dir), encoder given explicitly
    python -m unimate.tools.precompute_text_emb --input_dir dataset/features/truebones \\
        --encoder_type t5 --encoder_version google/flan-t5-base

    # re-encode even if a cache is already there
    python -m unimate.tools.precompute_text_emb --config <cfg> --overwrite

Re-run it after regenerating captions or joint names: entries the cache does
not cover are encoded at load time anyway, so a stale cache costs speed rather
than correctness, but a fresh one keeps the encoder out of the run entirely.
"""

import argparse
import glob
import json
import os
from os.path import join as pjoin

import numpy as np
import torch
from tqdm import tqdm

from unimate.configs.schema import MainConfig
from unimate.dataset.mixture.dataset import model_joint_names
from unimate.models.text_encoder.factory import create_text_encoder
from unimate.utils.logger import get_logger
from unimate.utils.text_emb_cache import (KINDS, cache_path, load_cache,
                                          pooled_from_hidden, save_cache,
                                          sequences_from_hidden)

logger = get_logger(file_name=__file__)

DEFAULT_GLOB = 'dataset/features/*'


# ---------------------------------------------------------------------------
# Collecting the strings a feature directory needs embedded
# ---------------------------------------------------------------------------

def collect_texts(root_dir):
    """What each source needs encoded -> ``{kind: {key: text}}``.

    ``'caption'`` maps every clip key to its caption; ``'joint'`` maps each
    unique joint name to itself. A source with no file simply does not appear.
    """
    out = {}

    captions_path = pjoin(root_dir, 'captions.json')
    if os.path.isfile(captions_path):
        with open(captions_path) as f:
            captions = json.load(f)
        out['caption'] = {clip: cap for clip, cap in captions.items() if cap}

    cond_path = pjoin(root_dir, 'cond.npy')
    if os.path.isfile(cond_path):
        cond = np.load(cond_path, allow_pickle=True).item()
        # Same rule the dataset applies (cleaned vocabulary, raw only as a
        # loud fallback) — a different rule here would leave the lookups the
        # dataset actually performs uncovered by this cache.
        out['joint'] = {
            name: name
            for ot, entry in cond.items()
            for name in model_joint_names(entry, ot)
            if name
        }

    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_texts(texts, encoder, chunk_size, label):
    """Encode *texts* in chunks -> ``({text: (T_i, D)}, {text: (D,) pooled})``.

    The encoder is built with ``pool=False``, so each chunk comes back as
    ``(B, T, D)`` padded to the chunk's longest text; ``sequences_from_hidden``
    slices every row back to its own token count, so no padding is stored and
    the result does not depend on how the texts were chunked. Chunking bounds
    peak GPU memory the same way the dataset's own pre-encoding does.
    """
    seqs_out, pooled_out = {}, {}
    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    for i in tqdm(range(0, len(texts), chunk_size), total=n_chunks,
                  desc=f'Encoding {label}'):
        chunk = texts[i:i + chunk_size]
        with torch.no_grad():
            inputs = encoder.tokenize(chunk)
            hidden = encoder(inputs)                       # (B, T, D)
        # Pool on the encoder's own terms, before anything moves to CPU, so
        # the adaLN path reads exactly what a pool=True encoder would return.
        pooled = pooled_from_hidden(hidden, inputs['attention_mask'])
        seqs = sequences_from_hidden(hidden.detach().cpu(),
                                     inputs['attention_mask'].cpu())
        seqs_out.update(dict(zip(chunk, seqs)))
        pooled_out.update({t: pooled[j] for j, t in enumerate(chunk)})
    return seqs_out, pooled_out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_roots(inputs):
    """Feature directories from explicit paths or the default glob."""
    if inputs:
        roots = list(inputs)
    else:
        roots = sorted(p for p in glob.glob(DEFAULT_GLOB) if os.path.isdir(p))
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        raise SystemExit(f'Not a directory: {", ".join(missing)}')
    return roots


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inputs', nargs='*', metavar='INPUT_DIR',
                    help='Feature directory / directories to process, e.g. '
                         'dataset/features/truebones. Same as --input_dir.')
    ap.add_argument('--input_dir', action='append', default=[], metavar='DIR',
                    help='Feature directory to process; repeatable. With '
                         f'neither form given, every {DEFAULT_GLOB} is done.')
    ap.add_argument('--config', default=None,
                    help='Training config to take the encoder type / version '
                         'from; --encoder_type / --encoder_version override it')
    ap.add_argument('--encoder_type', default=None, help="'t5', 'clip', 'bert'")
    ap.add_argument('--encoder_version', default=None,
                    help="e.g. 'google/flan-t5-base' (None = the type's default)")
    ap.add_argument('--chunk_size', type=int, default=256,
                    help='Texts per forward pass (default: 256)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-encode everything instead of extending the cache')
    return ap.parse_args()


def main():
    args = parse_args()

    encoder_type, encoder_version = args.encoder_type, args.encoder_version
    if args.config:
        cfg = MainConfig.from_json(args.config)
        encoder_type = encoder_type or cfg.model.text_encoder_type
        encoder_version = encoder_version or cfg.model.text_encoder_version
    if not encoder_type:
        raise SystemExit('Give --encoder_type or --config')

    roots = resolve_roots(list(args.inputs) + list(args.input_dir))
    logger.info(f'Encoder: {encoder_type} / {encoder_version} on {args.device}')

    # The encoder is built on the first directory that actually needs it, so a
    # run where every cache is already complete never loads it.
    encoder = None
    for root in roots:
        by_kind = collect_texts(root)
        if not by_kind:
            logger.info(f'[{root}] no captions.json / cond.npy; skipped')
            continue

        for kind, text_by_key in by_kind.items():
            hit = ({} if args.overwrite
                   else load_cache(root, kind, encoder_type, encoder_version))
            # A sequence kind loads as (tokens, pooled) pairs; split them so
            # both views survive an incremental re-run.
            if KINDS[kind][1]:
                cached, cached_pooled = dict(hit), {}
            else:
                cached = {k: v[0] for k, v in hit.items()}
                cached_pooled = {k: v[1] for k, v in hit.items()}
            todo_keys = [k for k in text_by_key if k not in cached]
            path = cache_path(root, kind)
            if not todo_keys:
                logger.info(f'[{root}] {kind}: {len(text_by_key)} keys already '
                            f'cached in {os.path.basename(path)}; nothing to do')
                continue

            # Encode each distinct string once even when several keys share it
            # (a caption repeated across clips), then fan the result back out.
            unique = sorted({text_by_key[k] for k in todo_keys})
            logger.info(f'[{root}] {kind}: {len(text_by_key)} keys, '
                        f'{len(todo_keys)} to encode over {len(unique)} distinct '
                        f'strings, {len(cached)} reused')
            if encoder is None:
                # pool=False: the cache stores token sequences; pooling for the
                # adaLN path happens in the data loader.
                encoder = create_text_encoder(encoder_type=encoder_type,
                                              encoder_version=encoder_version,
                                              device=args.device, pool=False)
            by_text, pooled_by_text = encode_texts(
                unique, encoder, args.chunk_size,
                f'{os.path.basename(root.rstrip("/"))}/{kind}')
            cached.update({k: by_text[text_by_key[k]] for k in todo_keys})
            cached_pooled.update({k: pooled_by_text[text_by_key[k]]
                                  for k in todo_keys})
            n = save_cache(path, cached, kind, encoder_type, encoder_version,
                           pooled_by_key=cached_pooled)
            what = 'pooled vectors' if KINDS[kind][1] else 'token sequences'
            logger.info(f'[{root}] wrote {n} {what} -> {path}')

    if encoder is None:
        logger.info('Every cache was already complete; the encoder was never loaded.')


if __name__ == '__main__':
    main()
