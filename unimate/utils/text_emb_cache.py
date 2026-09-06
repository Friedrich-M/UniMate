"""On-disk cache of text-encoder features.

The dataset pre-encodes every unique caption and joint name at construction
time, which means loading a text encoder onto the GPU and running it on every
run — seconds of startup and several GB of VRAM for a model that is thrown away
immediately afterwards. Caching those features next to the features removes
both: with a complete cache the encoder is never instantiated at all.

**Per-token, not pooled.** Each text is stored as its ``(T_i, D)`` sequence of
non-padding token features, because ``text_cond='cross_attn'`` needs the
sequence — a pooled vector would leave its cross-attention attending a single
token, i.e. a more expensive adaLN. The pooled vector ``adaLN`` wants is the
mean over those same rows, so it is derived on load rather than stored: one
source of truth, and it cannot drift from the tokens.

A caption key carries an embedding and a mask: ``emb`` is padded to the file's
longest sequence and ``mask`` marks its real tokens. ``load_cache`` returns
each key trimmed by its own mask, so the padding lives in the file (where zlib
compresses it away — a padded file is ~1% larger on disk) and never in RAM,
which is where it would actually cost: truebones captions are 42% full, so a
resident padded array would be 107 MB against 45 MB, and objaverse's 1.2 GB
against 431 MB.

Two files per feature directory, flat beside ``captions.json`` — captions and
joint names are separate sources with very different shapes (a caption is ~13
tokens, a joint name ~2), and only one of them changes when captions are
re-annotated::

    dataset/features/<ds>/caption_emb_cache.npz  one entry per CLIP KEY, i.e.
                                                 the keys of captions.json
        keys : (N,) unicode        the clip key to look up by
        emb  : (N, T_max, D) f32   per-token features, right-padded
        mask : (N, T_max) bool     True on this key's real tokens

    dataset/features/<ds>/joint_emb_cache.npz    one entry per UNIQUE joint name
        keys : (N,) unicode        the joint name
        emb  : (N, D) f32          the pooled vector, no mask

    both:
        meta : () object           {'encoder_type', 'encoder_version',
                                    'emb_dim', 'format', 'pooled'}

Joint names are stored pooled because that is all the model ever sees of them:
``joint_names_emb`` is one vector per joint. Captions keep their sequence
because ``text_cond='cross_attn'`` attends it; the pooled vector ``adaLN`` uses
is the mean over those rows, derived by the data loader.

The encoder is no longer part of the filename, so ``meta`` is the only thing
standing between a t5 cache and a clip run: ``load_cache`` refuses a file whose
recorded encoder is not the one being asked for, alongside the format-version
and dimension checks.

Written by ``unimate/tools/precompute_text_emb.py``; read by the mixture
dataset. A missing or partial cache is never fatal — the dataset encodes
whatever the cache does not cover.
"""

import os
from os.path import join as pjoin
from typing import Dict, List, Optional

import numpy as np

from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)

# Bumped when the on-disk layout changes. v1 stored one pooled vector per text,
# v2 a ragged token sequence, v3 an (emb, mask) pair per key, v4 pools the
# joint names. An older file is ignored rather than misread.
CACHE_FORMAT = 4


# The two text sources: the file each is cached in, and whether it is stored
# pooled (one vector) or as a token sequence with a mask.
KINDS = {'caption': ('caption_emb_cache.npz', False),  # captions.json
         'joint': ('joint_emb_cache.npz', True)}       # cond.npy joint names


def cache_path(root_dir: str, kind: str) -> str:
    """Path of one feature directory's cache for *kind* ('text' | 'joint').

    Flat, beside ``captions.json`` and ``cond.npy`` — the files it is derived
    from — rather than in a subdirectory of its own.
    """
    if kind not in KINDS:
        raise ValueError(f'unknown cache kind {kind!r}; expected {sorted(KINDS)}')
    return pjoin(root_dir, KINDS[kind][0])


def sequences_from_hidden(hidden, attention_mask) -> List[np.ndarray]:
    """Split an encoder's ``(B, T, D)`` output into per-text ``(T_i, D)`` rows.

    Padding is dropped by the mask. An all-masked row — which is how
    ``T5Conditioner.tokenize`` marks an empty string — yields a single ZERO
    row, so its pooled vector is the zero vector the ``pool=True`` encoder
    returns for empty text. Keeping the first real token instead would hand
    back the ``</s>`` state and quietly change what "unconditional" means.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    lengths = np.asarray(attention_mask).sum(axis=-1).astype(int)
    out = []
    for i, n in enumerate(lengths):
        out.append(hidden[i, :n] if n > 0
                   else np.zeros((1, hidden.shape[-1]), dtype=np.float32))
    return out


def pooled_from_hidden(hidden, attention_mask) -> np.ndarray:
    """The ``pool=True`` vector of an encoder's ``(B, T, D)`` output.

    This is byte-for-byte ``T5Conditioner.forward``'s pooling arithmetic —
    the masked sum over all T positions divided by the token count — run on
    the same device and dtype as the hidden states. Re-deriving it later from
    the trimmed rows (a plain mean over T_i) is mathematically the same but
    sums in a different order, so it lands a few ulps away; adaLN reads this
    stored vector instead, and gets exactly what the pooled encoder returned.
    """
    mask = attention_mask.to(hidden.dtype)
    token_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)       # (B, 1)
    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=-2) / token_count
    return pooled.detach().cpu().numpy().astype(np.float32)


def pool(tokens: np.ndarray) -> np.ndarray:
    """Pooled ``(D,)`` vector of a ``(T, D)`` token sequence.

    Mathematically the encoder's ``pool=True`` output — padding is already
    excluded from the stored rows — but summed in a different order, so it can
    differ in the last bits. Only for sequences whose stored pooled vector is
    unavailable; prefer :func:`pooled_from_hidden` at write time.
    """
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError(f'expected a non-empty (T, D) sequence, got {tokens.shape}')
    return tokens.mean(axis=0)


def save_cache(path: str, tokens_by_key: Dict[str, np.ndarray], kind: str,
               encoder_type: str, encoder_version: Optional[str],
               pooled_by_key: Optional[Dict[str, np.ndarray]] = None) -> int:
    """Write ``{key: (T_i, D) tokens}`` to *path*. Returns the entry count.

    A pooled *kind* ('joint') stores one vector per key as ``(N, D)`` and no
    mask, since that is all the model ever reads. A sequence kind ('caption')
    right-pads to the longest sequence, pairs it with a mask, and also stores
    the pooled vector so the adaLN path reads the encoder's own number rather
    than re-deriving it.

    ``pooled_by_key`` should come from :func:`pooled_from_hidden`; without it
    the pooled vectors fall back to :func:`pool` over the sequences.
    """
    if not tokens_by_key:
        raise ValueError('refusing to write an empty text-embedding cache')
    if kind not in KINDS:
        raise ValueError(f'unknown cache kind {kind!r}; expected {sorted(KINDS)}')
    pooled = KINDS[kind][1]

    keys = sorted(tokens_by_key)
    seqs = [np.asarray(tokens_by_key[k], dtype=np.float32) for k in keys]
    dims = {x.shape[1] for x in seqs}
    if len(dims) != 1:
        raise ValueError(f'inconsistent embedding dims in the cache: {sorted(dims)}')
    dim = dims.pop()

    def _pooled(k, x):
        if pooled_by_key is not None and k in pooled_by_key:
            return np.asarray(pooled_by_key[k], dtype=np.float32)
        return pool(x)

    arrays = {'keys': np.asarray(keys, dtype=object)}
    if pooled:
        arrays['emb'] = np.stack([_pooled(k, x)
                                  for k, x in zip(keys, seqs)])    # (N, D)
    else:
        t_max = max(len(x) for x in seqs)
        emb = np.zeros((len(keys), t_max, dim), dtype=np.float32)
        mask = np.zeros((len(keys), t_max), dtype=bool)
        for i, x in enumerate(seqs):
            emb[i, :len(x)] = x
            mask[i, :len(x)] = True
        arrays['emb'] = emb
        arrays['mask'] = mask
        arrays['pooled'] = np.stack([_pooled(k, x)
                                     for k, x in zip(keys, seqs)])  # (N, D)

    arrays['meta'] = np.asarray({'encoder_type': encoder_type,
                                 'encoder_version': encoder_version,
                                 'emb_dim': int(dim),
                                 'format': CACHE_FORMAT,
                                 'pooled': pooled}, dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Write-then-rename: an interrupted run must not leave a truncated cache
    # that later loads as a silently incomplete one. A file object keeps
    # np.savez from appending '.npz' to the temp name, which would leave the
    # rename pointing at a file that never existed.
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return len(keys)


def load_cache(root_dir: str, kind: str, encoder_type: str,
               encoder_version: Optional[str],
               expected_dim: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load one feature directory's *kind* cache.

    ``{key: (D,)}`` for a pooled kind ('joint'); for a sequence kind
    ('caption') ``{key: ((T_i, D) tokens, (D,) pooled)}`` — the tokens trimmed
    by their mask, the pooled vector as the encoder produced it.

    ``{}`` when absent or unusable. Never raises on a bad file: a corrupt,
    outdated, wrong-encoder or dimension-mismatched cache is logged and
    ignored, so the caller falls back to encoding.
    """
    path = cache_path(root_dir, kind)
    if not os.path.isfile(path):
        return {}
    try:
        data = np.load(path, allow_pickle=True)
        meta = data['meta'].item() if 'meta' in data else {}
        if meta.get('format') != CACHE_FORMAT:
            logger.warning(
                f'Ignoring text-embedding cache {path}: format '
                f'{meta.get("format")}, this build reads {CACHE_FORMAT} — '
                f'regenerate it with unimate/tools/precompute_text_emb.py')
            return {}
        got = (meta.get('encoder_type'), meta.get('encoder_version'))
        if got != (encoder_type, encoder_version):
            logger.warning(
                f'Ignoring text-embedding cache {path}: built with '
                f'{got[0]}/{got[1]}, this run uses {encoder_type}/'
                f'{encoder_version} — regenerate it with '
                f'unimate/tools/precompute_text_emb.py')
            return {}
        if bool(meta.get('pooled')) != KINDS[kind][1]:
            logger.warning(f'Ignoring text-embedding cache {path}: pooled='
                           f'{meta.get("pooled")} but {kind!r} expects '
                           f'pooled={KINDS[kind][1]}')
            return {}
        keys = [str(k) for k in data['keys']]
        emb = np.asarray(data['emb'], dtype=np.float32)
        mask = None if KINDS[kind][1] else np.asarray(data['mask'], dtype=bool)
        stored_pooled = (np.asarray(data['pooled'], dtype=np.float32)
                         if 'pooled' in data.files else None)
    except Exception as e:  # noqa: BLE001 — a bad cache is a cache miss
        logger.warning(f'Ignoring unreadable text-embedding cache {path}: {e}')
        return {}

    if len(keys) != len(emb) or (mask is not None and emb.shape[:2] != mask.shape):
        logger.warning(f'Ignoring text-embedding cache {path}: {len(keys)} keys '
                       f'against emb {emb.shape}')
        return {}
    if expected_dim is not None and emb.shape[-1] != expected_dim:
        logger.warning(
            f'Ignoring text-embedding cache {path}: {emb.shape[-1]}-d '
            f'features but the model expects {expected_dim}-d — '
            f'regenerate it with unimate/tools/precompute_text_emb.py')
        return {}
    if mask is None:
        logger.info(f'Loaded {len(keys)} pooled text embeddings from {path}')
        return {k: emb[i] for i, k in enumerate(keys)}
    if stored_pooled is None or len(stored_pooled) != len(keys):
        logger.warning(f'Ignoring text-embedding cache {path}: no stored pooled '
                       f'vectors — regenerate it with '
                       f'unimate/tools/precompute_text_emb.py')
        return {}
    # Boolean indexing copies, so the padded (N, T_max, D) block is released
    # once these per-key sequences are built — the padding stays on disk.
    logger.info(f'Loaded {len(keys)} cached text sequences '
                f'({int(mask.sum())} tokens) from {path}')
    return {k: (emb[i][mask[i]], stored_pooled[i]) for i, k in enumerate(keys)}


def load_caches(root_dirs: List[str], kind: str, encoder_type: str,
                encoder_version: Optional[str],
                expected_dim: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Merge the *kind* caches of several feature directories.

    Joint names are shared vocabulary and encode identically everywhere;
    clip keys are unique per dataset. Either way an overlapping entry is
    interchangeable and the last one simply wins.
    """
    merged: Dict[str, np.ndarray] = {}
    for root in root_dirs:
        if root:
            merged.update(load_cache(root, kind, encoder_type, encoder_version,
                                     expected_dim))
    return merged
