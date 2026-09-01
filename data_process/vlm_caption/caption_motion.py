"""Caption multi-view rendered motion sequences with a Vision-Language Model.

Supports three backends (see :mod:`data_process.vlm_caption.backends`):
    qwen    — Local Qwen3-VL / Qwen3.5 via HuggingFace transformers (default).
    openai  — OpenAI-compatible API (GPT-4o, GPT-5, ...).
    gemini  — Google Gemini API.

The backend is auto-detected from the model name, or set explicitly via
``--backend``. ``--task`` selects the dataset system prompt (see
:mod:`data_process.vlm_caption.prompts`): ``mixamo``, ``objaverse`` or
``truebones``.

Input layout: ``<render_root>/<motion_name>/v00{0..3}/*.png`` — one
directory per motion with one sub-directory of frame PNGs per camera view.

``--hints_json`` optionally feeds per-clip reference labels from dataset
metadata (Mixamo's official catalogue prompts, truebones' T2M4LVO
annotations) to the model as frame-grounded reference labels;
``run_caption_motion.sh`` wires this up for both by default.

Usage:
    # Local Qwen (auto-detected from model name)
    python -m data_process.vlm_caption.caption_motion \
        --render_root dataset/render/truebones --task truebones

    # Gemini 3 Flash API
    python -m data_process.vlm_caption.caption_motion \
        --render_root dataset/render/objaverse \
        --model gemini-3-flash-preview --task objaverse --num_workers 16

    # Multi-GPU (qwen backend only)
    python -m data_process.vlm_caption.caption_motion \
        --render_root dataset/render/objaverse --task objaverse --num_gpus 4 --gpu_id 0

    # Merge shards after multi-GPU
    python -m data_process.vlm_caption.caption_motion \
        --merge_shards --output_json motion_captions.json --num_gpus 4
"""

import argparse
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from data_process.vlm_caption.backends import (
    BACKEND_GEMINI,
    BACKEND_QWEN,
    MAX_RETRIES,
    VALID_BACKENDS,
    detect_backend,
    generate_gemini,
    generate_openai,
    generate_qwen,
    image_params,
)
from data_process.vlm_caption.prompts import TASK_PROMPTS, get_prompt
from data_process.utils.vlm import (
    DEFAULT_MAX_FRAMES_PER_VIEW,
    add_multi_gpu_args,
    find_motion_dirs,
    frame_mime_type,
    is_completed,
    load_json,
    load_multiview_frames,
    load_multiview_videos,
    load_qwen_model,
    merge_shards,
    render_incomplete_reason,
    save_json,
    shard_output_path,
)


SAVE_EVERY = 20  # captions per incremental JSON flush


def load_hints(hints_json):
    # type: (Optional[str]) -> Dict[str, str]
    """Load per-clip reference labels from a dataset metadata JSON.

    Accepts either flat ``{name: label}`` or record values like Mixamo's
    ``animation_motion_prompts.json`` (``{name: {prompt, description, ...}}``,
    where ``description`` is preferred over ``prompt`` as the more specific
    label). Keys are normalised to the render clip-directory name by
    stripping a file extension, so ``Jab_Cross.fbx`` matches the render dir
    ``Jab_Cross``.
    """
    if not hints_json:
        return {}
    hints = {}
    for key, value in load_json(hints_json).items():
        name = os.path.splitext(key)[0]
        if isinstance(value, dict):
            label = value.get("description") or value.get("prompt") or ""
            # Keep only the ACTION as reference: records that name their
            # subject (truebones' object_name, e.g. 'Tyrannosaurus Rex')
            # get it replaced with the neutral word so no species reaches
            # the model.
            subject = value.get("object_name")
            if subject and label:
                label = re.sub(re.escape(subject), "animal", label,
                               flags=re.IGNORECASE)
                label = (label.replace("A animal", "An animal")
                              .replace("a animal", "an animal"))
        else:
            label = str(value)
        if label:
            hints[name] = label
    logger.info("Loaded {} caption hints from {}".format(len(hints), hints_json))
    return hints


# ─────────────────────────────────────────────────────────────────────────────
# Batch captioning
# ─────────────────────────────────────────────────────────────────────────────

def caption_batch(
    render_root,
    backend,
    system_prompt,
    output_json=None,
    downsample_rate=1,
    max_frames_per_view=None,
    max_tokens=300,
    views=None,
    hints=None,
    vision_input="video",
    gpu_id=0,
    num_gpus=1,
    num_workers=1,
    # qwen-specific
    qwen_model=None,
    qwen_processor=None,
    # api-specific
    model_name=None,
    api_key=None,
    base_url=None,
    max_retries=MAX_RETRIES,
    # gemini-specific
    thinking_level="low",
    media_resolution="medium",
    thinking_budget=0,
):
    """Caption all motion sequences under a root render directory.

    Skips sequences that already have captions in the output JSON (for
    resumability). Clips whose render is incomplete are reported as failures
    and left out of the JSON, so they are retried once rendering finishes.
    Results are saved incrementally (every SAVE_EVERY captions).

    Args:
        max_frames_per_view: Cap on frames sent per view after
            ``downsample_rate`` (0 = no cap; frames are sampled
            uniformly across the clip, first and last always kept).
            Default None resolves per mode: no cap for 'video' (its
            2-frame temporal patching keeps even a 200-frame clip
            cheap), DEFAULT_MAX_FRAMES_PER_VIEW for 'images'.
        vision_input: 'video' (default) sends each view through the
            qwen backend's native video path (2-frame temporal patching
            + timestamp alignment, ~half the visual tokens of 'images');
            'images' sends individually labeled frames. Video is
            qwen-only; API backends fall back to images with a warning.
        hints: Optional ``{clip_name: reference_label}`` mapping (see
            ``load_hints``); a clip's label is passed to the model as a
            verify-against-frames hint.
        gpu_id: GPU/process index for multi-GPU sharding (default 0).
        num_gpus: Total number of GPUs/processes (default 1 = no sharding).
            When num_gpus > 1, each process handles every num_gpus-th motion
            and saves to a shard file. Use merge_shards() afterwards.
        num_workers: Concurrent in-process workers. Only effective for HTTP
            backends (gemini/openai); Qwen ignores this since one local model
            can't be shared across threads safely.
    """
    render_root = Path(render_root)

    if output_json is None:
        output_json = str(render_root / "motion_captions.json")

    save_path = shard_output_path(output_json, gpu_id) if num_gpus > 1 else output_json
    save_path_obj = Path(save_path)
    failed_path = str(save_path_obj.with_name(save_path_obj.stem + "_failed.txt"))

    captions = load_json(save_path)

    # Resume from the merged output as well: shard files are not cleaned up
    # after a merge, so a tidied-up directory would otherwise re-caption the
    # whole dataset. The shard wins on conflict; only the shard's own entries
    # are written back to save_path.
    already_done = dict(captions)
    if save_path != output_json:
        for name, caption in load_json(output_json).items():
            already_done.setdefault(name, caption)

    motion_dirs = find_motion_dirs(str(render_root))

    failed_set = set()
    if os.path.exists(failed_path):
        with open(failed_path, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    failed_set.add(name)

    if num_gpus > 1:
        motion_dirs = motion_dirs[gpu_id::num_gpus]
        logger.info("[GPU {}/{}] Assigned {} motion sequences".format(
            gpu_id, num_gpus, len(motion_dirs)))
    else:
        logger.info("Found {} motion sequences in {}".format(
            len(motion_dirs), render_root))

    pending = [d for d in motion_dirs if not is_completed(already_done.get(d.name))]
    logger.info("{} already done, {} pending.".format(
        len(motion_dirs) - len(pending), len(pending)))

    if backend == BACKEND_QWEN and num_workers > 1:
        logger.warning("Qwen backend does not support num_workers > 1; using 1.")
        num_workers = 1

    if vision_input == "video" and backend != BACKEND_QWEN:
        logger.warning("--vision_input video is qwen-only; "
                       "falling back to images for backend '{}'.".format(backend))
        vision_input = "images"

    if max_frames_per_view is None:
        max_frames_per_view = 0 if vision_input == "video" \
            else DEFAULT_MAX_FRAMES_PER_VIEW
    logger.info("vision_input={} max_frames_per_view={}".format(
        vision_input, max_frames_per_view or "uncapped"))

    max_size, jpeg_quality = image_params(backend, model_name or "")
    mime_type = frame_mime_type(max_size)
    logger.info("Frame loader: max_size={} jpeg_q={} mime={} "
                "downsample_rate={} max_frames_per_view={}".format(
                    max_size, jpeg_quality, mime_type, downsample_rate,
                    max_frames_per_view))

    state_lock = threading.Lock()
    n_ok = 0
    n_err = 0
    n_unsaved = 0

    def _process(motion_dir):
        motion_name = motion_dir.name
        try:
            # A partially rendered clip must never be captioned: the caption
            # would be frozen forever (is_completed() accepts any non-ERROR
            # string). Report it as a failure so a later run retries it once
            # rendering has finished.
            incomplete = render_incomplete_reason(str(motion_dir))
            if incomplete is not None:
                return motion_name, None, "incomplete render: {}".format(
                    incomplete)

            logger.info("Captioning [{}/{}]: {}".format(
                backend, vision_input, motion_name))
            video_fps = None
            if vision_input == "video":
                multiview_frames, video_fps = load_multiview_videos(
                    str(motion_dir),
                    downsample_rate=downsample_rate, views=views,
                    max_frames_per_view=max_frames_per_view,
                )
            else:
                multiview_frames = load_multiview_frames(
                    str(motion_dir),
                    downsample_rate=downsample_rate, views=views,
                    max_size=max_size, jpeg_quality=jpeg_quality,
                    max_frames_per_view=max_frames_per_view,
                )
            if not multiview_frames:
                return motion_name, None, "no frames"

            total_images = sum(len(f) for f in multiview_frames.values())
            logger.info("Total frames for VLM: {}".format(total_images))

            hint = hints.get(motion_name) if hints else None
            if backend == BACKEND_QWEN:
                caption = generate_qwen(
                    multiview_frames, motion_name, system_prompt,
                    model=qwen_model, processor=qwen_processor,
                    max_tokens=max_tokens, max_retries=max_retries,
                    hint=hint, video_fps=video_fps,
                )
            elif backend == BACKEND_GEMINI:
                caption = generate_gemini(
                    multiview_frames, motion_name, system_prompt,
                    model_name=model_name, api_key=api_key,
                    max_tokens=max_tokens, max_retries=max_retries,
                    mime_type=mime_type,
                    thinking_level=thinking_level,
                    media_resolution=media_resolution,
                    thinking_budget=thinking_budget,
                    hint=hint,
                )
            else:
                caption = generate_openai(
                    multiview_frames, motion_name, system_prompt,
                    model_name=model_name, api_key=api_key, base_url=base_url,
                    max_tokens=max_tokens, max_retries=max_retries,
                    mime_type=mime_type, hint=hint,
                )

            logger.info("Caption for '{}': {}".format(motion_name, caption))
            if caption:
                return motion_name, caption, None
            return motion_name, None, "empty response"
        except Exception as e:
            logger.error("Failed to caption {}: {}".format(motion_name, e))
            return motion_name, None, str(e)

    gpu_label = " GPU {}".format(gpu_id) if num_gpus > 1 else ""
    pbar = tqdm(total=len(pending), desc="Captioning{}".format(gpu_label),
                unit="seq", dynamic_ncols=True)

    def _record(motion_name, caption, error):
        nonlocal n_ok, n_err, n_unsaved
        with state_lock:
            if caption is not None:
                captions[motion_name] = caption
                n_ok += 1
                n_unsaved += 1
                # Rewriting the full JSON per caption is O(n^2) over a large
                # run; batch it (a crash loses at most SAVE_EVERY captions).
                if n_unsaved >= SAVE_EVERY:
                    save_json(captions, save_path)
                    n_unsaved = 0
            else:
                n_err += 1
                if motion_name not in failed_set:
                    failed_set.add(motion_name)
                    with open(failed_path, "a") as f:
                        f.write(motion_name + "\n")
                logger.warning("Skipped {} (not saved to JSON): {}".format(
                    motion_name, error))
            pbar.set_postfix_str("{} ok / {} err".format(n_ok, n_err),
                                 refresh=False)
        pbar.update(1)

    try:
        if num_workers <= 1:
            for d in pending:
                name, caption, error = _process(d)
                _record(name, caption, error)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(_process, d) for d in pending]
                try:
                    for fut in as_completed(futures):
                        name, caption, error = fut.result()
                        _record(name, caption, error)
                except KeyboardInterrupt:
                    # Every item is queued upfront; without cancel_futures the
                    # pool's shutdown would still run (and bill) all of them.
                    logger.warning("Interrupted — cancelling queued requests.")
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
    finally:
        with state_lock:
            if n_unsaved:
                save_json(captions, save_path)
                n_unsaved = 0

    pbar.close()
    logger.info("Done. {} ok / {} err this run; {} total saved to {}".format(
        n_ok, n_err, len(captions), save_path))
    if n_err:
        logger.info("Failed motion names appended to {}".format(failed_path))
    return captions


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Caption multi-view motion renders using a VLM "
                    "(local Qwen, OpenAI API, or Gemini API)."
    )

    parser.add_argument('--render_root', type=str,
                        help='Root directory for batch captioning all motions.')

    parser.add_argument('--output_json', type=str, default=None,
                        help='Output JSON file. '
                             'Defaults to <render_root>/motion_captions.json.')

    # ── Backend & task ──────────────────────────────────────────────────────
    parser.add_argument('--backend', type=str, default=None,
                        choices=VALID_BACKENDS,
                        help='VLM backend. Auto-detected from --model if '
                             'not specified (qwen/openai/gemini).')
    parser.add_argument('--task', type=str, required=True,
                        choices=sorted(TASK_PROMPTS.keys()),
                        help='Captioning task / system prompt to use.')
    parser.add_argument('--hints_json', type=str, default=None,
                        help='Optional JSON with per-clip reference labels '
                             'shown to the model as verify-against-frames '
                             'hints (e.g. dataset/raw/mixamo/'
                             'animation_motion_prompts.json). Keys may carry '
                             'a file extension; dict values use description '
                             'over prompt.')

    # ── Model ───────────────────────────────────────────────────────────────
    parser.add_argument('--model', type=str,
                        default='Qwen/Qwen3-VL-8B-Instruct',
                        help='Model name. For qwen: HuggingFace model ID '
                             '(default Qwen/Qwen3-VL-8B-Instruct; also e.g. '
                             'Qwen/Qwen3.5-9B). For openai: '
                             'API model (e.g., gpt-5-mini). For gemini: API model '
                             '(e.g., gemini-3-flash-preview).')

    # ── Qwen-specific ──────────────────────────────────────────────────────
    parser.add_argument('--torch_dtype', type=str, default=None,
                        choices=['float16', 'bfloat16'],
                        help='[qwen] Override model dtype (default: bfloat16).')
    parser.add_argument('--device_map', type=str, default='auto',
                        help='[qwen] Device placement strategy (default: auto).')
    parser.add_argument('--min_pixels', type=int, default=256 * 28 * 28,
                        help='[qwen] Min pixels per image '
                             '(default: 200704 = 256*28*28).')
    parser.add_argument('--max_pixels', type=int, default=256 * 28 * 28,
                        help='[qwen] Max pixels per image '
                             '(default: 200704 = 256*28*28).')

    # ── API-specific ────────────────────────────────────────────────────────
    parser.add_argument('--api_key', type=str, default=None,
                        help='[openai/gemini] API key (defaults to '
                             'OPENAI_API_KEY or GOOGLE_API_KEY env var).')
    parser.add_argument('--base_url', type=str, default=None,
                        help='[openai] Override API base URL.')
    parser.add_argument('--max_retries', type=int, default=MAX_RETRIES,
                        help='[openai/gemini] Max retry attempts for '
                             'empty responses.')

    # ── Gemini 3-specific ───────────────────────────────────────────────────
    parser.add_argument('--thinking_level', type=str, default='low',
                        choices=['minimal', 'low', 'medium', 'high'],
                        help='[gemini-3] Reasoning depth. Captioning works '
                             'well at "low" (default); raise to "medium" if '
                             'captions are noisy.')
    parser.add_argument('--media_resolution', type=str, default='medium',
                        choices=['low', 'medium', 'high', 'ultra_high'],
                        help='[gemini-3] Vision tokens per image. "medium" '
                             '(560 tok) is a good default for multi-view '
                             'frames; "high" (1120 tok) for fine details. '
                             '"ultra_high" is Gemini 3 Flash only.')
    parser.add_argument('--thinking_budget', type=int, default=0,
                        help='[gemini-2.5] Thinking token cap. 0 disables '
                             '(recommended for one-sentence captions); -1 '
                             'enables dynamic thinking; up to 24576.')

    # ── Common ──────────────────────────────────────────────────────────────
    parser.add_argument('--vision_input', type=str, default='video',
                        choices=['images', 'video'],
                        help="How frames reach the model. 'video' (default) "
                             "uses the qwen backend's native video path "
                             "(temporal patching + timestamps, ~half the "
                             "visual tokens; API backends fall back to "
                             "images); 'images' sends individually labeled "
                             "frames.")
    parser.add_argument('--views', type=str, nargs='+', default=None,
                        help='Subset of views to use (e.g., v000 v002). '
                             'Default: all views.')
    parser.add_argument('--downsample_rate', type=int, default=1,
                        help='Take every N-th frame per view (default 1 = '
                             'full 30 FPS; 2 -> 15 FPS).')
    parser.add_argument('--max_frames_per_view', type=int, default=None,
                        help='Cap on frames sent per view after '
                             '--downsample_rate (0 = no cap). Default: '
                             'no cap in video mode (temporal patching '
                             'keeps it cheap), {} in images mode.'
                        .format(DEFAULT_MAX_FRAMES_PER_VIEW))
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Max new tokens to generate.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Concurrent in-process workers for HTTP backends '
                             '(gemini/openai). Qwen ignores this.')

    add_multi_gpu_args(parser)

    return parser, parser.parse_args()


def main():
    parser, args = parse_args()

    # ── Merge mode ──────────────────────────────────────────────────────────
    if args.merge_shards:
        if not args.output_json:
            parser.error("--merge_shards requires --output_json")
        merge_shards(args.output_json, args.num_gpus)
        return

    if not args.render_root:
        parser.error("--render_root is required")

    # ── Resolve backend ─────────────────────────────────────────────────────
    backend = args.backend or detect_backend(args.model)

    task = args.task
    logger.info("Backend: {} | Model: {} | Task: {}".format(
        backend, args.model, task))

    system_prompt = get_prompt(task)

    # ── Multi-GPU is only for qwen (local model) ───────────────────────────
    num_gpus = args.num_gpus
    gpu_id = args.gpu_id
    if backend != BACKEND_QWEN and num_gpus > 1:
        logger.warning("Multi-GPU sharding is only supported for the qwen "
                       "backend. Ignoring --num_gpus/--gpu_id.")
        num_gpus = 1
        gpu_id = 0

    # ── Load qwen model (once) ──────────────────────────────────────────────
    qwen_model = None
    qwen_processor = None

    if backend == BACKEND_QWEN:
        gpu_id_for_model = gpu_id if num_gpus > 1 else None
        qwen_model, qwen_processor = load_qwen_model(
            args.model,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            gpu_id=gpu_id_for_model,
        )

    # ── Run ─────────────────────────────────────────────────────────────────
    caption_batch(
        render_root=args.render_root,
        backend=backend,
        system_prompt=system_prompt,
        output_json=args.output_json,
        downsample_rate=args.downsample_rate,
        max_frames_per_view=args.max_frames_per_view,
        max_tokens=args.max_tokens,
        views=args.views,
        hints=load_hints(args.hints_json),
        vision_input=args.vision_input,
        gpu_id=gpu_id, num_gpus=num_gpus,
        num_workers=args.num_workers,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
        thinking_level=args.thinking_level,
        media_resolution=args.media_resolution,
        thinking_budget=args.thinking_budget,
    )


if __name__ == "__main__":
    main()
