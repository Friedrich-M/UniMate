"""Shared utilities for the VLM captioning stage.

Frame loading / sampling / base64 encoding, multi-view prompt text,
render-directory discovery + completeness checks, resumable JSON I/O,
multi-GPU sharding, Qwen3-VL loading + generation, caption validation, and
the cached API clients used by :mod:`data_process.vlm_caption`.
"""

import base64
import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from loguru import logger
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Camera azimuth of each view (v000..v003) relative to v000, about the
# vertical axis. The renderer's on-disk metadata names these views
# front/back/left/right, but those are WORLD-space camera positions — the
# subject's orientation is arbitrary (normalize_scene never rotates it), so
# the words carry no information about the subject and would only bias a
# VLM into reading 'front view' as the subject's front. The model is
# therefore shown neutral azimuth labels instead.
VIEW_REL_AZIMUTHS = [0, 180, 270, 90]
NUM_VIEWS = len(VIEW_REL_AZIMUTHS)

# Rendered clips run to a few hundred frames per view; a fixed stride alone
# still sends 160 images at the median and 400 worst-case in a single request.
# Cap the per-view budget and sample uniformly across the clip instead.
# 64 frames x 4 views = 256 images/request: measured 31.6 GiB peak on
# Qwen3-VL-8B (A100-80GB), well within budget; the OOM retry ladder in
# generate_qwen halves this per attempt if a device is tighter.
DEFAULT_MAX_FRAMES_PER_VIEW = 64


# ─────────────────────────────────────────────────────────────────────────────
# Frame loading
# ─────────────────────────────────────────────────────────────────────────────

# Renders are saved with a transparent film; a plain .convert("RGB") drops
# alpha onto black, hiding dark subjects. Composite onto light grey instead
# (close to the render's own world-background tone).
RGBA_BACKGROUND = (235, 235, 235)


def flatten_rgba(img, background=RGBA_BACKGROUND):
    # type: (Image.Image, tuple) -> Image.Image
    """Composite a possibly-transparent image onto an opaque RGB background."""
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, background)
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def uniform_subsample(items, max_items):
    # type: (List, Optional[int]) -> List
    """Uniformly pick at most ``max_items`` entries, keeping first and last.

    Used both to cap frames per view and to shrink a request after a CUDA
    OOM. A non-positive or None ``max_items`` means "no cap".
    """
    n = len(items)
    if not max_items or max_items <= 0 or n <= max_items:
        return list(items)
    if max_items == 1:
        return [items[0]]
    # round() spreads the picks evenly over [0, n-1] and pins both ends, so
    # the prompt's "frame 1 is the start, last frame is the end" holds.
    idxs = sorted({int(round(i * (n - 1) / (max_items - 1)))
                   for i in range(max_items)})
    return [items[i] for i in idxs]


def select_frame_paths(paths, downsample_rate=1, max_frames=None):
    # type: (List[Path], int, Optional[int]) -> List[Path]
    """Choose which frames of one view to send to the VLM.

    ``downsample_rate`` applies the legacy fixed stride first (the last
    frame is re-appended — a bare ``[::rate]`` drops it whenever the count
    is not a multiple of the stride), then ``max_frames`` uniformly samples
    the remainder down to that budget.
    """
    if not paths:
        return []
    selected = list(paths)
    if downsample_rate > 1:
        selected = selected[::downsample_rate]
        if selected[-1] != paths[-1]:
            selected.append(paths[-1])
    return uniform_subsample(selected, max_frames)


def load_frames_as_base64(view_dir, max_size=None, jpeg_quality=85,
                          downsample_rate=1, max_frames=None):
    # type: (str, Optional[int], int, int, Optional[int]) -> List[str]
    """Load PNG frames from a view directory as base64 strings.

    Frame selection (stride + ``max_frames`` uniform cap, see
    :func:`select_frame_paths`) happens at the path level before decoding,
    so discarded frames are never read. When ``max_size`` is set, frames are
    resized so the longest side is at most ``max_size`` px and re-encoded as
    JPEG at ``jpeg_quality`` (10-20x payload shrink, quadratic savings in
    GPT-5 patch tokens). When ``max_size`` is None, raw PNG bytes pass
    through.
    """
    paths = select_frame_paths(
        sorted(Path(view_dir).glob("*.png")),
        downsample_rate=downsample_rate, max_frames=max_frames,
    )
    result = []
    for frame_path in paths:
        if max_size is None:
            with open(frame_path, "rb") as f:
                result.append(base64.b64encode(f.read()).decode("utf-8"))
            continue
        img = flatten_rgba(Image.open(frame_path))
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_size:
            scale = max_size / long_side
            img = img.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.LANCZOS,
            )
        buf = BytesIO()
        img.save(buf, "JPEG", quality=jpeg_quality, optimize=True)
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


def frame_mime_type(max_size):
    # type: (Optional[int]) -> str
    """Mime type produced by ``load_frames_as_base64`` for a given ``max_size``."""
    return "image/jpeg" if max_size is not None else "image/png"


def _view_label_from_id(view_id):
    # type: (str) -> str
    # Label is keyed by the numeric index in the view id (v000→azimuth 0°),
    # NOT by iteration order — so subsetting --views keeps labels correct.
    try:
        idx = int(view_id.lstrip("v"))
    except ValueError:
        return view_id
    if 0 <= idx < len(VIEW_REL_AZIMUTHS):
        return "camera azimuth {}°".format(VIEW_REL_AZIMUTHS[idx])
    return "view_{}".format(idx)


def load_multiview_frames(
    render_dir,
    downsample_rate=2,
    views=None,
    max_size=None,
    jpeg_quality=85,
    max_frames_per_view=DEFAULT_MAX_FRAMES_PER_VIEW,
):
    # type: (str, int, Optional[List[str]], Optional[int], int, Optional[int]) -> Dict[str, List[str]]
    """Load and downsample frames from every view of a rendered motion.

    Returns {"v000 (camera azimuth 0°)": [b64, …], …}. Bytes are PNG when
    ``max_size`` is None, JPEG otherwise (see ``frame_mime_type``). At most
    ``max_frames_per_view`` frames per view are kept (0/None = no cap).

    Raises:
        ValueError: a requested view directory is missing or holds no
            frames. Callers gate on :func:`is_render_complete` first, so
            this means the render is broken — captioning a subset of the
            views would silently freeze a partial-render caption.
    """
    render_path = Path(render_dir)
    if views is None:
        views = sorted([d.name for d in render_path.iterdir()
                        if d.is_dir() and d.name.startswith("v")])

    result = {}
    for view_id in views:
        view_dir = render_path / view_id
        if not view_dir.exists():
            raise ValueError("View directory not found: {}".format(view_dir))
        frames = load_frames_as_base64(
            str(view_dir), max_size=max_size, jpeg_quality=jpeg_quality,
            downsample_rate=downsample_rate, max_frames=max_frames_per_view,
        )
        if not frames:
            raise ValueError("No frames found in {}".format(view_dir))
        view_label = _view_label_from_id(view_id)
        result["{} ({})".format(view_id, view_label)] = frames
        logger.info("Loaded {} ({}): {} frames (downsample_rate={}, "
                    "max_frames_per_view={}, max_size={})".format(
                        view_id, view_label, len(frames), downsample_rate,
                        max_frames_per_view, max_size))

    return result


RENDER_SOURCE_FPS = 30.0  # fallback when a view's metadata JSON is absent


def _view_source_fps(render_path, view_id):
    # type: (Path, str) -> float
    """Source fps of one view, from the renderer's ``v00X.json`` metadata."""
    meta = render_path / "{}.json".format(view_id)
    try:
        with open(meta) as f:
            return float(json.load(f)["fps"])
    except (OSError, KeyError, ValueError):
        return RENDER_SOURCE_FPS


def load_multiview_videos(
    render_dir,
    downsample_rate=2,
    views=None,
    max_frames_per_view=DEFAULT_MAX_FRAMES_PER_VIEW,
):
    # type: (str, int, Optional[List[str]], Optional[int]) -> Tuple[Dict[str, List], Dict[str, float]]
    """Load each view's frame PNGs as PIL lists for native video input.

    Same discovery, downsampling and per-view cap as
    :func:`load_multiview_frames`, but returns raw PIL frames (qwen video
    mode feeds them to the video processor, which does its own resize and
    2-frame temporal patching) plus each view's effective fps after
    sampling, so timestamps stay truthful.

    Returns:
        ``({view_key: [PIL.Image, ...]}, {view_key: effective_fps})``.
    """
    render_path = Path(render_dir)
    if views is None:
        views = sorted([d.name for d in render_path.iterdir()
                        if d.is_dir() and d.name.startswith("v")])

    videos, fps_map = {}, {}
    for view_id in views:
        view_dir = render_path / view_id
        if not view_dir.exists():
            raise ValueError("View directory not found: {}".format(view_dir))
        frame_paths = sorted(
            p for p in view_dir.iterdir() if p.suffix.lower() == ".png")
        frame_paths = frame_paths[::downsample_rate]
        n_downsampled = len(frame_paths)
        frame_paths = uniform_subsample(frame_paths, max_frames_per_view)
        if not frame_paths:
            raise ValueError("No frames found in {}".format(view_dir))
        fps = _view_source_fps(render_path, view_id) / downsample_rate
        if n_downsampled:
            fps *= len(frame_paths) / float(n_downsampled)
        key = "{} ({})".format(view_id, _view_label_from_id(view_id))
        videos[key] = [flatten_rgba(Image.open(str(p))) for p in frame_paths]
        fps_map[key] = fps
        logger.info("Loaded {} as video: {} frames @ {:.1f} fps".format(
            key, len(videos[key]), fps))
    return videos, fps_map


def b64_to_pil(b64_str):
    # type: (str) -> Image.Image
    """Convert a base64-encoded PNG string to an opaque RGB PIL Image."""
    return flatten_rgba(Image.open(BytesIO(base64.b64decode(b64_str))))


# ─────────────────────────────────────────────────────────────────────────────
# Multiview prompt text (shared by API and local backends)
# ─────────────────────────────────────────────────────────────────────────────

def build_multiview_intro():
    # type: () -> str
    """Build the intro text for a multiview caption prompt.

    The clip directory name is never shown to the model — dataset clip
    names are (or contain) the action label itself (mixamo:
    ``Acknowledging``; objaverse: ``{hex}-{Action}``; truebones:
    ``{Species}-{Action}``), so embedding one would leak the answer or a
    species word. Per-clip reference labels ride
    :func:`build_hint_footer` after the views instead.
    """
    intro = (
        "Below are multi-view video frames of a single motion clip. "
        "Each view shows the same motion from a different camera angle. "
        "Frames within each view are in chronological order (frame 1 is the "
        "start, last frame is the end). Analyze the temporal progression "
        "across all views and describe the motion.\n"
    )
    return intro


def build_hint_footer(hint):
    # type: (Optional[str]) -> Optional[str]
    """Closing text carrying the per-clip reference label, placed AFTER the
    view sections so the frames are read unprimed and the label sits next
    to the generation point (image-first VQA layout; how much weight it
    gets is defined by the task's system prompt)."""
    if not hint:
        return None
    return ("Reference label from the dataset catalogue: '{}'.".format(hint))


def build_view_header(view_name, num_frames):
    # type: (str, int) -> str
    """Build the header text for a single view section."""
    return "\n=== {} (frames 1-{}, chronological order) ===".format(
        view_name, num_frames)


# ─────────────────────────────────────────────────────────────────────────────
# Directory discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_motion_dirs(render_root):
    # type: (str) -> List[Path]
    """Find all motion directories under a root (must contain view subdirs).

    A motion directory is one that has at least one child directory whose
    name starts with "v" (e.g., v000, v001). Partially rendered clips are
    intentionally still returned so callers can report them as (retryable)
    failures instead of silently dropping them — use
    :func:`render_incomplete_reason` to gate captioning.

    Returns:
        Sorted list of Path objects.
    """
    render_root = Path(render_root)
    return sorted([
        d for d in render_root.iterdir()
        if d.is_dir() and any(
            sub.is_dir() and sub.name.startswith("v")
            for sub in d.iterdir()
        )
    ])


def render_incomplete_reason(render_dir, num_views=NUM_VIEWS):
    # type: (str, int) -> Optional[str]
    """Return None when a clip's multi-view render is complete, else why not.

    Mirrors ``data_process.utils.blender_render.is_render_complete``:
    ``num_views`` view directories (v000..) with their camera metadata JSONs
    (v000.json..), each holding the same non-zero number of PNGs. Views
    render sequentially, so an interrupted render always leaves either a
    missing view directory or a view with fewer frames than the first.
    Reimplemented here (rather than imported) because ``blender_render``
    imports ``bpy``, which the captioning stage must not require.
    """
    render_path = Path(render_dir)
    if not render_path.is_dir():
        return "not a directory: {}".format(render_dir)

    frame_counts = []
    for view_idx in range(num_views):
        view_name = "v{:03d}".format(view_idx)
        if not (render_path / "{}.json".format(view_name)).is_file():
            return "missing camera metadata {}.json".format(view_name)
        view_dir = render_path / view_name
        if not view_dir.is_dir():
            return "missing view directory {}".format(view_name)
        frame_counts.append(len(list(view_dir.glob("*.png"))))

    if frame_counts[0] == 0:
        return "view v000 has no frames"
    if len(set(frame_counts)) != 1:
        return "unequal frame counts across views: {}".format(
            ", ".join("v{:03d}={}".format(i, c)
                      for i, c in enumerate(frame_counts)))
    return None


def is_render_complete(render_dir, num_views=NUM_VIEWS):
    # type: (str, int) -> bool
    """True when every expected view of a clip rendered to the same length."""
    return render_incomplete_reason(render_dir, num_views=num_views) is None


# ─────────────────────────────────────────────────────────────────────────────
# JSON I/O (resumable batch processing)
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    # type: (str) -> Dict
    """Load a JSON file, returning an empty dict if it doesn't exist."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded {} existing entries from {}".format(len(data), path))
        return data
    return {}


def save_json(data, path):
    # type: (Dict, str) -> None
    """Atomically save a dict to a JSON file with indentation."""
    tmp = "{}.tmp.{}".format(path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def is_completed(result):
    # type: (Optional[str]) -> bool
    """Check if a result string represents a successful completion."""
    return bool(result) and not result.startswith("ERROR:")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU sharding
# ─────────────────────────────────────────────────────────────────────────────

def shard_output_path(output_json, gpu_id):
    # type: (str, int) -> str
    """Return the shard-specific output path for a given GPU."""
    base, ext = os.path.splitext(output_json)
    return "{}.shard{}{}".format(base, gpu_id, ext)


def find_shard_paths(output_json):
    # type: (str) -> List[str]
    """Every ``<base>.shard<N><ext>`` file sitting next to ``output_json``.

    Globbing (rather than trusting a ``num_gpus`` argument) means a merge
    run launched with fewer GPUs than the run that produced the shards can
    no longer silently drop the extra shards.
    """
    base, ext = os.path.splitext(output_json)
    pattern = "{}.shard*{}".format(os.path.basename(base), ext)
    paths = Path(base).parent.glob(pattern)

    def _gid(path):
        # type: (Path) -> Tuple[int, str]
        tail = path.name[len(os.path.basename(base)) + len(".shard"):]
        digits = tail[:-len(ext)] if ext else tail
        return (int(digits), path.name) if digits.isdigit() else (2 ** 31, path.name)

    return [str(p) for p in sorted(paths, key=_gid)]


def merge_shards(output_json, num_gpus=None):
    # type: (str, Optional[int]) -> Dict[str, str]
    """Merge per-GPU shard files into a single output JSON.

    Loads every ``output_json.shard*`` file found next to ``output_json``,
    merges them on top of the existing ``output_json`` contents, and writes
    the combined result back. ``num_gpus`` is only used to warn when fewer
    shards than expected are present.

    Returns:
        The merged dict.
    """
    merged = {}  # type: Dict[str, str]

    if os.path.exists(output_json):
        with open(output_json) as f:
            merged.update(json.load(f))
        logger.info("Loaded {} existing entries from {}".format(
            len(merged), output_json))

    shard_paths = find_shard_paths(output_json)
    for spath in shard_paths:
        with open(spath) as f:
            shard_data = json.load(f)
        merged.update(shard_data)
        logger.info("Merged shard {} ({} entries)".format(
            spath, len(shard_data)))

    if not shard_paths:
        logger.warning("No shard files found for {}".format(output_json))
    elif num_gpus is not None and len(shard_paths) < num_gpus:
        logger.warning("Found {} shard files but --num_gpus={}".format(
            len(shard_paths), num_gpus))

    save_json(merged, output_json)
    logger.info("Merged {} shards -> {} total entries in {}".format(
        len(shard_paths), len(merged), output_json))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Qwen model loading & generation
# ─────────────────────────────────────────────────────────────────────────────

def load_qwen_model(
    model_name,
    device_map="auto",
    torch_dtype=None,
    min_pixels=256 * 28 * 28,
    max_pixels=256 * 28 * 28,
    gpu_id=None,
):
    # type: (str, str, Optional[str], int, int, Optional[int]) -> Tuple
    """Load Qwen3-VL model and processor.

    Args:
        model_name: HuggingFace model ID or local path
            (e.g., "Qwen/Qwen3-VL-8B-Instruct").
        device_map: Device placement strategy (default "auto").
        torch_dtype: Override dtype ("float16", "bfloat16", or None for auto).
        min_pixels: Minimum pixels per image (default 200704 = 256*28*28).
            Lower = fewer visual tokens = faster + less memory.
        max_pixels: Maximum pixels per image (default 200704 = 256*28*28).
            The Qwen3-VL default is 1843200 (1280*1440) which is very
            expensive for multi-image inputs. 256*28*28 yields ~256 tokens
            per image, a good balance for rendered motion frames.
        gpu_id: If set, load the model on this specific GPU (overrides
            device_map). Used for multi-GPU data-parallel captioning.

    Returns:
        (model, processor) tuple.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16
    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16

    if gpu_id is not None:
        device_map = "cuda:{}".format(gpu_id)

    # flash_attn is an optional compiled extra; fall back to PyTorch SDPA
    # (still fused attention) when it isn't installed rather than failing.
    try:
        import flash_attn  # noqa: F401
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = "sdpa"

    logger.info("Loading model: {} (dtype={}, device_map={}, attn={})".format(
        model_name, dtype, device_map, attn_implementation))

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    logger.info("Model loaded (min_pixels={}, max_pixels={}, "
                "~{} visual tokens per image)".format(
                    min_pixels, max_pixels, max_pixels // (28 * 28)))
    return model, processor


def qwen_generate(model, processor, messages, max_tokens=300,
                  return_num_tokens=False, do_sample=False,
                  videos=None, video_fps=None):
    # type: (object, object, List[Dict], int, bool, bool, Optional[List], Optional[List[float]]) -> Union[str, Tuple[str, int]]
    """Run Qwen3-VL generation on chat messages and return decoded text.

    Decoding is greedy by default: the shipped Qwen generation_config has
    ``do_sample=True``, which makes captions drift between identical runs
    (chirality flips, stray adverbs). Callers enable ``do_sample`` on
    retry attempts only, so a rejected output gets a genuinely different
    second chance while the first pass stays reproducible — mirroring the
    fixed temperatures of the API backends.

    Args:
        model: Loaded Qwen3-VL model.
        processor: Loaded Qwen3-VL processor.
        messages: Chat messages in Qwen format (with PIL Images inline).
        max_tokens: Maximum new tokens to generate.
        return_num_tokens: Also return the number of generated tokens, so
            callers can detect truncation (``n == max_tokens``).
        do_sample: Sample (temperature 0.7) instead of greedy decoding.
        videos: Native video inputs, one ``[PIL.Image, ...]`` frame list per
            ``{"type": "video"}`` item in *messages*. When given, the chat
            template is rendered to text and the frames go through the
            processor's video path (2-frame temporal patching, timestamp
            alignment) instead of the per-image path.
        video_fps: Per-video effective fps matching *videos*, so the
            processor's timestamps reflect the true frame spacing.

    Returns:
        Decoded output string (stripped), or ``(text, num_new_tokens)`` when
        ``return_num_tokens`` is set.
    """
    if videos is None:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        video_kwargs = {"fps": video_fps} if video_fps else {}
        try:
            inputs = processor(text=[text], videos=videos,
                               return_tensors="pt", **video_kwargs)
        except TypeError:
            # Older processors without an fps kwarg.
            inputs = processor(text=[text], videos=videos,
                               return_tensors="pt")
    inputs.pop("token_type_ids", None)
    logger.debug("Qwen prompt tokens: {}".format(inputs["input_ids"].shape[-1]))
    inputs = inputs.to(model.device)

    gen_kwargs = dict(max_new_tokens=max_tokens)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None,
                          top_k=None)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if return_num_tokens:
        return text, len(generated_ids_trimmed[0])
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Qwen argparse helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_qwen_model_args(parser):
    """Add common Qwen model CLI arguments to an argparse parser."""
    parser.add_argument('--model', type=str,
                        default='Qwen/Qwen3-VL-8B-Instruct',
                        help='HuggingFace model ID or local path '
                             '(e.g., Qwen/Qwen3-VL-8B-Instruct, '
                             'Qwen/Qwen3-VL-72B-Instruct).')
    parser.add_argument('--torch_dtype', type=str, default=None,
                        choices=['float16', 'bfloat16'],
                        help='Override model dtype (default: bfloat16).')
    parser.add_argument('--device_map', type=str, default='auto',
                        help='Device placement strategy (default: auto).')
    parser.add_argument('--min_pixels', type=int, default=256 * 28 * 28,
                        help='Min pixels per image (default: 200704 = 256*28*28).')
    parser.add_argument('--max_pixels', type=int, default=256 * 28 * 28,
                        help='Max pixels per image (default: 200704 = 256*28*28).')


def add_multi_gpu_args(parser):
    """Add multi-GPU sharding CLI arguments to an argparse parser."""
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Total number of GPUs for data-parallel captioning.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU index for this process (0-indexed).')
    parser.add_argument('--merge_shards', action='store_true',
                        help='Merge per-GPU shard files into a single '
                             'output JSON and exit.')


# ─────────────────────────────────────────────────────────────────────────────
# Caption validation
# ─────────────────────────────────────────────────────────────────────────────

# finish_reason values meaning "stopped at the token cap" — the caption is
# truncated mid-sentence and must never be frozen into the output JSON.
TRUNCATED_FINISH_REASONS = ("length", "max_tokens")

MIN_CAPTION_CHARS = 8

# Matched against the START of the response only, so a legitimate caption
# that happens to contain one of these words is never rejected.
REFUSAL_PREFIXES = (
    "i'm sorry", "i am sorry", "sorry,", "sorry.", "i cannot", "i can't",
    "i can not", "i'm unable", "i am unable", "i'm not able", "i am not able",
    "unable to", "as an ai", "i apologize", "there is no image",
    "no image", "i don't see", "i do not see",
)


def caption_rejection_reason(text, finish_reason=None):
    # type: (Optional[str], object) -> Optional[str]
    """Return None when *text* is a usable caption, else why it is rejected.

    Rejects empty/whitespace output, truncation at the token cap (OpenAI's
    ``finish_reason='length'``, Gemini's ``MAX_TOKENS``, Qwen hitting
    ``max_new_tokens``) and plain-text refusals. Deliberately conservative:
    anything else — including an oddly worded caption — is accepted, since
    a rejected caption costs another API call.
    """
    if not text or not text.strip():
        return "empty response (finish_reason={})".format(finish_reason)

    if finish_reason is not None:
        # Gemini reports an enum (types.FinishReason.MAX_TOKENS), OpenAI a str.
        token = str(finish_reason).rsplit(".", 1)[-1].lower()
        if token in TRUNCATED_FINISH_REASONS:
            return "truncated at the token cap (finish_reason={})".format(
                finish_reason)

    stripped = text.strip()
    if len(stripped) < MIN_CAPTION_CHARS:
        return "caption too short: {!r}".format(stripped)

    lowered = stripped.lower().lstrip("\"'*` ")
    if lowered.startswith(REFUSAL_PREFIXES):
        return "looks like a refusal: {!r}".format(stripped[:120])

    return None


# ─────────────────────────────────────────────────────────────────────────────
# API clients + retry classification
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_HTTP_TIMEOUT_MS = 90_000
RETRYABLE_ERROR_TOKENS = (
    "429", "503", "500", "504",
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
)


@lru_cache(maxsize=4)
def openai_client(api_key, base_url):
    """Cache one thread-safe client per (api_key, base_url) tuple.

    A fresh ``OpenAI()`` per request leaks its connection pool (and the file
    descriptors behind it) across a 10k-clip run.
    """
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


@lru_cache(maxsize=4)
def gemini_client(api_key, is_gemini3):
    """Cache one thread-safe Client per (api_key, api_version) tuple."""
    from google import genai

    http_options = {"timeout": GEMINI_HTTP_TIMEOUT_MS}
    if is_gemini3:
        http_options["api_version"] = "v1alpha"
    return genai.Client(api_key=api_key, http_options=http_options)


def is_retryable_error(exc):
    """True for transient VLM API failures worth retrying (rate limit / 5xx)."""
    msg = str(exc)
    return any(tok in msg for tok in RETRYABLE_ERROR_TOKENS)
