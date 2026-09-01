"""VLM backend adapters for motion captioning.

One captioning request is the same everywhere — a system prompt plus an
interleaved [view header, frame label, image, ...] user turn — but each
backend speaks a different dialect. This module owns those dialects:

    qwen    — local Qwen3-VL / Qwen3.5 via HuggingFace transformers; PIL
              images inline, OOM-aware retries that halve the frame count.
    openai  — OpenAI-compatible chat API; base64 ``data:`` image URLs,
              GPT-5 vs GPT-4o parameter differences handled here.
    gemini  — Google Gemini API; raw image bytes as Parts, thinking and
              media-resolution controls for Gemini 2.5 / 3.

Each ``generate_*`` function takes the loaded multi-view frames and returns
a caption string ("" after all retries fail); the caller decides what to do
with failures. :func:`detect_backend` maps a model name to a backend and
:func:`image_params` picks the frame loader's resize/quality settings per
backend, so the orchestrator (:mod:`.caption_motion`) never needs
backend-specific knowledge beyond dispatching on the backend name.
"""

import base64
import os
import random
import time

from loguru import logger

from data_process.utils.vlm import (
    b64_to_pil,
    build_hint_footer,
    build_multiview_intro,
    build_view_header,
    caption_rejection_reason,
    gemini_client,
    is_retryable_error,
    openai_client,
    qwen_generate,
    uniform_subsample,
)


BACKEND_QWEN = "qwen"
BACKEND_OPENAI = "openai"
BACKEND_GEMINI = "gemini"
VALID_BACKENDS = [BACKEND_QWEN, BACKEND_OPENAI, BACKEND_GEMINI]

MAX_RETRIES = 3


def detect_backend(model_name):
    # type: (str) -> str
    """Auto-detect backend from model name.

    - Contains 'qwen' (case-insensitive) -> qwen
    - Starts with 'gemini' -> gemini
    - Everything else -> openai
    """
    if "qwen" in model_name.lower():
        return BACKEND_QWEN
    elif model_name.startswith("gemini"):
        return BACKEND_GEMINI
    else:
        return BACKEND_OPENAI


# Loader-side resize targets per (backend, model). Qwen relies on its
# AutoProcessor (min/max_pixels) for server-side resize; API backends pre-shrink
# to keep payloads small and, for GPT-5 patch vision, to cut tokens quadratically.
# GPT-5: ceil(w/32)²·mult — 384² → ~233 tok vs 415 at 512², q85 stays crisp.
# GPT-4o tile scheme with detail='low' is flat 85 tok — size only affects bandwidth.
# Gemini: server-side media_resolution sets tokens; 768 caps the inline upload.
def image_params(backend, model_name):
    # type: (str, str) -> tuple
    """Return ``(max_size, jpeg_quality)`` for the frame loader."""
    if backend == BACKEND_QWEN:
        return None, 85
    if backend == BACKEND_GEMINI:
        return 768, 85
    if model_name.startswith("gpt-5"):
        return 384, 85
    return 512, 80


# ─────────────────────────────────────────────────────────────────────────────
# Message building
# ─────────────────────────────────────────────────────────────────────────────

def _build_qwen_messages(multiview_frames, system_prompt, hint=None):
    # type: (Dict[str, List[str]], str, Optional[str]) -> List[Dict]
    """Build Qwen3-VL chat messages with PIL Images inline."""
    user_content = []
    user_content.append({
        "type": "text",
        "text": build_multiview_intro(),
    })

    for view_name, frames in multiview_frames.items():
        user_content.append({
            "type": "text",
            "text": build_view_header(view_name, len(frames)),
        })
        for idx, frame_b64 in enumerate(frames):
            user_content.append({
                "type": "text",
                "text": "[frame {}]".format(idx + 1),
            })
            user_content.append({
                "type": "image",
                "image": b64_to_pil(frame_b64),
            })

    footer = build_hint_footer(hint)
    if footer:
        user_content.append({"type": "text", "text": footer})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def _build_qwen_video_messages(multiview_videos, system_prompt, hint=None):
    # type: (Dict[str, List], str, Optional[str]) -> List[Dict]
    """Build Qwen3-VL chat messages with one native video item per view.

    Each view's PIL frame list rides the processor's video path (2-frame
    temporal patching + timestamp alignment) instead of being spelled out
    as individually labeled images — half the visual tokens and native
    temporal encoding for the same frames.
    """
    user_content = [{
        "type": "text",
        "text": build_multiview_intro(),
    }]
    for view_name, frames in multiview_videos.items():
        user_content.append({
            "type": "text",
            "text": build_view_header(view_name, len(frames)),
        })
        user_content.append({"type": "video", "video": frames})

    footer = build_hint_footer(hint)
    if footer:
        user_content.append({"type": "text", "text": footer})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def _build_openai_messages(multiview_frames, system_prompt, mime_type,
                           hint=None):
    # type: (Dict[str, List[str]], str, str, Optional[str]) -> List[Dict]
    """Build OpenAI-compatible chat messages with base64 image URLs.

    Frames are already resized + encoded by the loader; this just wraps them
    in the ``data:`` URL scheme.
    """
    user_content = []
    user_content.append({
        "type": "text",
        "text": build_multiview_intro(),
    })

    for view_name, frames in multiview_frames.items():
        user_content.append({
            "type": "text",
            "text": build_view_header(view_name, len(frames)),
        })
        for idx, frame_b64 in enumerate(frames):
            user_content.append({
                "type": "text",
                "text": "[frame {}]".format(idx + 1),
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:{};base64,{}".format(mime_type, frame_b64),
                    "detail": "low",
                },
            })

    footer = build_hint_footer(hint)
    if footer:
        user_content.append({"type": "text", "text": footer})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_qwen(multiview_frames, motion_name, system_prompt,
                  model, processor, max_tokens, max_retries=MAX_RETRIES,
                  hint=None, video_fps=None):
    # type: (Dict[str, List], str, str, object, object, int, int, Optional[str], Optional[Dict[str, float]]) -> str
    """Generate caption using a local Qwen3-VL / Qwen3.5 model.

    ``multiview_frames`` carries base64 PNG/JPEG strings in image mode, or
    PIL frame lists when ``video_fps`` is given — the latter selects the
    native video input path (one video item per view).

    Retries transient failures (OOM, empty/truncated/refusal output) up to
    max_retries times. An OOM halves the frames per view before retrying —
    resubmitting the identical image stack would OOM again deterministically.
    """
    import torch  # local import; keeps module light if never hit

    frames = multiview_frames
    video_mode = video_fps is not None

    def _assemble(f, fps_scale=1.0):
        if video_mode:
            msgs = _build_qwen_video_messages(f, system_prompt, hint=hint)
            fps = [video_fps[v] * fps_scale for v in f]
            return msgs, {"videos": [f[v] for v in f], "video_fps": fps}
        return _build_qwen_messages(f, system_prompt, hint=hint), {}

    fps_scale = 1.0
    messages, vid_kwargs = _assemble(frames)

    for attempt in range(1, max_retries + 1):
        try:
            # First attempt is greedy (reproducible); retries sample so a
            # rejected output gets a genuinely different second chance.
            text, num_tokens = qwen_generate(
                model, processor, messages, max_tokens=max_tokens,
                return_num_tokens=True, do_sample=attempt > 1,
                **vid_kwargs,
            )
            # Qwen exposes no finish_reason; hitting the cap means truncation.
            finish_reason = "length" if num_tokens >= max_tokens else "stop"
            reason = caption_rejection_reason(text, finish_reason)
            if reason is None:
                return text.strip()
            logger.warning("Attempt {}/{}: rejected Qwen output — {}".format(
                attempt, max_retries, reason))
        except torch.cuda.OutOfMemoryError as e:
            logger.warning("Attempt {}/{}: CUDA OOM: {}".format(
                attempt, max_retries, e))
            torch.cuda.empty_cache()
            frames = {
                view: uniform_subsample(f, max(1, len(f) // 2))
                for view, f in frames.items()
            }
            # Halving the frames doubles their temporal spacing.
            fps_scale *= 0.5
            logger.warning("Retrying with {} frames (halved per view)".format(
                sum(len(f) for f in frames.values())))
            messages, vid_kwargs = _assemble(frames, fps_scale=fps_scale)
        except Exception as e:
            logger.warning("Attempt {}/{}: Qwen error: {}".format(
                attempt, max_retries, e))

        if attempt < max_retries:
            time.sleep(1)

    logger.error("All {} Qwen attempts failed or returned empty for '{}'".format(
        max_retries, motion_name))
    return ""


def generate_openai(multiview_frames, motion_name, system_prompt,
                    model_name, api_key, base_url, max_tokens, max_retries,
                    mime_type, hint=None):
    # type: (Dict[str, List[str]], str, str, str, Optional[str], Optional[str], int, int, str, Optional[str]) -> str
    """Generate caption using OpenAI-compatible API."""
    client = openai_client(
        api_key or os.environ.get("OPENAI_API_KEY"), base_url,
    )

    messages = _build_openai_messages(
        multiview_frames, system_prompt, mime_type=mime_type, hint=hint,
    )

    # gpt-5+ reasoning models use max_completion_tokens and don't support
    # temperature; older models (gpt-4o, etc.) use max_tokens.
    is_gpt5 = "gpt-5" in model_name
    create_kwargs = dict(model=model_name, messages=messages)
    if is_gpt5:
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["max_tokens"] = max_tokens
        create_kwargs["temperature"] = 0.3

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**create_kwargs)

            choice = response.choices[0]
            content = choice.message.content
            refusal = getattr(choice.message, "refusal", None)
            logger.debug("Attempt {}: finish_reason={}, refusal={}".format(
                attempt, choice.finish_reason, refusal))

            if refusal:
                logger.warning("Attempt {}/{}: model refused: {}".format(
                    attempt, max_retries, refusal))
            else:
                reason = caption_rejection_reason(content, choice.finish_reason)
                if reason is None:
                    return content.strip()
                logger.warning("Attempt {}/{}: rejected response — {}".format(
                    attempt, max_retries, reason))
        except Exception as e:
            logger.warning("Attempt {}/{}: OpenAI API error: {}".format(
                attempt, max_retries, e))

        if attempt < max_retries:
            time.sleep(2)

    logger.error("All {} attempts failed or returned empty for '{}'".format(
        max_retries, motion_name))
    return ""


def generate_gemini(multiview_frames, motion_name, system_prompt,
                    model_name, api_key, max_tokens, max_retries,
                    mime_type="image/png",
                    thinking_level="low", media_resolution="medium",
                    thinking_budget=0, hint=None):
    # type: (Dict[str, List[str]], str, str, str, Optional[str], int, int, str, str, str, int, Optional[str]) -> str
    """Generate caption using Google Gemini API.

    Gemini 3 uses ``thinking_level`` (enum) + ``media_resolution``; Gemini 2.5
    uses ``thinking_budget`` (int token cap, 0 disables, -1 dynamic). For a
    one-sentence caption, dynamic thinking is wasteful — default is 0.
    """
    from google.genai import types

    is_gemini3 = "gemini-3" in model_name
    is_gemini25 = "gemini-2.5" in model_name

    # Gemini 3's thinking_config draws reasoning tokens from the same budget
    # as the final answer. At thinking_level='low' the reasoning pass commonly
    # burns 200–500 tokens, so <1024 total risks returning an empty answer.
    if is_gemini3 and max_tokens < 1024:
        logger.warning(
            "max_tokens={} is tight for Gemini 3 + thinking; raising to 1024."
            .format(max_tokens))
        max_tokens = 1024

    client = gemini_client(
        api_key or os.environ.get("GOOGLE_API_KEY"), is_gemini3
    )

    contents = [types.Part.from_text(text=build_multiview_intro())]
    for view_name, frames in multiview_frames.items():
        contents.append(types.Part.from_text(
            text=build_view_header(view_name, len(frames)),
        ))
        for idx, frame_b64 in enumerate(frames):
            contents.append(types.Part.from_text(
                text="[frame {}]".format(idx + 1),
            ))
            contents.append(types.Part.from_bytes(
                data=base64.b64decode(frame_b64),
                mime_type=mime_type,
            ))
    footer = build_hint_footer(hint)
    if footer:
        contents.append(types.Part.from_text(text=footer))

    config_kwargs = dict(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.0,
        stop_sequences=["\n\n"],
    )
    if is_gemini3:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
        )
        config_kwargs["media_resolution"] = "MEDIA_RESOLUTION_{}".format(
            media_resolution.upper()
        )
    elif is_gemini25:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
        )
    config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            candidates = getattr(response, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) \
                if candidates else None
            reason = caption_rejection_reason(response.text, finish_reason)
            if reason is None:
                return response.text.strip()
            # A rejected response (empty, truncated at MAX_TOKENS, refusal)
            # is worth another shot; fall through to the backoff.
            logger.warning("Attempt {}/{}: rejected Gemini response — {}".format(
                attempt, max_retries, reason))
        except Exception as e:
            retryable = is_retryable_error(e)
            level = "warning" if retryable else "error"
            getattr(logger, level)(
                "Attempt {}/{}: Gemini API error ({}): {}".format(
                    attempt, max_retries,
                    "retryable" if retryable else "fatal", e))
            if not retryable:
                break

        if attempt < max_retries:
            # Exponential backoff with jitter: 5s, 15s, 45s, … capped at 60s.
            delay = min(60.0, 5.0 * (3 ** (attempt - 1))) + random.uniform(0, 2)
            time.sleep(delay)

    logger.error("All {} attempts failed or returned empty for '{}'".format(
        max_retries, motion_name))
    return ""
