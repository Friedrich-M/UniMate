"""Stage 2b — VLM captioning of multi-view renders.

Entry points (also available through ``data_process/scripts/run_caption_*.sh``):
    caption_motion     — per-clip motion captions from ``render/<dataset>/``.
    classify_category  — per-asset body-plan categories from T-pose grids.

Supporting modules: ``backends`` (Qwen / OpenAI / Gemini adapters) and
``prompts`` (system prompts + task registry). Shared VLM plumbing lives in
:mod:`data_process.utils.vlm`.
"""
