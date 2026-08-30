"""Shared LLM scaffolding for the ``joint_annotation/*_llm.py`` entry points.

Every LLM-assisted joint-annotation script (joint-name cleaning, name
correction, face-joint selection) needs the same plumbing: backend detection
from the model name, a local HuggingFace causal-LM loader, one generate
function per backend (local / OpenAI-compatible), response-noise stripping, a
per-rig SIGALRM timeout, and simple JSON persistence. :class:`LLMClient`
bundles the backend state so task scripts only own their prompts, parsers
and per-rig orchestration.
"""

import json
import os
import re
import signal
import tempfile

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_LOCAL = "local"
BACKEND_OPENAI = "openai"
VALID_BACKENDS = [BACKEND_LOCAL, BACKEND_OPENAI]

MAX_RETRIES = 4
# gpt-5-mini is the alternative once OPENAI credits are topped up; DeepSeek
# think-low reached quality parity (0 side errors, 99.6% format compliance,
# A/B 2026-08-30) at a fraction of the cost.
DEFAULT_MODEL = "deepseek-v4-flash"

# DeepSeek's API speaks the OpenAI protocol, so ``deepseek-*`` models ride the
# OpenAI backend with this base URL and DEEPSEEK_API_KEY (see generate_openai).
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def detect_backend(model_name):
    # type: (str) -> str
    """Infer the backend from a model name (``gpt-*`` / ``deepseek-*`` /
    HF id)."""
    if model_name.startswith(("gpt-", "o1", "o3", "o4", "deepseek")):
        return BACKEND_OPENAI
    return BACKEND_LOCAL


# ─────────────────────────────────────────────────────────────────────────────
# Local LLM loading & generation
# ─────────────────────────────────────────────────────────────────────────────

def load_local_model(model_name, device_map="auto", torch_dtype=None):
    """Load a text-only causal LM + tokenizer from HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16

    logger.info("Loading LLM: {} (dtype={}, device_map={})".format(
        model_name, dtype, device_map))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def generate_local(model, tokenizer, system_prompt, user_prompt, max_tokens):
    """Run a single chat completion through a local causal LM.

    Qwen3 and other "thinking" models emit long ``<think>...</think>`` blocks
    by default, which easily exhausts the token budget before the answer is
    produced; ``enable_thinking=False`` is passed when the chat template
    supports it. Sampling follows the Qwen3 model-card recommendation for
    non-thinking mode (temperature=0.7, top_p=0.8, top_k=20) — greedy
    decoding is documented to cause endless repetitions.
    """
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    trimmed = generated[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(trimmed, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# API backends
# ─────────────────────────────────────────────────────────────────────────────

def generate_openai(system_prompt, user_prompt, model_name, api_key, base_url,
                    max_tokens, reasoning_effort=None):
    from openai import OpenAI

    is_deepseek = model_name.startswith("deepseek")
    if api_key is None:
        key_var = "DEEPSEEK_API_KEY" if is_deepseek else "OPENAI_API_KEY"
        api_key = os.environ.get(key_var)
        if api_key is None:
            # Fail loudly: with api_key=None the OpenAI client silently falls
            # back to OPENAI_API_KEY, which sends the wrong vendor's key to a
            # DeepSeek endpoint (and produces a confusing 401).
            raise RuntimeError(
                "{} is not set (and no --api_key given) for model {!r}.".format(
                    key_var, model_name))
    if base_url is None and is_deepseek:
        base_url = DEEPSEEK_BASE_URL
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = dict(model=model_name, messages=messages)
    is_gpt5 = model_name.startswith("gpt-5") and "chat" not in model_name
    is_o_series = model_name.startswith(("o1", "o3", "o4"))
    if is_gpt5 or is_o_series:
        # Reasoning models reject max_tokens and non-default temperature;
        # max_completion_tokens counts reasoning + visible output together.
        kwargs["max_completion_tokens"] = max_tokens
        if is_gpt5:
            # The default effort burns hidden thinking tokens on every call
            # and the joint-annotation tasks are classification, not
            # multi-step reasoning — force the lowest effort. The original
            # gpt-5 family calls it "minimal"; gpt-5.1+ dropped "minimal"
            # for "none" and errors on the old value. o-series models keep
            # their default effort ("minimal"/"none" don't exist there, and
            # o1-mini rejects the parameter entirely).
            versioned = re.match(r"gpt-5\.\d", model_name)
            kwargs["reasoning_effort"] = reasoning_effort or (
                "none" if versioned else "minimal")
    elif is_deepseek:
        # DeepSeek V4 thinks by default (effort 'high') and the reasoning
        # tokens count against max_tokens. Effort 'none' disables thinking —
        # but V4-Flash WITHOUT thinking drifts on long batched arrays
        # (systematic label/side misalignment on ~10% of truebones rigs,
        # A/B'd 2026-08-30), so default to the cheapest thinking tier.
        # Thinking mode ignores temperature.
        kwargs["reasoning_effort"] = reasoning_effort or "low"
        if kwargs["reasoning_effort"] == "none":
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0.0
        else:
            # Reasoning shares the max_tokens budget; a small task budget
            # (512 for face selection) gets eaten by thought and returns
            # EMPTY content with finish_reason='length'. Billing is per
            # token used, so a generous floor costs nothing extra.
            kwargs["max_tokens"] = max(max_tokens, 8192)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as err:
        msg = str(err)
        if any(marker in msg for marker in (
                "insufficient_quota", "credit_balance_exhausted",
                "invalid_api_key", "Authentication Fails",
                "account_deactivated")):
            raise FatalLLMError(msg) from err
        raise
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if not content and choice.finish_reason == "length":
        # Thinking consumed the whole budget — retrying with identical
        # parameters cannot succeed; surface an actionable error instead.
        raise RuntimeError(
            "empty content with finish_reason='length': the reasoning "
            "budget is exhausted — raise --max_tokens or lower "
            "--reasoning_effort (model {}).".format(model_name))
    return content


# ─────────────────────────────────────────────────────────────────────────────
# Unified client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """One text-in / text-out generate() over a local or OpenAI-compatible model.

    Args:
        model_name: OpenAI model (``gpt-5-mini``), DeepSeek model
            (``deepseek-v4-flash``) or HuggingFace model id for the local
            backend (``Qwen/Qwen3-8B``).
        backend: One of :data:`VALID_BACKENDS`; auto-detected when ``None``.
        api_key: API key for the OpenAI-compatible backends (defaults to
            the env vars).
        base_url: OpenAI-compatible base URL override.
        device_map, torch_dtype: Local-backend loading knobs.
    """

    def __init__(self, model_name, backend=None, api_key=None, base_url=None,
                 device_map="auto", torch_dtype=None, reasoning_effort=None):
        self.model_name = model_name
        self.backend = backend or detect_backend(model_name)
        if self.backend not in VALID_BACKENDS:
            raise ValueError("Unknown backend {!r}; expected one of {}".format(
                self.backend, VALID_BACKENDS))
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self._model = self._tokenizer = None
        if self.backend == BACKEND_LOCAL:
            self._model, self._tokenizer = load_local_model(
                model_name, device_map=device_map, torch_dtype=torch_dtype)

    @classmethod
    def from_args(cls, args):
        """Build a client from the namespace produced by :func:`add_llm_args`."""
        return cls(
            args.model, backend=args.backend,
            api_key=args.api_key, base_url=args.base_url,
            device_map=args.device_map, torch_dtype=args.torch_dtype,
            reasoning_effort=getattr(args, "reasoning_effort", None),
        )

    def generate(self, system_prompt, user_prompt, max_tokens):
        # type: (str, str, int) -> str
        if self.backend == BACKEND_LOCAL:
            return generate_local(self._model, self._tokenizer,
                                  system_prompt, user_prompt, max_tokens)
        return generate_openai(system_prompt, user_prompt,
                               self.model_name, self.api_key, self.base_url,
                               max_tokens, reasoning_effort=self.reasoning_effort)

    def __repr__(self):
        return "LLMClient(backend={!r}, model={!r})".format(self.backend, self.model_name)


def add_llm_args(parser, default_model=DEFAULT_MODEL, default_max_tokens=2048):
    """Add the backend / model / retry flags shared by every ``*_llm.py`` CLI."""
    group = parser.add_argument_group("LLM backend")
    group.add_argument("--backend", type=str, default=None, choices=VALID_BACKENDS,
                       help="LLM backend. Auto-detected from --model if unset.")
    group.add_argument("--model", type=str, default=default_model,
                       help="OpenAI model (e.g. gpt-5-mini), DeepSeek model (e.g. "
                            "deepseek-v4-flash), or HF model id for the local "
                            "backend (e.g. Qwen/Qwen3-8B).")
    group.add_argument("--max_tokens", type=int, default=default_max_tokens,
                       help="Max new tokens per LLM call.")
    group.add_argument("--max_retries", type=int, default=MAX_RETRIES,
                       help="Max LLM retries per rig before falling back.")
    group.add_argument("--torch_dtype", type=str, default=None, choices=["float16", "bfloat16"],
                       help="[local] Model dtype override (default: bfloat16).")
    group.add_argument("--device_map", type=str, default="auto",
                       help="[local] Device placement strategy (default: auto).")
    group.add_argument("--api_key", type=str, default=None,
                       help="[openai] Defaults to OPENAI_API_KEY / DEEPSEEK_API_KEY, "
                            "matched to the model.")
    group.add_argument("--base_url", type=str, default=None,
                       help="[openai] Override API base URL (deepseek-* models default "
                            "to the DeepSeek endpoint; use with --backend openai for "
                            "any other OpenAI-compatible server, e.g. a local vLLM).")
    group.add_argument("--reasoning_effort", type=str, default=None,
                       help="[openai/deepseek] Reasoning effort override. Defaults: "
                            "gpt-5 'minimal'/'none' (version-dependent); deepseek "
                            "'low' ('none' disables thinking but degrades batched "
                            "labeling accuracy).")
    return group


# ─────────────────────────────────────────────────────────────────────────────
# Response hygiene
# ─────────────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_llm_noise(text):
    # type: (str) -> str
    """Remove ``<think>`` blocks and markdown fences from an LLM response."""
    cleaned = _THINK_RE.sub("", text)
    return _FENCE_RE.sub("", cleaned).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Per-rig timeout (SIGALRM, Unix main thread only)
# ─────────────────────────────────────────────────────────────────────────────

class FatalLLMError(Exception):
    """Non-retryable backend failure (exhausted credits, bad API key).

    Retry loops must re-raise this instead of swallowing it: falling back to
    the rule-based cleaner on a quota outage would silently degrade EVERY
    rig of a batch run to rule-based output."""


class RigTimeout(Exception):
    """Raised by the per-rig SIGALRM handler to abort an over-budget rig."""


def _alarm_handler(signum, frame):
    raise RigTimeout()


def install_rig_alarm(timeout_seconds):
    # type: (int) -> Optional[object]
    """Arm a SIGALRM for *timeout_seconds*; returns the previous handler, or
    ``None`` when no alarm was installed (disabled, unsupported, or not in
    the main thread)."""
    if not timeout_seconds or timeout_seconds <= 0:
        return None
    if not hasattr(signal, "SIGALRM"):
        return None
    try:
        prev = signal.signal(signal.SIGALRM, _alarm_handler)
    except ValueError:
        return None
    signal.alarm(int(timeout_seconds))
    return prev


def uninstall_rig_alarm(prev_handler):
    if prev_handler is None:
        return
    signal.alarm(0)
    signal.signal(signal.SIGALRM, prev_handler)


# ─────────────────────────────────────────────────────────────────────────────
# JSON persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    # type: (str) -> Dict
    """Load a JSON dict if *path* exists, else return ``{}``."""
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        logger.info("Loaded {} existing entries from {}".format(len(data), path))
        return data
    return {}


def save_json(data, path):
    # type: (Dict, str) -> None
    """Write *data* as JSON to *path*, atomically.

    The batch drivers re-save the whole file after every rig (3676 times for
    an Objaverse sweep), so a plain truncate-rewrite leaves a truncated,
    unparseable file if the process is killed mid-write — and the two batch
    stages keep no ``.bak``. Write a temp file in the same directory and
    ``os.replace`` it into place instead, so readers only ever see the old
    or the new file.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_id_list(path):
    # type: (str) -> List[str]
    """Read one id per line, skipping blanks and ``#`` comments."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def append_id_list(path, ids):
    # type: (str, Iterable[str]) -> None
    with open(path, "a") as f:
        for rig_id in ids:
            f.write(rig_id + "\n")
