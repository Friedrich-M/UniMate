# Adapted from the T5 conditioner in AudioCraft (Meta Platforms, Inc., MIT License):
# https://github.com/facebookresearch/audiocraft
"""T5-based text conditioner.

Provides :class:`T5Conditioner` which encodes text (joint names, captions,
categories) into dense vectors via a frozen (or fine-tuned) T5 encoder.
Joint-name cleaning is handled upstream during data preprocessing.
"""

import hashlib
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union, no_type_check
import warnings
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5EncoderModel, T5Tokenizer

from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)

# Type aliases
ConditionType = Tuple[torch.Tensor, torch.Tensor]  # (embedding, mask)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TorchAutocast:
    """Context manager wrapping ``torch.autocast`` with graceful no-op."""

    def __init__(self, enabled: bool, *args, **kwargs):
        self.autocast = torch.autocast(*args, **kwargs) if enabled else None

    def __enter__(self):
        if self.autocast is not None:
            self.autocast.__enter__()

    def __exit__(self, *args, **kwargs):
        if self.autocast is not None:
            self.autocast.__exit__(*args, **kwargs)


def _hash_trick(word: str, vocab_size: int) -> int:
    """Map *word* to an index in ``[0, vocab_size)`` via SHA-256."""
    h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return h % vocab_size


def _length_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert a 1-D lengths tensor to a boolean mask.

    Example: ``[3, 5]`` → ``[[1,1,1,0,0],[1,1,1,1,1]]``
    """
    assert lengths.dim() == 1
    final = lengths.max().item() if max_len is None else max_len
    final = max(final, 1)
    return torch.arange(final, device=lengths.device)[None, :] < lengths[:, None]


# ---------------------------------------------------------------------------
# Tokenizers (used only when normalize_text=True)
# ---------------------------------------------------------------------------

class WhiteSpaceTokenizer:
    """Hash-based tokenizer with optional lemmatization and stopword removal.

    Used internally by :class:`T5Conditioner` when ``normalize_text=True``.
    """

    PUNCTUATION = "?:!.,;"

    def __init__(
        self,
        n_bins: int,
        pad_idx: int = 0,
        language: str = "en_core_web_sm",
        lemma: bool = True,
        stopwords: bool = True,
    ):
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            import spacy
            self.nlp = spacy.load(language)
        except IOError:
            import spacy
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @no_type_check
    def __call__(
        self,
        texts: List[Optional[str]],
        return_text: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from num2words import num2words

        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            text = re.sub(r"(\d+)", lambda m: num2words(int(m.group(0))), text)
            text = self.nlp(text)
            if self.stopwords:
                text = [w for w in text if not w.is_stop]
            text = [w for w in text if w.text not in self.PUNCTUATION]
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]

            texts[i] = " ".join(text)
            lengths.append(len(text))
            tokens = torch.Tensor([_hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = _length_to_mask(torch.IntTensor(lengths)).int()
        padded = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded, mask, texts
        return padded, mask


# ---------------------------------------------------------------------------
# Base conditioner
# ---------------------------------------------------------------------------

class BaseConditioner(nn.Module):
    """Abstract base for text conditioners."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def tokenize(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def forward(self, inputs: Any) -> ConditionType:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# T5 Conditioner
# ---------------------------------------------------------------------------

class T5Conditioner(BaseConditioner):
    """Encode text into dense vectors via a frozen (or fine-tuned) T5 encoder.

    The T5 encoder output is mean-pooled over non-padding tokens to produce
    a single ``(B, D)`` embedding per input string.

    Args:
        name: HuggingFace T5 model identifier.
        finetune: Whether to keep T5 gradients during training.
        device: Target device (``"cuda"`` or ``"cpu"``).
        autocast_dtype: Dtype string for mixed precision (or None to disable).
        word_dropout: Probability of dropping each word during training.
        normalize_text: If True, apply lemmatization / stopword removal.
        pool: If True (default), mean-pool over non-padding tokens and return
            ``(B, D)``. If False, return raw token-level features
            ``(B, T, D)`` — caller can read the padding mask from
            ``inputs["attention_mask"]`` returned by :meth:`tokenize`.
    """

    # Supported T5 / Flan-T5 models and their encoder hidden sizes; also
    # consumed by ``text_encoder.factory`` so the table lives in one place.
    # Flan-T5's large variants are published as ``-xl`` / ``-xxl``.
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
        self,
        name: str,
        finetune: bool,
        device: str,
        autocast_dtype: Optional[str] = "float32",
        word_dropout: float = 0.0,
        normalize_text: bool = False,
        pool: bool = True,
    ):
        if name not in self.MODELS_DIMS:
            raise ValueError(
                f"Unknown T5 model: {name!r}. Known: {list(self.MODELS_DIMS)}"
            )
        super().__init__(self.MODELS_DIMS[name])
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        self.normalize_text = normalize_text
        self.pool = pool

        # --- Autocast setup ---
        if autocast_dtype is None or device == "cpu":
            self.autocast = TorchAutocast(enabled=False)
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 autocast dtype: {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=device, dtype=dtype)

        # --- Load T5 (suppress noisy HF warnings) ---
        prev_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(prev_level)

        if finetune:
            self.t5 = t5
        else:
            # Store outside nn.Module state so it's excluded from checkpoints
            self.__dict__["t5"] = t5.to(device)

        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    # ------------------------------------------------------------------
    # Tokenize & forward
    # ------------------------------------------------------------------

    def tokenize(
        self,
        x: Union[str, List[Optional[str]]],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text via T5 BPE.

        Args:
            x: Single string or list of strings to encode.
        """
        if isinstance(x, str):
            x = [x]

        entries = [xi if xi is not None else "" for xi in x]

        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)

        if self.word_dropout > 0.0 and self.training:
            entries = [
                " ".join(w for w in e.split() if random.random() >= self.word_dropout)
                for e in entries
            ]

        empty_idx = torch.LongTensor([i for i, e in enumerate(entries) if e == ""])

        inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True).to(self.device)
        inputs["attention_mask"][empty_idx, :] = 0
        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode tokenized inputs.

        Returns:
            ``(B, D)`` mean-pooled embeddings if ``self.pool`` is True,
            otherwise raw token-level features ``(B, T, D)`` (padded; mask
            is in ``inputs["attention_mask"]``).
        """
        mask = inputs["attention_mask"]
        with torch.set_grad_enabled(self.finetune), self.autocast:
            hidden = self.t5(**inputs).last_hidden_state  # (B, T, D)
            if not self.pool:
                return hidden
            # Mean-pool over non-padding tokens (avoid division by zero)
            token_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
            embeds = (hidden * mask.unsqueeze(-1)).sum(dim=-2) / token_count
        return embeds
