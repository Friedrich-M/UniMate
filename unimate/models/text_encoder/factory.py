"""Unified text encoder factory.

Provides :func:`create_text_encoder` which returns a text encoder with a
consistent interface (``tokenize`` + ``forward`` -> ``(B, D)`` mean-pooled
embeddings) for any supported backend (T5, CLIP, BERT).

Joint-name cleaning is handled upstream by the data-processing pipeline
(the ``clean_joint_names`` field of ``cond.npy``), so text encoders receive
already-cleaned text and encode it as-is.
"""

from typing import Dict, List, Optional, Union

import torch
from torch import nn

from unimate.models.text_encoder.t5 import T5Conditioner
from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)


# ---------------------------------------------------------------------------
# Embedding dimension lookup tables
# ---------------------------------------------------------------------------

# T5 dims live in `T5Conditioner.MODELS_DIMS` (single source of truth).
T5_MODEL_DIM = T5Conditioner.MODELS_DIMS

CLIP_MODEL_DIM = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}

BERT_MODEL_DIM = {
    "distilbert/distilbert-base-uncased": 768,
    "distilbert-base-uncased": 768,
    "bert-base-uncased": 768,
    "bert-large-uncased": 1024,
}


# ---------------------------------------------------------------------------
# CLIP Conditioner
# ---------------------------------------------------------------------------

class CLIPConditioner(nn.Module):
    """CLIP text encoder with unified ``tokenize`` / ``forward`` interface."""

    def __init__(self, version: str = "ViT-B/32", device: str = "cpu",
                 pool: bool = True):
        super().__init__()
        if not pool:
            # CLIP's encode_text returns the EOS-token projection only —
            # exposing per-token features needs a custom forward through the
            # text transformer. Not implemented; raise instead of silently
            # returning pooled features.
            raise NotImplementedError(
                "CLIPConditioner only supports pool=True. Use T5 or BERT for "
                "token-level features."
            )
        import clip as clip_lib
        self.device = device
        self.version = version
        self.pool = pool
        self.clip_model, _ = clip_lib.load(version, device=device, jit=False)
        clip_lib.model.convert_weights(self.clip_model)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.dim = CLIP_MODEL_DIM.get(version, 512)
        self._clip_lib = clip_lib

    def tokenize(self, x: Union[str, List[Optional[str]]]) -> Dict[str, torch.Tensor]:
        if isinstance(x, str):
            x = [x]
        entries = [xi if xi is not None else "" for xi in x]
        tokens = self._clip_lib.tokenize(entries).to(self.device)
        return {"tokens": tokens}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            features = self.clip_model.encode_text(inputs["tokens"])  # (B, D)
        return features.float()


# ---------------------------------------------------------------------------
# BERT Conditioner
# ---------------------------------------------------------------------------

class BERTConditioner(nn.Module):
    """BERT text encoder with unified ``tokenize`` / ``forward`` interface."""

    def __init__(self, model_path: str = "distilbert/distilbert-base-uncased",
                 device: str = "cpu", pool: bool = True):
        super().__init__()
        import os
        from transformers import AutoTokenizer, AutoModel, logging
        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = device
        self.pool = pool
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModel.from_pretrained(model_path).to(device)
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.dim = self.text_model.config.hidden_size

    def tokenize(self, x: Union[str, List[Optional[str]]]) -> Dict[str, torch.Tensor]:
        if isinstance(x, str):
            x = [x]
        entries = [xi if xi is not None else "" for xi in x]
        inputs = self.tokenizer(entries, return_tensors="pt", padding=True).to(self.device)
        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return ``(B, D)`` mean-pooled features (default) or raw ``(B, T, D)``
        token-level features when ``self.pool`` is False — caller can read
        the padding mask from ``inputs["attention_mask"]``."""
        with torch.no_grad():
            hidden = self.text_model(**inputs).last_hidden_state  # (B, T, D)
        if not self.pool:
            return hidden
        mask = inputs["attention_mask"]
        token_count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        embeds = (hidden * mask.unsqueeze(-1).float()).sum(dim=-2) / token_count
        return embeds


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_text_encoder_dim(encoder_type: str, encoder_version: str) -> int:
    """Return the embedding dimension for a given encoder type and version."""
    _dim_tables = {
        "t5": T5_MODEL_DIM,
        "clip": CLIP_MODEL_DIM,
        "bert": BERT_MODEL_DIM,
    }
    table = _dim_tables.get(encoder_type)
    if table is None:
        raise ValueError(f"Unknown text_encoder_type: {encoder_type}")
    if encoder_version not in table:
        raise ValueError(
            f"Unknown {encoder_type} version '{encoder_version}'. "
            f"Known versions: {list(table.keys())}"
        )
    return table[encoder_version]


def create_text_encoder(
    encoder_type: str,
    encoder_version: str,
    device: str = "cpu",
    pool: bool = True,
):
    """Create a text encoder with unified interface.

    All returned encoders support::

        tokens = encoder.tokenize(texts)
        embeddings = encoder(tokens)
            # pool=True  -> (B, D) mean-pooled
            # pool=False -> (B, T, D) per-token (T5/BERT only)

    Args:
        encoder_type: One of 'clip', 'bert', 't5'.
        encoder_version: Model identifier (e.g. 't5-base', 'ViT-B/32').
        device: Target device.
        pool: If True (default), the encoder mean-pools over non-padding
            tokens. If False, returns raw token-level features ``(B, T, D)``.
            Not supported for CLIP (raises ``NotImplementedError``).

    Returns:
        A text encoder module.
    """
    if encoder_type == "t5":
        logger.info(f"Creating T5 text encoder: {encoder_version} (pool={pool})")
        return T5Conditioner(
            name=encoder_version,
            finetune=False,
            word_dropout=0.0,
            normalize_text=False,
            device=device,
            pool=pool,
        )
    elif encoder_type == "clip":
        logger.info(f"Creating CLIP text encoder: {encoder_version} (pool={pool})")
        return CLIPConditioner(version=encoder_version, device=device, pool=pool)
    elif encoder_type == "bert":
        logger.info(f"Creating BERT text encoder: {encoder_version} (pool={pool})")
        return BERTConditioner(model_path=encoder_version, device=device, pool=pool)
    else:
        raise ValueError(f"Unknown text_encoder_type: {encoder_type}")
