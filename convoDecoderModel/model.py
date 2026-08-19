from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

from .transformer.decoder import TransformerDecoder
from .transformer.kv_cache import KVCache


@dataclass(frozen=True, slots=True)
class AcaiModelConfig:
    r"""
    Configuration for Decoder-only Transformer model.
    """

    vocab_size: int
    d_model: int
    num_layers: int
    num_attn_heads: int
    d_ff: int
    max_seq_len: int
    rope_baseFreq: float
    dropout_rate: float
    pad_token_id: int | None
    tie_word_embeddings: bool
    init_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AcaiModelConfig":
        return cls(**values)

    def save_json(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, sort_keys=True)
            f.write("\n")

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("Vocab size must be positve")
        if self.d_model <= 0:
            raise ValueError("d_model must be positve")
        if self.num_layers <= 0:
            raise ValueError("number of decoder layers must be positve")
        if self.num_attn_heads <= 0:
            raise ValueError("number of attention heads must be positve")
        if self.d_ff <= 0:
            raise ValueError("dimensions of ffnn must be positve")
        if self.max_seq_len <= 0:
            raise ValueError("max sequence length must be positve")
        if self.dropout_rate < 0 or 1.0 <= self.dropout_rate:
            raise ValueError("dropout rate must be (greater-than/equal to 0) AND (less than 1)")
        if self.rope_baseFreq <= 0:
            raise ValueError("RoPE base freq must be positve")
        if self.init_std <= 0:
            raise ValueError("intial std must be positve")
        if self.d_model % self.num_attn_heads != 0:
            raise ValueError("dimensions of model must be divisible by number of attention heads")
        if (self.d_model // self.num_attn_heads) % 2 != 0:
            raise ValueError("dimension within an attention head must be even for RoPE")


class CausalOutput(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor | None
    attention_weights: list[torch.Tensor] | None




class AcaiTransformer(nn.Module):
    """
    Class definition for the decoder only transformer model
    """

    def __init__(
        self, 
        config: AcaiModelConfig
    ) -> None:
        
        super().__init__()
        self.config = config
        self.nn_tokenEmbedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id
        )
        self.nn_embeddingDropout = nn.Dropout(config.dropout_rate)

        self.acai_decoder = TransformerDecoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_attn_heads=config.num_attn_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            rope_baseFreq=config.rope_baseFreq,
            dropout_rate=config.dropout_rate
        )

        self.acai_lmHead = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        self._gradient_checkpointing = False

        self._init_weights()
        
        if config.tie_word_embeddings:
            self.acai_lmHead.weight = self.nn_tokenEmbedding.weight

        


    def _init_weights(self) -> None:

        def init_module(module: nn.Module) -> None:

            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

        self.apply(init_module)

        residual_std = self.config.init_std / math.sqrt(2.0 * self.config.num_layers)
        for layer in self.acai_decoder.stack_transformer_layers:
            nn.init.normal_(
                layer.sublayer1_attention.nn_linearProj_w_output.weight,
                mean=0.0,
                std=residual_std
            )
            nn.init.normal_(
                layer.sublayer2_ff.down_proj.weight,
                mean=0.0,
                std=residual_std
            )


    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True
    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing = False

    def num_params(
        self,
        *,
        exclude_embeds: bool = False
    ) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if exclude_embeds:
            total -= self.nn_tokenEmbedding.weight.numel()
        return total


    def create_kv_caches(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> list[KVCache]:
        
        return [
            KVCache(
                batch_size=batch_size,
                num_kv_heads=layer.sublayer1_attention.num_attn_heads,
                max_seq_len=max_seq_len,
                head_dim=layer.sublayer1_attention.d_k,
                device=device,
                dtype=dtype                
            )
            for layer in self.acai_decoder.stack_transformer_layers
        ]


    def forward(
        self,
        tnsr_inputIds: torch.Tensor,
        *,
        tnsr_labels: torch.Tensor | None = None,
        tnsr_attn_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
        return_logits: bool =True,
        kv_caches: list[KVCache] | None=None
    ) -> CausalOutput:
        """
        """

        if self.training and kv_caches is not None:
            raise RuntimeError("KV caching cannot be used during training.") 
        if tnsr_inputIds.ndim != 2:
            raise ValueError("input ids must have shape (batch, sequence length)")
        if tnsr_inputIds.dtype != torch.int64:
            raise TypeError("input ids must use torch.long dtype (torch.int64)")

        batch_size, seq_len = tnsr_inputIds.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum sequence length: {self.config.max_seq_len}")

        if tnsr_attn_mask is not None:
            if tnsr_attn_mask.shape != (batch_size, seq_len):
                raise ValueError("attention mask's first two dimensions must be same as input ids")
            tnsr_attn_mask = tnsr_attn_mask.to(device=tnsr_inputIds.device, dtype=torch.bool)

        if (tnsr_labels is not None) and (tnsr_labels.shape != tnsr_inputIds.shape):
            raise ValueError("labels must have the same shape as input ids")


        embedded_ids = self.nn_tokenEmbedding(tnsr_inputIds)
        embedded_ids = self.nn_embeddingDropout(embedded_ids)

        # For debugging/educational purposes
        if return_attn_weights:
            output_features, attn_weights = self.acai_decoder(embedded_ids, tnsr_attn_mask, return_attn_weights, kv_caches)

        elif (self.training and self._gradient_checkpointing):
            # Checkpoing + memory effiecient path
            features = embedded_ids
            for decoderLayer in self.acai_decoder.stack_transformer_layers:

                def run_layer(states: torch.Tensor, currLayer: nn.Module = decoderLayer) -> torch.Tensor:
                    output, _ = currLayer(states, tnsr_attn_mask, False)
                    return output

                features = checkpoint(run_layer, features, use_reentrant=False)

            output_features = self.acai_decoder.final_norm(features)
            attn_weights = None
                # Inference. Checkpointing uncessessary. 
        else:
            output_features, _ = self.acai_decoder(embedded_ids, tnsr_attn_mask, return_attn_weights=False, kv_caches=kv_caches)
            attn_weights = None

        # Language Model Projection
        logits = self.acai_lmHead(output_features)

        loss = None
        if tnsr_labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size).float(),
                tnsr_labels.reshape(-1),
                ignore_index=-100
            )

        return CausalOutput(
            loss = loss,
            logits = logits if return_logits else None,
            attention_weights=attn_weights if return_attn_weights else None
        )




