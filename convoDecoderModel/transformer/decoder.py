import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .rms_norm import RMSNorm
from .ff_swiglu import SwiGLU
from .kv_cache import KVCache


class TransformerDecoderLayer(nn.Module):
    """
    Decoder-only Transformer layer.
    Pre-Norm RMSNorm -> Causal Self-Attention (RoPE)  -> Residual
    Pre-Norm RMSNorm -> SwiGLU                        -> Residual

    """

    d_model: int
    num_attn_heads: int
    d_ff: int
    max_seq_len: int
    rope_baseFreq: float
    dropout_rate: float

    def __init__(
        self,
        d_model,
        num_attn_heads,
        d_ff,
        max_seq_len=7000,
        rope_baseFreq=10_000.0,
        dropout_rate=0.0
    ) -> None:
        """
        Args:
            d_model:          <int>     Model dimension
            num_attn_heads:   <int>     Number of attention heads
            d_ff:             <int>     SwiGLU hidden dimension
            max_seq_len:      <int>     Max sequence length for RoPE 
            rope_baseFreq:    <float>   RoPE base frequency
            dropout_rate:     <float>   Dropout rate
        """
        super().__init__()

        self.sublayer1_norm = RMSNorm(d_model)
        self.sublayer1_attention = CausalSelfAttention(d_model, num_attn_heads, max_seq_len, rope_baseFreq, dropout_rate)
        self.sublayer1_nn_dropout = nn.Dropout(dropout_rate)

        self.sublayer2_norm = RMSNorm(d_model)
        self.sublayer2_ff = SwiGLU(d_model, d_ff)
        self.sublayer2_nn_dropout = nn.Dropout(dropout_rate)


    def forward(
        self,
        tnsr_X: torch.Tensor,
        tnsr_padding_mask: torch.Tensor | None = None,
        return_attn_weights: bool=False,
        kv_cache: KVCache | None=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            
            tnsr_X:               <tensor>  shape: (batch_size, seq_len, d_model)
            tnsr_padding_mask:    <tensor>  shape: (batch_size, seq_len). Optional.

        Return:
            tnsr_X:                      <tensor>  shape: (batch_size, seq_len, d_model)
            tnsr_causalSelfAttn_weights:      <tensor>  shape: (batch_size, num_attn_heads, seq_len, seq_len)
        """
        # 1) Pre-Norm Causal Self-Attention (RoPE) + Residual
        tnsr_normed1 = self.sublayer1_norm(tnsr_X)
        tnsr_attn_output, tnsr_attn_weights = self.sublayer1_attention(tnsr_normed1, tnsr_padding_mask, return_attn_weights, kv_cache)
        tnsr_X = tnsr_X + self.sublayer1_nn_dropout(tnsr_attn_output)

        # 2) Pre-Norm SwiGLU FFN + Residual
        tnsr_normed2 = self.sublayer2_norm(tnsr_X)
        tnsr_ff_output = self.sublayer2_ff(tnsr_normed2)
        tnsr_X = tnsr_X + self.sublayer2_nn_dropout(tnsr_ff_output)

        return tnsr_X, tnsr_attn_weights



class TransformerDecoder(nn.Module):
    """
    Stack of decoder-only Transformer layers (Pre-Norm RMSNorm + RoPE self-attn + SwiGLU),
    with a final RMSNorm before the output (the LM head, e.g. tied to the token embedding,
    is expected to live outside this stack).
    """

    num_layers: int
    d_model: int
    num_attn_heads: int
    d_ff: int
    max_seq_len: int
    rope_baseFreq: float
    dropout_rate: float

    def __init__(
        self,
        num_layers,
        d_model,
        num_attn_heads,
        d_ff,
        max_seq_len=7000,
        rope_baseFreq=10_000.0,
        dropout_rate=0.0
    ) -> None:
        """
        Args:
            num_layers:       <int>     Number of decoder layers
            d_model:          <int>     Model dimension
            num_attn_heads:   <int>     Number of attention heads
            d_ff:             <int>     SwiGLU hidden dimension
            max_seq_len:      <int>     Max sequence length for RoPE / causal mask cache
            rope_baseFreq:    <float>   RoPE base frequency
            dropout_rate:     <float>   Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.stack_transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, num_attn_heads, d_ff, max_seq_len, rope_baseFreq, dropout_rate, 
            ) for layer in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)


    def forward(
        self,
        tnsr_X: torch.Tensor,
        tnsr_padding_mask: torch.Tensor | None = None,
        return_attn_weights: bool=False,
        kv_caches: list[KVCache] | None=None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            tnsr_X:               <tensor>  shape: (batch_size, seq_len, d_model)
            tnsr_padding_mask:    <tensor>  shape: (batch_size, seq_len). Optional.
            return_attn_weights   <bool>    default does not return attn weights. True will return attention weights
        Return:
            tnsr_X:                        <tensor>  shape: (batch_size, seq_len, d_model)
            all_causalSelfAttn_weights:    list<tensor>
        """

        if kv_caches is not None:    
            if len(kv_caches) != len(self.stack_transformer_layers):
                raise ValueError("Number of KV caches must equal number of transformer layers")


        all_attn_weights = []

        for iLayer, decoderLayer in enumerate(self.stack_transformer_layers):

            layer_cache = None if kv_caches is None else kv_caches[iLayer]

            tnsr_X, tnsr_attn_weights = decoderLayer(tnsr_X, tnsr_padding_mask, return_attn_weights, layer_cache)

            if return_attn_weights:
                all_attn_weights.append(tnsr_attn_weights)


        tnsr_X = self.final_norm(tnsr_X)

        return tnsr_X, all_attn_weights
