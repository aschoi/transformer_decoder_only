import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .rotaryEmbeddings import RotaryEmbedding, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    r"""
    Attention Module
    Multi-Head causal self-attention with RoPE
    Decoder-only
    """

    d_model: int
    num_attn_heads: int
    max_seq_len: int
    rope_baseFreq: float
    dropout_rate: float
    weights_init: str

    def __init__(
        self,
        d_model,
        num_attn_heads,
        max_seq_len=7000,
        rope_baseFreq=10_000.0,
        dropout_rate=0.0,
        weights_init='xavier'
    ) -> None:
        """
        Multi-Head Attention Module Constructor

        Args:
            d_model:         <int>    Model Dimension
            num_attn_heads:  <int>    Number of Attention Heads
            max_seq_len      <int>    Max seq len for the RoPE cache and causal mask buffer
            rope_baseFreq    <float>  RoPE base frequency
            dropout_rate:    <float>  Dropout rate
            weights_init:    <str>    Weights initialization strategy
        """

        super().__init__()
        assert d_model % num_attn_heads == 0, "d_model must be divisible by num_attn_heads"

        self.d_model = d_model
        self.num_attn_heads = num_attn_heads
        self.d_k = d_model // num_attn_heads
        self.dropout_rate = dropout_rate

        # Linear Projections for Query, Key, Value, and output
        self.nn_linearProj_w_Q = nn.Linear(d_model, d_model, bias=False)
        self.nn_linearProj_w_K = nn.Linear(d_model, d_model, bias=False)
        self.nn_linearProj_w_V = nn.Linear(d_model, d_model, bias=False)
        self.nn_linearProj_w_output = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.d_k, max_seq_len, rope_baseFreq)
        self.nn_dropout = nn.Dropout(dropout_rate)
        
        self._init_weights(weights_init)


    def _init_weights(
        self, 
        weights_init: str
    ) -> None:
        '''
        Initialize weights using Xavier Uniform Initialization 

        Args:
            weights_init:   <str>
        '''

        if (weights_init == 'xavier'):
            for nn_linear_proj in [self.nn_linearProj_w_Q, self.nn_linearProj_w_K, self.nn_linearProj_w_V, self.nn_linearProj_w_output]:
                nn.init.xavier_uniform_(nn_linear_proj.weight)
                if nn_linear_proj.bias is not None:
                    nn.init.constant_(nn_linear_proj.bias, 0)


    def forward(
        self,
        tnsr_X: torch.Tensor,
        tnsr_padding_mask: torch.Tensor | None=None,
        return_attn_weights: bool=False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward Propogation for Multi-head Attention.

        Args:
            tnsr_X:               <tensor>  shape: (batch_size, seq_len, d_model)
            tnsr_padding_mask:    <tensor>  shape: (batch_size, seq_len), 1 = keep, 0 = pad. Optional.
            return_attn_weights   <bool>    False means faster implementation, but less educational
        Return:
            tnsr_output:          <tensor>  shape: (batch_size, seq_len, d_model)
            tnsr_attn_weights:    <tensor>  shape: (batch_size, num_attn_heads, seq_len, seq_len) or None
        """

        batch_size, seq_len, _ = tnsr_X.shape

        tnsr_Q = self.nn_linearProj_w_Q(tnsr_X).view(batch_size, seq_len, self.num_attn_heads, self.d_k).transpose(1, 2)
        tnsr_K = self.nn_linearProj_w_K(tnsr_X).view(batch_size, seq_len, self.num_attn_heads, self.d_k).transpose(1, 2)
        tnsr_V = self.nn_linearProj_w_V(tnsr_X).view(batch_size, seq_len, self.num_attn_heads, self.d_k).transpose(1, 2)

        # Rotate Q/K by absolute position (RoPE) instead of adding a position vector to the embedding
        tnsr_cos, tnsr_sin = self.rotary_emb(seq_len, tnsr_X.device, tnsr_X.dtype)
        tnsr_Q, tnsr_K = apply_rotary_pos_emb(tnsr_Q, tnsr_K, tnsr_cos, tnsr_sin)

        if return_attn_weights: # for demo/education purposes
            # EXPLICIT attention mask creation. In the pre-training version, PyTorch has optimized version.
            # Combine causal mask with optional padding mask. Built fresh each call (cheap relative to
            # the attention matmuls) so it is never bounded by the max_seq_len passed at construction time.
            # shape: (1, 1, seq_len, seq_len)   This is the causal_mask, but will be combined with padding_mask if it exists
            tnsr_attn_mask = torch.tril(
                torch.ones(
                    seq_len, 
                    seq_len, 
                    dtype=torch.bool, 
                    device=tnsr_X.device
                )
            ).unsqueeze(0).unsqueeze(0)

            if tnsr_padding_mask is not None:
                tnsr_attn_mask = tnsr_attn_mask & tnsr_padding_mask[:, None, None, :].bool()      # shape: (batch_size, 1, 1, seq_len)

            tnsr_scores = torch.matmul(tnsr_Q, tnsr_K.transpose(-2, -1)) / math.sqrt(self.d_k)
            tnsr_scores = tnsr_scores.masked_fill(
                ~tnsr_attn_mask, 
                float("-inf")
            )

            tnsr_attn_weights = F.softmax(tnsr_scores, dim=-1)
            tnsr_attn_weights = self.nn_dropout(tnsr_attn_weights)
            tnsr_attn_output = torch.matmul(tnsr_attn_weights, tnsr_V)

        else:  # Don't return attention weights - for actual pretraining. Optimizing pretraining
            if tnsr_padding_mask is None:  # fastest / common training path
                tnsr_attn_output = F.scaled_dot_product_attention(
                    tnsr_Q,
                    tnsr_K,
                    tnsr_V,
                    attn_mask=None,
                    dropout_p=self.dropout_rate if self.training else 0.0,
                    is_causal=True
                )

            else:  # combine causal + attention mask
                tnsr_attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tnsr_X.device)).unsqueeze(0).unsqueeze(0)
                if tnsr_padding_mask is not None:
                    tnsr_attn_mask = tnsr_attn_mask & tnsr_padding_mask[:, None, None, :].bool()      # shape: (batch_size, 1, 1, seq_len)

                tnsr_attn_output = F.scaled_dot_product_attention(
                    tnsr_Q, 
                    tnsr_K, 
                    tnsr_V, 
                    attn_mask=tnsr_attn_mask,
                    dropout_p=(self.dropout_rate if self.training else 0.0),
                    is_causal=False
                )

            tnsr_attn_weights = None


        # Concatenate heads and apply output Linear Projection
        tnsr_attn_output = tnsr_attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        tnsr_output = self.nn_linearProj_w_output(tnsr_attn_output)

        return tnsr_output, tnsr_attn_weights

