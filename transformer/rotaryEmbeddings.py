import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    class definition for Rotary Position Embedding (RoPE)

    Precomputes cos & sin tables for a given head dimension and sequence length.
    Position is applied directly to Q, K INSIDE attention.
    """
    
    d_head: int
    max_seq_len: int
    rope_baseFreq: float

    def __init__(
        self,
        d_head,
        max_seq_len=7000,
        rope_baseFreq=10_000.0
    ) -> None:
        """
        Args:
            d_head:         <int>    Dimension of each attention head 
            max_seq_len:    <int>    Sequence length to precompute cos/sin tables for
            rope_baseFreq:  <float>  RoPE base frequency
        """
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"

        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.rope_baseFreq = rope_baseFreq

         # tnsr_inv_freq shape: (d_head / 2)
        tnsr_inv_freq = 1.0 / (
            rope_baseFreq**(
                torch.arange(0, d_head, 2, dtype=torch.float32) / d_head
            )
        )

        self.register_buffer('tnsr_inv_freq', tnsr_inv_freq, persistent=False)

        tnsr_cos, tnsr_sin = self._build_cache(max_seq_len)
        self.register_buffer('tnsr_cos_cached', tnsr_cos, persistent=False)
        self.register_buffer('tnsr_sin_cached', tnsr_sin, persistent=False)


    def _build_cache(
        self,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_len:    <int>
        Return:
            tnsr_cos:   <tensor>  shape: (seq_len, d_head)
            tnsr_sin:   <tensor>  shape: (seq_len, d_head)
        """
        tnsr_position = torch.arange(
            seq_len, 
            dtype=torch.float32,
            device=self.tnsr_inv_freq.device
        )

        # tnsr_freqs shape: (seq_len, d_head / 2)
        tnsr_freqs = torch.outer(tnsr_position, self.tnsr_inv_freq)

        # duplicate along last dim -> (seq_len, d_head) to match the rotate_half convention
        # (first half / second half of the head dim), rather than interleaving pairs
        tnsr_angles = torch.cat([tnsr_freqs, tnsr_freqs], dim=-1)

        return torch.cos(tnsr_angles), torch.sin(tnsr_angles)

    
    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_len:    <int>    Sequence length to retrieve rotary tables for
            device:     <torch.device>
            dtype:      <torch.dtype>
        Return:
            tnsr_cos:   <tensor>  shape: (seq_len, d_head)
            tnsr_sin:   <tensor>  shape: (seq_len, d_head)
        """
        if seq_len > self.max_seq_len:
            tnsr_cos, tnsr_sin = self._build_cache(seq_len)
            self.tnsr_cos_cached = tnsr_cos.to(device=device)
            self.tnsr_sin_cached = tnsr_sin.to(device=device)
            self.max_seq_len = seq_len

        return (
            self.tnsr_cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.tnsr_sin_cached[:seq_len].to(device=device, dtype=dtype)
        )
    


def rotate_half(
    tnsr_X: torch.Tensor
) -> torch.Tensor:
    """
    Splits the last dim in half and rotates: [x1, x2] -> [-x2, x1]

    Args:
        tnsr_X:    <tensor>  shape: (..., d_head)
    Return:
        <tensor>   shape: (..., d_head)
    """
    tnsr_x1 = tnsr_X[..., : tnsr_X.shape[-1] // 2]
    tnsr_x2 = tnsr_X[..., tnsr_X.shape[-1] // 2 :]

    return torch.cat((-tnsr_x2, tnsr_x1), dim=-1)


def apply_rotary_pos_emb(
    tnsr_q: torch.Tensor,
    tnsr_k: torch.Tensor,
    tnsr_cos: torch.Tensor,
    tnsr_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotates query and key vectors by their absolute position, injecting relative
    position information into the dot product q . k without ever adding a
    position vector to the token embedding itself.

    Args:
        tnsr_q:      <tensor>  shape: (batch_size, num_heads, seq_len, d_head)
        tnsr_k:      <tensor>  shape: (batch_size, num_heads, seq_len, d_head)
        tnsr_cos:    <tensor>  shape: (seq_len, d_head)
        tnsr_sin:    <tensor>  shape: (seq_len, d_head)
    Return:
        tnsr_q_rotated, tnsr_k_rotated:    both <tensor> shape: (batch_size, num_heads, seq_len, d_head)
    """
    tnsr_cos = tnsr_cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head)
    tnsr_sin = tnsr_sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head)

    tnsr_q_rotated = (tnsr_q * tnsr_cos) + (rotate_half(tnsr_q) * tnsr_sin)
    tnsr_k_rotated = (tnsr_k * tnsr_cos) + (rotate_half(tnsr_k) * tnsr_sin)

    return tnsr_q_rotated, tnsr_k_rotated
