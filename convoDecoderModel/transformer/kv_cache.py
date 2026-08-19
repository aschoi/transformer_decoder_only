from __future__ import annotations

import torch

class KVCache:
    """
    

    Tensor layout:      (batch_size, num_kv_heads, max_seq_len, head_dim)

    During inference, new key/value states are appened to the cache as tokens
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device, 
        dtype: torch.dtype
    ) -> None:

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        if num_kv_heads <= 0:
            raise ValueError("KV heads must be greater than 0")
        if max_seq_len <= 0:
            raise ValueError("Max Sequence Length must be greater than 0")
        if head_dim <= 0:
            raise ValueError("Head dimensions must be greater than 0")

        self.key = torch.empty(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            device=device, dtype=dtype
        )
        self.value = torch.empty(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            device=device, dtype=dtype
        )
        self.seq_len = 0



    @property
    def max_seq_len(self) -> int:
        return self.key.size(2)

    @property
    def batch_size(self) -> int:
        return self.key.size(0)


    def reset(self) -> None:
        """
        Logically clear the cache
        Memory does nto need to be zeroed b/c positions >= seq_len are never read
        """
        self.seq_len = 0


    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append newly computed K/V tensors to the cache
        
        Args:
            key:        <tensor>  shape: (batch_size, num_kv_heads, NEW_seq_len, head_dim)
            value:      <tensor>  shape: (batch_size, num_kv_heads, NEW_seq_len, head_dim)
        
        Return:
            tuple of cached K and V tensors containing all tokens up to current point

            K           <tensor>  shape: (batch_size, num_kv_heads, TOTAL_seq_len, head_dim)
            V           <tensor>  shape: (batch_size, num_kv_heads, TOTAL_seq_len, head_dim)
        """

        # Argument Validation Checks
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch. key: {key.shape} != value: {value.shape}")
        if key.ndim != 4:
            raise ValueError(f"expected rank-4 K tensor. current key-rank: {key.ndim}")
        if value.ndim != 4:
            raise ValueError(f"expected rank-4 V tensor. current value-rank: {value.ndim}")
        if key.size(0) != self.key.size(0) or value.size(0) != self.value.size(0):
            raise ValueError(f"KV cache batch size does not match input batch size")
        if key.size(1) != self.key.size(1) or value.size(1) != self.value.size(1):
            raise ValueError(f"KV cache head count does not match input head count")
        if key.size(3) != self.key.size(3) or value.size(3) != self.value.size(3):
            raise ValueError(f"KV cache head dimensions do not match input head dimenions")

        new_seq_len = key.size(2)
        start = self.seq_len
        end = start + new_seq_len

        if end > self.max_seq_len:
            raise RuntimeError(f"KV cache capacity exceeded: {end} > {self.max_seq_len}")


        self.key[:, :, start:end, :].copy_(key)
        self.value[:, :, start:end, :].copy_(value)

        self.seq_len = end

        return (
            self.key[:, :, :end, :],
            self.value[:, :, :end, :]
        )
        

    
