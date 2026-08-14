import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    """

    d_model: int
    eps: float

    def __init__(
        self,
        d_model,
        eps=1e-6
    ) -> None:
        """
        RMSNorm Constructor.
        Rescales by root-mean-square only (no mean-centering, no bias),
        
        Args:
            d_model:    <int>    Model dimension
            eps:        <float>  Numerical stability epsilon
        """
        
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model))



    def forward(
        self,
        tnsr_X: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tnsr_X:    <tensor>    shape: (batch_size, seq_len, d_model)
        Return:
            <tensor>   shape: (batch_size, seq_len, d_model)
        """

        input_dtype = tnsr_X.dtype
        tnsr_X_fp32 = tnsr_X.float()

        tnsr_mean_square = tnsr_X_fp32.pow(2).mean(dim=-1, keepdim=True)
        
        tnsr_normed = tnsr_X_fp32 * torch.rsqrt(
            tnsr_mean_square + self.eps
        )

        return (self.weights * tnsr_normed).to(input_dtype)
    


    
