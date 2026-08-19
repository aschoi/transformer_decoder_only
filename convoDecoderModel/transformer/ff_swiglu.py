import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    class definition for SwiGLU Feed foward + activation
    """

    d_model: int
    d_ff: int
    weights_init: str

    def __init__(
        self, 
        d_model: int,
        d_ff: int,
        weights_init='xavier'
    ) -> None:

        """
        swiqlu constructor  
        """

        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self._init_weights(weights_init)
        
        
    def _init_weights(
        self, 
        weights_init: str
    ) -> None:
        '''
        Initialize weights using Xavier Uniform Initialization 

        Args:
            weights_init:       
        '''

        if (weights_init == 'xavier'):
            for nn_linear_proj in [self.gate_proj, self.up_proj, self.down_proj]:
                nn.init.xavier_uniform_(nn_linear_proj.weight)
                if nn_linear_proj.bias is not None:
                    nn.init.constant_(nn_linear_proj.bias, 0)



    def forward(
        self, 
        tnsr_X: torch.Tensor
    ) -> torch.Tensor:

        tnsr_gate = F.silu(self.gate_proj(tnsr_X))
        tnsr_value = self.up_proj(tnsr_X)
        tnsr_hiddenFeatures = tnsr_gate * tnsr_value
        tnsr_output = self.down_proj(tnsr_hiddenFeatures)

        return tnsr_output
