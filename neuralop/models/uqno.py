from copy import deepcopy

import torch
import torch.nn as nn


from .base_model import BaseModel

class UQNO(BaseModel, name="UQNO"):
    """General N-dim (alpha, delta) Risk-Controlling Neural Operator
    Source: https://arxiv.org/abs/2402.01960

    The UQNO must be trained on 

    Parameters
    ----------
    base_model : nn.Module
        pre-trained solution operator
    alpha : float
        fraction of points excluded from codomain coverage,
        i.e. target codomain coverage rate is 1-alpha
    delta : float
        1 - delta controls the expected proportion of functions 
        that predict an overall coverage of 1-alpha within a given band
    residual_model : nn.Module, optional
        architecture to train as the UQNO's 
        quantile model
    """
    def __init__(self,
                 base_model: nn.Module,
                 alpha:float,
                 delta: float,
                 residual_model: nn.Module=None,
                 ):
        super().__init__()

        self.alpha = alpha
        self.delta = delta
        self.base_model = base_model
        if residual_model is None:
            residual_model = deepcopy(base_model)
        self.residual_model = residual_model
    
    def forward(self, *args, **kwargs):
        """
        Forward pass returns the solution u(a,x)
        and the uncertainty ball E(a,x) as a pair
        for pointwise quantile loss
        """
        self.base_model.eval()
        solution = self.base_model(*args, **kwargs)
        quantile = self.residual_model(*args, **kwargs)
        return (solution, quantile)