from torch import nn
import torch

from codes.model.base_model import BaseModel


class Model(BaseModel):
    """ Model to compute inconsistency between two input distributions """

    def __init__(self, config):
        super().__init__(config=config)
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, x):
        p, q = x
        close_loop_loss = self.pdist(p, q)
        return  close_loop_loss, torch.zeros_like(close_loop_loss)
