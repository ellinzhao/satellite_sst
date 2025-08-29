from collections.abc import Sequence

import torch.nn as nn


class CombinedLoss(nn.Module):

    def __init__(self, losses: Sequence, weights: Sequence[float, int]):
        super().__init__()
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights

    def __str__(self):
        display = ''
        for w, loss_fn in zip(self.weights, self.losses):
            display += f' {w} * {loss_fn}'
        return display

    def forward(self, data):
        total_loss = 0
        for w, loss_fn in zip(self.weights, self.losses):
            total_loss += w * loss_fn(data)
        return total_loss
