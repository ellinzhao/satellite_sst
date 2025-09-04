from collections.abc import Sequence

import torch.nn as nn


class CombinedLoss(nn.Module):

    def __init__(self, losses: Sequence, weights: Sequence[float, int], debug: float = False):
        super().__init__()
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights
        self.debug = debug

    def __str__(self):
        display = ''
        for w, loss_fn in zip(self.weights, self.losses):
            display += f' {w} * {loss_fn}'
        return display

    def forward(self, data):
        total_loss = 0
        losses = []
        for w, loss_fn in zip(self.weights, self.losses):
            loss_val = loss_fn(data)
            total_loss += w * loss_val
            losses.append(loss_val)
        if self.debug:
            print(' '.join([f'{loss.item():.5f}' for loss in losses]))
        return total_loss
