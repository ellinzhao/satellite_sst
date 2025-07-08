import torch
import torch.nn as nn

from .transform import UnscaleSST
from ..util import gradient


class MaskedLoss(nn.Module):

    def __init__(self, loss_class):
        super().__init__()
        self.loss = loss_class()

    def forward(self, pred, target):
        known_mask = ~torch.isnan(target)

        # Sometimes nans still propogate even after maskings
        target = torch.nan_to_num(target)

        # Do NOT remove nans from prediction - we want to know if the model is predicting nan!
        return self.loss(target[known_mask], pred[known_mask])

    def __str__(self):
        return 'Masked version of ' + str(self.loss)


class FullLoss(nn.Module):

    def __init__(self, sst_weight=1, grad_weight=1):
        super().__init__()
        self.masked_l1 = MaskedLoss(nn.L1Loss)
        self.inverse_tform = UnscaleSST()
        self.sst_weight = sst_weight
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        pred = self.inverse_tform(pred)
        target = self.inverse_tform(target)
        sst_loss = self.masked_l1(pred, target)

        # Hacky way of toggling the gradient loss
        if self.grad_weight == 0:
            grad_loss = 0
        else:
            pred_grad = gradient(pred, axis=(-2, -1))  # B, C, H, W
            target_grad = gradient(target, axis=(-2, -1))
            grad_loss = self.masked_l1(pred_grad, target_grad)
        return self.sst_weight * sst_loss + self.grad_weight * grad_loss
