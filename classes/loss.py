import torch
import torch.nn as nn
from pytorch_msssim import SSIM

from .transform import UnscaleSST, SST_MIN, SST_MAX
from ..util import gradient


class SSIMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.ssim_module = SSIM(
            data_range=1., size_average=True,
            channel=1, nonnegative_ssim=True,
        )
        self.inverse_tform = UnscaleSST()

    def _unnormalize(self, x):
        # Unstandardize the data and map to range [0, 1]
        x = self.inverse_tform(x)
        x = (x - SST_MIN) / (SST_MAX - SST_MIN)
        return torch.nan_to_num(x)

    def forward(self, pred, target):
        return self.ssim_module(self._unnormalize(pred), self._unnormalize(target))


class MaskedLoss(nn.Module):

    def __init__(self, loss_class):
        super().__init__()
        self.loss = loss_class()

    def forward(self, pred, target, mask=None):
        known_mask = ~torch.isnan(target)
        if mask is not None:
            known_mask = known_mask & mask

        # Sometimes nans still propogate even after maskings
        target = torch.nan_to_num(target)

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

    def forward(self, pred, target, mask=None):
        pred = self.inverse_tform(pred)
        target = self.inverse_tform(target)

        sst_loss = self.masked_l1(pred, target)

        # Hacky way of toggling the gradient loss
        if self.grad_weight == 0:
            grad_loss = 0
        else:
            pred_grad = gradient(pred, axis=(-2, -1))  # B, C, H, W
            target_grad = gradient(target, axis=(-2, -1))
            grad_loss = self.masked_l1(pred_grad, target_grad, mask=mask)
        return self.sst_weight * sst_loss + self.grad_weight * grad_loss


class GradWeightedLoss(nn.Module):

    def __init__(self, grad_preprocess_fn=lambda x: x):
        super().__init__()
        self.masked_l1 = MaskedLoss(nn.L1Loss)
        self.inverse_tform = UnscaleSST()
        self.grad_preprocess_fn = grad_preprocess_fn

    def forward(self, pred, target, mask=None):
        pred = self.inverse_tform(pred)
        target = self.inverse_tform(target)

        spatial_weights = gradient(target, axis=(-2, -1))
        spatial_weights = torch.nan_to_num(spatial_weights)
        spatial_weights = self.grad_preprocess_fn(spatial_weights)
        sst_loss = self.masked_l1(
            pred * spatial_weights,
            target * spatial_weights,
            mask=mask,
        )
        return sst_loss
