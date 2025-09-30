import torch
import torch.nn as nn
from pytorch_msssim import SSIM

from ..util import center_crop, gradient


def batch_masked_mean(x, mask=1, keep_dim=True):
    mean = torch.sum(x * mask, dim=(1, 2, 3), keepdim=True) / torch.sum(mask, dim=(1, 2, 3), keepdim=True)
    if keep_dim:
        mean = torch.ones_like(x) * mean
    return mean


class SSIMLoss(nn.Module):

    def __init__(self, n_crop=0):
        super().__init__()
        self.ssim_module = SSIM(
            data_range=1., size_average=True,
            channel=1, nonnegative_ssim=True,
        )
        self.n_crop = n_crop

    def _preprocess(self, x):
        return x

    def _unnormalize(self, x):
        # Unstandardize the data and map to range [0, 1]
        b = x.shape[0]
        flat_x = x.view(b, -1)
        tile_min = flat_x.min(dim=1).values[:, None, None, None]
        tile_max = flat_x.max(dim=1).values[:, None, None, None]
        x = (x - tile_min) / (tile_max - tile_min)
        return x

    def forward(self, data):
        pred = data.get('pred_sst')
        target = data.get('target_sst')
        mask = data.get('target_mask')

        pred = self._preprocess(pred)
        target = self._preprocess(target)

        pred = torch.nan_to_num(pred)
        target = torch.nan_to_num(target)

        known_mask = ~torch.isnan(target)
        if mask is not None:
            known_mask = known_mask & mask.bool()
        if self.n_crop > 0:
            known_mask = center_crop(known_mask, self.n_crop)

        mask = mask.bool()
        target = torch.where(known_mask, 0, target)
        pred = torch.where(known_mask, 0, pred)

        target = self._unnormalize(target)
        pred = self._unnormalize(pred)
        return 1 - self.ssim_module(pred, target)


class GradientSSIMLoss(SSIMLoss):

    def _preprocess(self, x):

        return gradient(x)
