import torch
import torch.nn as nn

from ..util import center_crop


def gradient(x, axis=(-2, -1)):
    derivatives = torch.gradient(x, dim=axis)
    sum_squared = sum([deriv**2 for deriv in derivatives])
    return torch.sqrt(sum_squared + 1e-8)


class MaskedLoss(nn.Module):

    def __init__(self, loss_class=nn.L1Loss, n_crop=0):
        super().__init__()
        self.loss = loss_class()
        self.n_crop = n_crop

    def _prepare_data(self, data):
        target = data.get('target_sst')
        pred = data.get('pred_sst')
        mask = data.get('target_mask')

        known_mask = ~torch.isnan(target)
        if mask is not None:
            known_mask = known_mask & mask.bool()

        # Sometimes nans still propogate even after maskings
        target = torch.nan_to_num(target)
        if self.n_crop > 0:
            known_mask = center_crop(known_mask, self.n_crop)
        return target, pred, known_mask

    def forward(self, data):
        target, pred, known_mask = self._prepare_data(data)
        return self.loss(target[known_mask], pred[known_mask])

    def __str__(self):
        return 'Masked version of ' + str(self.loss)


class GradientMaskedLoss(MaskedLoss):

    def forward(self, data):
        target, pred, known_mask = self._prepare_data(data)
        target_derivs = torch.gradient(target, axis=(-2, -1))
        pred_derivs = torch.gradient(pred, axis=(-2, -1))

        total_loss = 0
        for target_d, pred_d in zip(target_derivs, pred_derivs):
            total_loss += self.loss(target_d[known_mask], pred_d[known_mask])
        return total_loss
