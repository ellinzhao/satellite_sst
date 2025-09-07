import torch
import torch.nn as nn


def gradient(x, axis=(-2, -1)):
    derivatives = torch.gradient(x, dim=axis)
    sum_squared = sum([deriv**2 for deriv in derivatives])
    return torch.sqrt(sum_squared + 1e-8)


class MaskedLoss(nn.Module):

    def __init__(self, loss_class=nn.L1Loss):
        super().__init__()
        self.loss = loss_class()

    def forward(self, data):
        target = data.get('target_sst')
        pred = data.get('pred_sst')
        mask = data.get('target_mask')

        known_mask = ~torch.isnan(target)
        if mask is not None:
            known_mask = known_mask & mask.bool()

        # Sometimes nans still propogate even after maskings
        target = torch.nan_to_num(target)

        return self.loss(target[known_mask], pred[known_mask])

    def __str__(self):
        return 'Masked version of ' + str(self.loss)


class GradWeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()  # MaskedLoss(nn.L1Loss)

    def forward(self, data):
        pred = data.get('pred_sst')
        target = data.get('target_sst')

        spatial_weights = gradient(target, axis=(-2, -1))
        spatial_weights = torch.nan_to_num(spatial_weights)
        spatial_weights = spatial_weights**0.3
        spatial_weights += 1e-6
        spatial_weights /= spatial_weights.max()

        known_mask = ~torch.isnan(target)
        target = torch.nan_to_num(target)
        sst_loss = self.l1(
            (pred * spatial_weights)[known_mask],
            (target * spatial_weights)[known_mask],
        )
        return sst_loss


class PredMaskReconLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, data):
        # Inverse transform can include adding the foundation SST ?
        pred = data.get('pred_sst')
        input = data.get('input_sst')
        target = data.get('target_sst')
        pred_mask = data.get('pred_mask').argmax(axis=1).unsqueeze(1)

        known_mask = ~torch.isnan(target)
        # Sometimes nans still propogate even after maskings
        target = torch.nan_to_num(target)

        combined_pred = pred_mask * pred + input * (~pred_mask)
        return self.l1(target[known_mask], combined_pred[known_mask])
