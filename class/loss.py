import torch
import torch.nn as nn


# inverse_transform = UnscaleSST()


class MaskedLoss(nn.Module):

    def __init__(self, loss_class):
        self.loss = loss_class()

    def forward(self, pred, target):
        # gt = inverse_transform(gt)
        # est = inverse_transform(est)

        known_mask = ~torch.isnan(target)

        # Sometimes nans still propogate even after maskings
        target = torch.nan_to_num(target)
        return self.loss(target[known_mask], pred[known_mask])

    def __str__(self):
        return 'Masked version of ' + str(self.loss)
