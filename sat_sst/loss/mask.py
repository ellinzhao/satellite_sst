import torch
import torch.nn as nn


class MaskBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, data):
        target = data.get('target_mask')
        pred = data.get('pred_mask')
        # convert target to one hot encdoed array
        target = target.bool()
        target = torch.cat([target, ~target], dim=1).float()
        return self.bce(pred, target)
