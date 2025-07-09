from .unet_ir import UNet as UNetAnomaly
from .unet_mw import UNet as UNetBase
import torch.nn as nn
from .sst_dataset import upsample


class UNetSeries(nn.Module):

    def __init__(self):
        super().__init__()
        self.unet_base = UNetBase(in_ch=2, n_class=1, chs=[8, 12, 16, 24])
        self.unet_anomaly = UNetAnomaly(in_ch=1, n_class=1, chs=[32, 48, 64, 84])

    def forward(self, input_base, input_ir):
        pred_base = self.unet_base(input_base)
        input_anomaly = input_ir - upsample(pred_base)
        pred_anomaly = self.unet_anomaly(input_anomaly)
        return pred_base, pred_anomaly
