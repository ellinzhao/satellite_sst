import torch.nn as nn


SST_MIN = -4.2256  # 0.5897
SST_MAX = 3.6316  # 30.396
SST_MEAN = 0.0056  # 21.59
SST_STD = 0.2001  # 2.94


class ScaleSST(nn.Module):

    def forward(self, x):
        # torch.clamp?
        return (x - SST_MEAN) / SST_STD
        # return (x - SST_MIN) / (SST_MAX - SST_MIN)


class UnscaleSST(nn.Module):

    def forward(self, y):
        # torch.clamp?
        return y * SST_STD + SST_MEAN
        # return y * (SST_MAX - SST_MIN) + SST_MIN
