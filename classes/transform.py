import matplotlib as mpl
import torch.nn as nn
from matplotlib.colors import Normalize


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


class SSTtoColor(nn.Module):

    def __init__(self, cmap_type='Spectral_r'):
        self.cmap = mpl.colormaps[cmap_type]

    def forward(self, y):
        norm = Normalize(vmin=0.58967, vmax=31, clip=True)
        return self.cmap(norm(y))  # need to remove alpha channel
