import matplotlib as mpl
import torch.nn as nn
from matplotlib.colors import Normalize


SST_MIN = -4.2256  # 0.5897
SST_MAX = 3.6316  # 30.396
SST_MEAN = 18.9057  # 21.59
SST_STD = 4.2406  # 2.94


def get_scaling_tforms(mode):
    if mode == 'sst':
        mean, std = 18.9057, 4.2406
    elif mode == 'anomaly':
        mean, std = 0, 1
    return {'forward': ScaleSST(mean, std), 'inverse': UnscaleSST(mean, std)}


class ScaleSST(nn.Module):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class UnscaleSST(nn.Module):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, y):
        return y * self.std + self.mean


class SSTtoColor(nn.Module):

    def __init__(self, cmap_type='Spectral_r'):
        self.cmap = mpl.colormaps[cmap_type]

    def forward(self, y):
        norm = Normalize(vmin=0.58967, vmax=31, clip=True)
        return self.cmap(norm(y))  # need to remove alpha channel
