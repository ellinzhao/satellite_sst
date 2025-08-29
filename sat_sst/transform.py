import matplotlib as mpl
import torch.nn as nn
from matplotlib.colors import Normalize

from dataclasses import dataclass


@dataclass
class DatasetStats:
    name: str
    vmin: float = -1
    vmax: float = 1
    mean: float = 0
    std: float = 1


STATS = {
    'sst': DatasetStats('sst', 0.5897, 30.396, 21.59, 2.94),
    'ssta': DatasetStats('ssta', -13.4471, 7.7939, -0.1983, 1.6712),
}


class ScaleSST(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class UnscaleSST(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
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


def get_scaling_tforms(mode):
    stats = STATS[mode]
    return {
        'forward': ScaleSST(stats.mean, stats.std),
        'inverse': UnscaleSST(stats.mean, stats.std)
    }
