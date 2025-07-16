import warnings

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel(width=11, sigma=1):
    """
    Create gaussian kernel with side length `width` and a sigma of `sig`
    """
    assert width % 2 == 1
    ax = np.linspace(-(width - 1) / 2, (width - 1) / 2, width)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def _fill_step(input, width, valid_count_override=False):
    pad = int(width // 2)
    kernel = gaussian_kernel(width)
    kernel = torch.from_numpy(kernel).float()[None, None]
    convolve = lambda x: F.conv2d(x, kernel, padding=pad)

    input = input.unsqueeze(dim=0)
    nan_mask = torch.isnan(input)
    data_sum = convolve(torch.nan_to_num(input))
    valid_count = convolve((~nan_mask).float())

    filtered = data_sum / valid_count
    filtered = torch.where(nan_mask, filtered, input)
    if valid_count_override:
        return filtered.squeeze(0)
    filtered = torch.where(valid_count < 0.25, np.nan, filtered)
    return filtered.squeeze(0)


def smooth_fill(input, width_base=7, increment_every_k=3):
    i = 0
    while torch.isnan(input).sum() > 0:
        override = False
        if i < 20:
            k = int(np.floor(i // increment_every_k) * 2) + width_base
        elif i < 50:
            override = True
            k = input.shape[1] // 2
        else:
            warnings.warn('Smooth fill has not converged')
            tile_mean = input[~torch.isnan(input)].mean()
            return torch.where(torch.isnan(input), tile_mean, input)
        input = _fill_step(input, k, valid_count_override=override)
        i += 1
    return input
