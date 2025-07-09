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


def _fill_step(input, width):
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
    return filtered.squeeze(0)


def smooth_fill(input, width_base=7, increment_every_k=3):
    i = 0
    while torch.isnan(input).sum() > 0:
        k = int(np.floor(i // increment_every_k) * 2) + width_base
        input = _fill_step(input, k)
        i += 1
    return input
