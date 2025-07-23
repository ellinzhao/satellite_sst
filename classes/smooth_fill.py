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


def _fill_step(input, width, sigma=1):
    pad = int(width // 2)
    kernel = gaussian_kernel(width, sigma=sigma)
    kernel = torch.from_numpy(kernel).float()[None, None]
    convolve = lambda x: F.conv2d(x, kernel, padding=pad)

    input = input.unsqueeze(dim=0)
    nan_mask = torch.isnan(input)
    data_sum = convolve(torch.nan_to_num(input))
    valid_count = convolve((~nan_mask).float())

    filtered = data_sum / valid_count
    filtered = torch.where(valid_count < 0.01, np.nan, filtered)
    filtered = torch.where(nan_mask, filtered, input)
    return filtered.squeeze(0)


def smooth_fill(input, width_base=7, increment_every_k=2):
    i = 0
    sigma = 1
    if torch.isnan(input).float().mean() == 1:
        # No information to work with!
        return None

    while torch.isnan(input).sum() > 0:
        if i < 200:
            # Kernel width increases every `increment_every_k` steps
            # Sigma is set so that the FWHM (approximately) fills the kernel
            k = int(np.floor(i // increment_every_k) * 2) + width_base
            sigma = k
        else:
            warnings.warn('Smooth fill has not converged')
            print(input.shape, torch.isnan(input).float().mean())
            tile_mean = input[~torch.isnan(input)].mean()
            return torch.where(torch.isnan(input), tile_mean, input)
        input = _fill_step(input, k, sigma=sigma)
        i += 1
    return input
