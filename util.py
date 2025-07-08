import torch


def gradient(x, axis=(-2, -1)):
    derivatives = torch.gradient(x, dim=axis)
    sum_squared = sum([deriv**2 for deriv in derivatives])
    return torch.sqrt(sum_squared + 1e-8)
