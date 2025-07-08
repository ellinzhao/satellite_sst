import torch


def gradient(x, axis=(-2, -1)):
    derivatives = torch.gradient(x, dim=axis)
    sum_squared = sum([deriv**2 for deriv in derivatives])
    return torch.sqrt(sum_squared + 1e-8)


def resolution_components(sst):
    ir, mw = torch.unbind(sst)
    return torch.stack([ir - mw, mw], dim=0)


def get_satellite(res_components, ):
    laplac, gauss = torch.unbind(res_components)
    return laplac + gauss, gauss
