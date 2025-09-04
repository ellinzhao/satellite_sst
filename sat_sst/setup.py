import random

import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader

import sat_sst.dataset
import sat_sst.loss
from sat_sst.datatypes import ModelData
from sat_sst.model import ReconModel
from sat_sst.transform import get_scaling_tforms


def _get_dataloader(cfg, var, sst_dir, cloud_dir, transform, split):
    dataset_cls = getattr(sat_sst.dataset, cfg.name)
    dataset = dataset_cls(
        var, sst_dir, cloud_dir, split,
        transform=transform, **cfg.kwargs
    )
    return DataLoader(dataset, **cfg.loader)


def set_seed(i):
    torch.backends.cudnn.deterministic = True
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)


def setup_data(cfg):
    var = cfg.var
    sst_dir, cloud_dir = cfg.sst_dir, cfg.cloud_dir
    out = get_scaling_tforms(var)
    scale_tform = out['forward']
    unscale_tform = out['inverse']
    transform = {'sst': scale_tform}

    train_loader = _get_dataloader(cfg.train, var, sst_dir, cloud_dir, transform, 'train')
    val_loader = _get_dataloader(cfg.val, var, sst_dir, cloud_dir, transform, 'val')
    wrapper_cls = lambda data: ModelData(var, data=data, inv_tform=unscale_tform)
    return train_loader, val_loader, wrapper_cls


def setup_model_optim(cfg, device):
    model = ReconModel(**cfg.model).to(device)
    optim_cls = getattr(torch.optim, cfg.optim.name)
    optimizer = optim_cls(model.parameters(), lr=cfg.optim.lr)
    return model, optimizer


def setup_loss(cfg, debug=False):
    loss_names = cfg.loss
    losses = []
    weights = []
    for loss_name in loss_names:
        loss_cfg = cfg.loss[loss_name]
        loss_cls = getattr(sat_sst.loss, loss_cfg.name)
        loss = loss_cls()
        losses.append(loss)
        weights.append(loss_cfg.weight)
    return sat_sst.loss.CombinedLoss(losses, weights, debug=debug)
