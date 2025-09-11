import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import sat_sst.dataset
import sat_sst.loss
from sat_sst.datatypes import AnomalyData, ModelData
from sat_sst.model import ReconModel
from sat_sst.transform import get_scaling_tforms


def _get_dataloader(cfg, var, sst_dir, cloud_dir, transform, split):
    dataset_cls = getattr(sat_sst.dataset, cfg.name)
    dataset = dataset_cls(
        var, sst_dir, cloud_dir, split,
        transform=transform, **cfg.kwargs
    )
    return DataLoader(dataset, **cfg.loader)


def load_components(fname):
    cfg = OmegaConf.load(fname)
    base_cfgs = []
    if 'base_files' in cfg:
        base_cfgs.extend([OmegaConf.load(f) for f in cfg.base_files])
    cfg = OmegaConf.merge(*base_cfgs, cfg)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_loader, val_loader, wrapper_cls = setup_data(cfg)
    model, optim, scheduler = setup_model_optim(cfg, device)
    loss, triplet_loss, val_losses = setup_loss(cfg)
    start_epoch, run = load_training_state(cfg, model, optim, scheduler)
    return {
        'cfg': cfg,
        'device': device,
        'model': model,
        'run': run,
        'start_epoch': start_epoch,
        'end_epoch': cfg.epochs,
        'data': [train_loader, val_loader, wrapper_cls],
        'loss': [loss, triplet_loss, val_losses],
        'train_params': [optim, scheduler],
    }


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
    data_cls = ModelData if var == 'sst' else AnomalyData
    wrapper_cls = lambda data: data_cls(var, data=data, inv_tform=unscale_tform)

    if cfg.train.name == 'TripletSSTDataset':
        assert cfg.triplet_loss is not None
    return train_loader, val_loader, wrapper_cls


def setup_model_optim(cfg, device):
    model = ReconModel(**cfg.model).to(device)
    optim_cls = getattr(torch.optim, cfg.optim.name)
    optimizer = optim_cls(model.parameters(), lr=cfg.optim.lr)
    scheduler = None
    if 'scheduler' in cfg:
        scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)
        args = cfg.scheduler.args
        kwargs = cfg.scheduler.kwargs or {}
        scheduler = scheduler_cls(optimizer, *args, **kwargs)
    return model, optimizer, scheduler


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
    loss_fn = sat_sst.loss.CombinedLoss(losses, weights, debug=debug)

    val_loss_fns = {}
    val_loss_names = cfg.val_loss
    for loss_name in val_loss_names:
        loss_cfg = cfg.val_loss[loss_name]
        loss_cls = getattr(sat_sst.loss, loss_cfg.name)
        val_loss_fns[loss_name] = loss_cls()

    weighted_trip_loss_fn = lambda a, p, n: 0
    if cfg.get('triplet_loss', None):
        trip_loss_fn = nn.TripletMarginLoss(**cfg.triplet_loss.get('kwargs', {}))
        weighted_trip_loss_fn = lambda a, p, n: cfg.triplet_loss.weight * trip_loss_fn(a, p, n)
    return loss_fn, weighted_trip_loss_fn, val_loss_fns


def save_checkpoint(cfg, epoch, model, optimizer, scheduler):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler:
        state['scheduler_state_dict'] = scheduler.state_dict()
    save_path = os.path.join(cfg.save_dir, cfg.wandb.name, 'training_ckpt.tar')
    torch.save(state, save_path)
    print('Saved checkpoint to', save_path)


def check_cfg_match(cfg1, cfg2, model_only=False):
    cfg1 = OmegaConf.to_object(cfg1)
    cfg2 = OmegaConf.to_object(cfg2)
    if model_only:
        return cfg1.model == cfg2.model
    cfg_wandb1 = cfg1.pop('wandb', {})
    cfg_wandb2 = cfg2.pop('wandb', {})
    id1 = cfg_wandb1.pop('id', None)
    id2 = cfg_wandb2.pop('id', None)
    if id1 is not None and id2 is not None:
        return id1 == id2
    return cfg1 == cfg2


def load_training_state(cfg, model, optimizer, scheduler):
    run_save_dir = os.path.join(cfg.save_dir, cfg.wandb.name)
    cfg_save_path = os.path.join(run_save_dir, 'config.yaml')
    pretrained_path = cfg.get('pretrained_weights_path', None)
    epoch = 0
    if pretrained_path:
        assert pretrained_path.endswith('.tar')
        assert os.path.isfile(pretrained_path)
        try:
            ckpt = torch.load(pretrained_path, weights_only=True)
        except RuntimeError:
            ckpt = torch.load(pretrained_path, weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model_state_dict'])
        print('Loaded pretrained model weights...')

    if not pretrained_path and os.path.isdir(run_save_dir) and os.path.isfile(cfg_save_path):
        print('Found previous run...')
        saved_cfg = OmegaConf.load(cfg_save_path)
        assert check_cfg_match(cfg, saved_cfg)
        run_id = saved_cfg.wandb.id
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, id=run_id, resume='must')

        # Even if run exists, it is possible that checkpoint was NOT saved
        ckpt_path = os.path.join(run_save_dir, 'training_ckpt.tar')
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            epoch = ckpt['epoch'] + 1
            print('Loaded previous training state...')
    else:
        os.makedirs(run_save_dir, exist_ok=True)
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_object(cfg),
        )
        cfg.wandb.id = run.id
        OmegaConf.save(cfg, cfg_save_path)
    return epoch, run
