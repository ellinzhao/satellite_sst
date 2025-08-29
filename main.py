import torch
from omegaconf import OmegaConf

from sat_sst.setup import set_seed, setup_data, setup_loss, setup_model_optim
from sat_sst.train import train_epoch


cfg = OmegaConf.load('default.yaml')

set_seed(cfg.seed)
device = torch.device(cfg.device)
train_loader, val_loader, wrapper_cls = setup_data(cfg)
model, optim = setup_model_optim(cfg, device)
loss = setup_loss(cfg)

loss_val = train_epoch(train_loader, model, optim, device, cfg.use_loc, loss, wrapper_cls, plot=True)
# TODO(ellin): if freezing weights, DO NOT freeze all encoder weights - this includes the mask token and input projection
