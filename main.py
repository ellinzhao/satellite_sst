import os

import matplotlib.pyplot as plt
import wandb

from sat_sst.setup import load_components, save_checkpoint
from sat_sst.train import train_epoch, evaluate_dataset


run_components = load_components('default.yaml')
cfg, device, train_loader, val_loader, wrapper_cls = run_components[:5]
model, optim, scheduler, loss, start_epoch, run = run_components[5:]

print(start_epoch)

for epoch in range(start_epoch, cfg.epochs):
    train_loss = train_epoch(train_loader, model, optim, device, cfg.use_loc, loss, wrapper_cls, scheduler=scheduler)

    plot_fname = os.path.join(cfg.save_dir, cfg.wandb.name, f'epoch_{epoch}.png')
    val_loss = evaluate_dataset(
        val_loader, model, device, cfg.use_loc, loss, wrapper_cls,
        plot=True, plot_fname=plot_fname,
    )
    if epoch > 0 and epoch % cfg.save_epoch == 0:
        save_checkpoint(cfg, epoch, model, optim, scheduler)
    run.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val': [wandb.Image(plt.imread(plot_fname), caption='')],
    }, step=epoch)

wandb.finish()
