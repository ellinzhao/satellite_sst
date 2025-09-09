import os

import matplotlib.pyplot as plt
import torch
import wandb
from tqdm.notebook import tqdm
# from tqdm import tqdm

from .plotting import plot_model_data
from .setup import save_checkpoint


def set_resnet_training(model, freeze: bool = True):
    for param in model.enc.down_blocks.parameters():
        param.requires_grad = freeze


def get_loc_emb(data, device, satclip_model):
    lat_lon = data['coord']
    with torch.no_grad():
        loc_emb = satclip_model(lat_lon.to(device).double()).float()
    return loc_emb.to(device)


def process_batch(data, model, use_loc, use_triplet, device, wrapper_cls):
    data = {k: v.to(device) for k, v in data.items()}
    forward_kwargs = {'return_z': use_triplet}
    if use_loc:
        forward_kwargs['loc_emb'] = get_loc_emb(data, device, None)
    out = model(data['input_sst'], mask=data['target_mask'], **forward_kwargs)
    data = wrapper_cls(data | out)
    return data


def train_epoch(
    loader, model, optimizer, device, use_loc, loss_fn, wrapper_cls, scheduler=None,
    pbar_title='', **kwargs,
):
    model.train()
    use_triplet = False
    epoch_loss = 0.
    pbar = tqdm(loader)
    pbar.set_description(pbar_title)
    for i, data in enumerate(pbar):
        out = None  # prevent memory leak
        optimizer.zero_grad()
        out = process_batch(data, model, use_loc, use_triplet, device, wrapper_cls)
        loss = loss_fn(out)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
    pbar.close()
    if scheduler:
        scheduler.step()
    return epoch_loss / len(loader)


def train_triplet_epoch(
    loader, model, optimizer, device, use_loc, loss_fn, wrapper_cls, scheduler=None,
    pbar_title=None, triplet_loss_fn=None, **kwargs
):
    model.train()
    use_triplet = True
    epoch_loss = 0.
    pbar = tqdm(loader)
    pbar.set_description(pbar_title)
    for i, triplet_data in enumerate(pbar):
        out = None  # prevent memory leak
        optimizer.zero_grad()
        feats = {}
        for k, data in triplet_data.items():
            out = process_batch(data, model, use_loc, use_triplet, device, wrapper_cls)
            loss = loss_fn(out)
            feats[k] = {'sst': out.get('z_sst'), 'mask': out.get('z_mask')}

        sstA, sstP, sstN = [feats[k]['sst'] for k in ('x1_m1', 'x1_m2', 'x2_m1')]
        loss += triplet_loss_fn(sstA, sstP, sstN)

        # maskA, maskP, maskN = [feats[k]['mask'] for k in ('x1_m1', 'x2_m1', 'x1_m2')]
        # loss += triplet_loss_fn(maskA, maskP, maskN)
        # # mask reconstruction converges quickly - triplet loss not needed?

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    pbar.close()
    if scheduler:
        scheduler.step()
    return epoch_loss / len(loader)


def evaluate_dataset(loader, model, device, use_loc, loss_fn, wrapper_cls, plot=True, plot_fname=None):
    model.eval()
    use_triplet = False  # the triplet configuration is not needed for eval
    test_loss = 0.
    plot_batch_i = len(loader) // 2
    with torch.no_grad():
        for i, data in enumerate(loader):
            out = process_batch(data, model, use_loc, use_triplet, device, wrapper_cls)
            loss = loss_fn(out)
            test_loss += loss.item()
            if i == plot_batch_i:
                plot_data = out
    test_loss /= len(loader)
    if plot:
        assert plot_fname is not None
        plot_model_data(plot_data, i=0, save_name=plot_fname)
    return test_loss


def train_and_eval_epoch(comp, epoch):
    cfg, device, model, run = [comp[k] for k in ('cfg', 'device', 'model', 'run')]
    train_loader, val_loader, wrapper_cls = comp['data']
    loss, triplet_loss = comp['loss']
    optim, scheduler = comp['train_params']
    end_epoch = comp['end_epoch']

    train_epoch_fn = train_epoch if not cfg.use_triplet else train_triplet_epoch

    log = {}
    if scheduler:
        log['lr'] = scheduler.get_last_lr()[0]

    train_loss = train_epoch_fn(
        train_loader, model, optim, device, cfg.use_loc, loss, wrapper_cls,
        scheduler=scheduler, pbar_title=f'{epoch}/{end_epoch}', triplet_loss_fn=triplet_loss,
    )
    log['train_loss'] = train_loss

    if epoch > 0 and epoch % cfg.save_epoch == 0:
        save_checkpoint(cfg, epoch, model, optim, scheduler)
    if epoch % cfg.val_epoch == 0:
        plot_fname = os.path.join(cfg.save_dir, cfg.wandb.name, f'epoch_{epoch}.png')
        val_loss = evaluate_dataset(
            val_loader, model, device, cfg.use_loc, loss, wrapper_cls,
            plot=True, plot_fname=plot_fname,
        )
        log['val_loss'] = val_loss
        log['val_recon'] = [wandb.Image(plt.imread(plot_fname), caption='')]
    run.log(log, step=epoch)
