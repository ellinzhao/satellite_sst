import torch
from tqdm.notebook import tqdm
# from tqdm import tqdm

from .plotting import plot_model_data


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
    loader, model, optimizer, device, use_loc, loss_fn, wrapper_cls, scheduler=None, **kwargs,
):
    model.train()
    use_triplet = False
    epoch_loss = 0.
    pbar = tqdm(loader)
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
    loader, model, optimizer, device, use_loc, loss_fn, wrapper_cls, scheduler=None, triplet_loss_fn=None,
):
    model.train()
    use_triplet = True
    epoch_loss = 0.
    pbar = tqdm(loader)
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
