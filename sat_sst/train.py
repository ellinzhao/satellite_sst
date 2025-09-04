import torch

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


def train_epoch(loader, model, optimizer, device, use_loc, loss_fn, wrapper_cls,
                scheduler=None, plot=False, epoch=0):
    model.train()
    use_triplet = False
    epoch_loss = 0.
    for i, data in enumerate(loader):
        out = None  # prevent memory leak?
        optimizer.zero_grad()
        out = process_batch(data, model, use_loc, use_triplet, device, wrapper_cls)
        loss = loss_fn(out)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if scheduler:
        scheduler.step()
    if plot:
        plot_model_data(out, i=0, save_name=f'epoch_{epoch}.png')
    return epoch_loss / len(loader)


def train_triplet_epoch(loader, model, optimizer, device, use_loc, loss_fn, triplet_loss_fn):
    use_triplet = True
    epoch_loss = 0.
    for _, triplet_data in enumerate(loader):
        optimizer.zero_grad()
        feats = {}
        recon_loss = 0
        mask_recon_loss = 0

        for k, data in triplet_data.items():
            out = process_batch(data, model, use_loc, use_triplet, device)
            loss = loss_fn(out)
            feats[k] = (out['z'], out['z_mask'])

        sstA, sstP, sstN = [feats[k][0] for k in ('x1_m1', 'x1_m2', 'x2_m1')]
        trip_sst_loss = triplet_loss_fn(sstA, sstP, sstN)

        maskA, maskP, maskN = [feats[k][0] for k in ('x1_m1', 'x2_m1', 'x1_m2')]
        trip_mask_loss = triplet_loss_fn(maskA, maskP, maskN)
        loss = recon_loss + 0.25 * mask_recon_loss + trip_sst_loss + trip_mask_loss

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate_dataset(loader, model, loss_fn, use_loc):
    model.eval()
    use_triplet = False  # the triplet configuration is not needed for eval
    test_loss = 0
    with torch.no_grad():
        for data in loader:
            out = process_batch(data, model, use_loc, use_triplet)
            loss = loss_fn(out['pred_sst'], out['target_sst'], mask=out['target_mask'].bool())
            test_loss += loss.item()
    test_loss /= len(loader)
    return test_loss
