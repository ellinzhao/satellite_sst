from sat_sst.setup import load_components
from sat_sst.train import train_epoch, evaluate_dataset


cfg, device, train_loader, val_loader, wrapper_cls, model, optim, scheduler, loss = load_components('default.yaml')

train_loss = train_epoch(train_loader, model, optim, device, cfg.use_loc, loss, wrapper_cls, plot=True)
eval_loss = evaluate_dataset(val_loader, model, device, cfg.use_loc, loss, wrapper_cls)
print(train_loss, eval_loss)
