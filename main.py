import wandb

from sat_sst.setup import load_components
from sat_sst.train import set_resnet_training, train_and_eval_epoch


cfg_path = 'triplet.yaml'
components = load_components(cfg_path)
start_epoch, end_epoch = components['start_epoch'], components['end_epoch']
model = components['model']

set_resnet_training(model, freeze=True)

for epoch in range(start_epoch, end_epoch):
    if epoch > 10:
        set_resnet_training(model, freeze=False)
    train_and_eval_epoch(components, epoch=epoch)

wandb.finish()
