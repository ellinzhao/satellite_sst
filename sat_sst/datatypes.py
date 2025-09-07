from enum import Enum

import numpy as np
import torch


class Var(Enum):
    SST = 'sst'
    SSTA = 'ssta'
    SSTA_PYR = 'ssta_pyr'


class ModelData:
    '''
    Represents the model input and outputs, possibly batched.
    It would be much cleaner to use __setattr__ and __getattribute__ instead of
    a data dictionary, but it's more annoying to debug. So this class is a glorified
    dictionary in which transforms are applied to some of the values.
    '''

    KEYS = [
        'input_sst', 'target_sst', 'target_mask', 'fnd_sst',
        'pred_sst', 'pred_mask', 'z_sst', 'z_mask',
    ]

    def __init__(self, var, data={}, inv_tform=lambda x: x):
        self.var = var
        self.data = data
        self.inv_tform = inv_tform

    def __str__(self):
        return ' '.join(list(self.data.keys()))

    def add(self, key, x):
        self.data[key] = x

    def _convert_array_type(self, x, device, array_type):
        if type(device) is str:
            device = torch.device(device)
        if isinstance(x, torch.Tensor):
            x = x.to(device)
            if array_type == 'numpy':
                x = x.detach()
                x = x.numpy()
        elif isinstance(x, np.array) and array_type == 'torch':
            # Can't think of a use case for this, but adding this check for completeness
            x = torch.from_numpy(x)
        return x

    def _transform(self, key, x, **kwargs):
        if '_sst' in key and 'z_' not in key:
            x = self.inv_tform(x)
        return x

    def get(self, key, device='cpu', array_type='torch', **kwargs):
        x = self.data[key]
        x = self._transform(key, x, **kwargs)
        x = self._convert_array_type(x, device, array_type)
        return x


class AnomalyData(ModelData):

    def _transform(self, key, x, **kwargs):
        x = super()._transform(key, x, **kwargs)
        if '_sst' in key and 'z_' not in key:
            x += self.data['fnd_sst']
        return x
