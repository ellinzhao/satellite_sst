import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from .smooth_fill import smooth_fill


DS_CACHE = {}


def add_channel_dim(x):
    if x.ndim < 2 or x.ndim > 3:
        raise IndexError(f'Input has {x.ndim} dims, expected 2 or 3')
    elif x.ndim == 2:
        x = x.unsqueeze(0)
    return x


def masked_mean(sst, cloud):
    return sst[~cloud].mean()


class SSTDataset(Dataset):

    def __init__(
        self, sst_dir, cloud_dir, split, preload=True, transform=None,
        K=10, fill={'method': 'constant', 'value': 0}, fnd_sst_path=None, return_coord=False,
    ):
        self.sst_dir = sst_dir
        self.cloud_dir = cloud_dir
        self.split = split
        self.sst_df = self._load_csv(sst_dir)
        self.cloud_df = self._load_csv(cloud_dir)
        self.fnd_sst = self._load_fnd_sst(fnd_sst_path)
        self.transform = transform
        self.return_coord = return_coord
        self.fill = fill
        if preload:
            # SST files are larger, so preload these tiles only
            self.preload_tiles(self.sst_dir, self.sst_df)
            self.preload_tiles(self.cloud_dir, self.cloud_df)

        self._generate_random_samples(K)

        self.hflip = RandomHorizontalFlip(1)  # randomness is in the _get_random_flip fn
        self.vflip = RandomVerticalFlip(1)

    def _load_fnd_sst(self, path):
        if path is None:
            return None
        ds = xr.open_dataset(path)
        da = ds['sst'].load()
        return da

    def _load_csv(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'split.csv'))
        df = df[df['split'] == self.split]
        return df

    def _generate_random_samples(self, K):
        # Pick K random cloud masks for each SST pair
        N = len(self.sst_df)
        self.df = self.sst_df.loc[self.sst_df.index.repeat(K)]
        idx = np.arange(len(self.df))
        self.df.set_index(idx, inplace=True)

        rng = np.random.default_rng()
        if self.split == 'train':
            cloud_indices = np.hstack([rng.integers(0, len(self.cloud_df), size=K) for _ in range(N)])
        else:
            # cloud indices should be deterministic for non-train datasets
            # so use the unshuffled cloud index
            cloud_indices = np.arange(0, K * N) % len(self.cloud_df)

        random_cloud_df = self.cloud_df.iloc[cloud_indices].set_index(idx)
        self.df['cloud'] = random_cloud_df['ir']

    def __len__(self):
        return len(self.df)

    def __del__(self):
        for _, v in DS_CACHE.items():
            # Close open file handles
            v.close()

    def preload_tiles(self, data_dir, df):
        for _, row in df.iterrows():
            self.get_tile(data_dir, row['ir'], save_to_cache=True)

    def get_tile(self, data_dir, fname, save_to_cache=False):
        path = os.path.join(data_dir, fname)
        if path in DS_CACHE:
            ds = DS_CACHE[path]
        else:
            ds = xr.open_dataset(path)
            if save_to_cache:
                DS_CACHE[path] = ds.load()
        return ds

    def init_gaps(self, sst, cloud, method=None, **kwargs):
        if method is None:
            method = self.fill['method']
            kwargs = self.fill.copy()
            kwargs.pop('method')

        if method == 'smooth':
            fill = smooth_fill(torch.where(cloud, np.nan, sst), **kwargs)
        elif method == 'tile_mean':
            fill = masked_mean(sst, cloud)
        elif method == 'constant':
            fill = kwargs['value']
        sst = torch.where(cloud, fill, sst)
        return sst

    def _get_random_flip(self):
        vflip = torch.rand(1).item() > 0.5
        hflip = torch.rand(1).item() > 0.5

        def _flip(ir):
            if self.split == 'train':
                if vflip:
                    # Random vertical flip
                    ir = self.vflip(ir)
                if hflip:
                    # Random horizontal flip
                    ir = self.hflip(ir)
            return ir
        return _flip

    def _transform_data(self, ir_sst, cloud):
        if self.transform is not None:
            if 'sst' in self.transform:
                ir_sst = self.transform['sst'](ir_sst)
            if 'cloud' in self.transform:
                cloud = self.transform['cloud'](cloud)
        return ir_sst, cloud

    def __getitem__(self, i):
        row = self.df.iloc[i]
        flip_sst = self._get_random_flip()
        flip_cloud = self._get_random_flip()
        return self._get_data_by_row(row, flip_sst, flip_cloud)

    def _get_data_by_row(self, row, flip_sst=None, flip_cloud=None):
        ir_sst = self.get_ir_tensor(self.sst_dir, row['ir'])
        cloud = self.get_ir_tensor(self.cloud_dir, row['cloud'])
        cloud = cloud > 0

        # Flips must occur in the dataset class because of the nan mask
        if flip_sst is not None:
            ir_sst = flip_sst(ir_sst)
        if flip_cloud is not None:
            cloud = flip_cloud(cloud)

        # Add nan regions to cloud mask after flipping SST
        cloud = cloud | torch.isnan(ir_sst)

        # After this point, all tensors have shape (C, H, W)
        ir_sst, cloud = [add_channel_dim(x) for x in (ir_sst, cloud)]
        ir_sst, cloud = self._transform_data(ir_sst, cloud)

        input_ir = self.init_gaps(ir_sst, cloud)
        out = {
            'input_ir': input_ir, 'gt_ir': ir_sst, 'cloud': cloud,
        }
        if self.return_coord:
            coord = row.loc[['lat_start', 'lon_start']].astype('float32')
            coord = torch.from_numpy(coord.to_numpy())
            out['coord'] = coord
        return out

    def get_ir_tensor(self, data_dir, ir_fname):
        _get_da = lambda fname: self.get_tile(data_dir, fname).sst.astype('float32')
        ir_da = _get_da(ir_fname)
        bounds = {
            k: slice(ir_da[k][0], ir_da[k][-1])
            for k in ('lat', 'lon')
        }
        if self.fnd_sst is not None:
            fnd_sst_da = self.fnd_sst.sel(bounds)
            ir_da = ir_da - fnd_sst_da
        ir = torch.from_numpy(ir_da.values)

        # Some tiles are (112, 113)
        ir = ir[:112, :112]
        return ir


def get_input_target(data):
    input_base = data['input_base']
    return data['input_ir'] - input_base, data['gt_ir'] - input_base


def get_recon_target(data, pred):
    input_base = data['input_base']
    return pred + input_base, data['gt_ir']
