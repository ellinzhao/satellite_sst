import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


DS_CACHE = {}


class SSTDataset(Dataset):

    def __init__(
        self, sst_dir, cloud_dir, split, preload=True, transform=None,
        K=10, fill={'method': 'constant', 'value': 0},
    ):
        self.sst_dir = sst_dir
        self.cloud_dir = cloud_dir
        self.split = split
        self.sst_df = self._load_csv(sst_dir)
        self.cloud_df = self._load_csv(cloud_dir)
        self.transform = transform
        self.fill = fill
        if preload:
            # SST files are larger, so preload these tiles only
            self.preload_tiles(self.sst_dir, self.sst_df)
            self.preload_tiles(self.cloud_dir, self.cloud_df)

        # Pick K random cloud masks for each SST pair
        N = len(self.sst_df)
        self.df = self.sst_df.loc[self.sst_df.index.repeat(K)]
        idx = np.arange(len(self.df))
        self.df.set_index(idx, inplace=True)

        rng = np.random.default_rng()
        cloud_indices = np.hstack([rng.integers(0, len(self.cloud_df), size=K) for _ in range(N)])
        random_cloud_df = self.cloud_df.iloc[cloud_indices].set_index(idx)

        # random_cloud_df.set_index(idx, inplace=True)

        self.df['cloud_ir'] = random_cloud_df['ir']
        self.df['cloud_mw'] = random_cloud_df['mw']

    def _load_csv(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'split.csv'))
        df = df[df['split'] == self.split]
        df = df[['mw', 'ir']]
        return df

    def __len__(self):
        return len(self.df)

    def __del__(self):
        for _, v in DS_CACHE.items():
            # Close open file handles
            v.close()

    def preload_tiles(self, data_dir, df):
        for _, row in df.iterrows():
            self.get_tile(data_dir, row['mw'], save_to_cache=True)
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

    def init_gaps(self, sst, cloud):
        method = self.fill['method']
        ir_sst, mw_sst = torch.unbind(sst)
        ir_cloud, mw_cloud = torch.unbind(cloud)
        if method == 'tile_mean':
            tile_mean = 0.5 * (ir_sst[~ir_cloud].mean() + mw_sst[~mw_cloud].mean())
            fill = tile_mean
        elif method == 'gaussian':
            tile_mean = 0.5 * (ir_sst[~ir_cloud].mean() + mw_sst[~mw_cloud].mean())
            tile_std = 0.5 * (ir_sst[~ir_cloud].std() + mw_sst[~mw_cloud].std())
            fill = torch.normal(tile_mean, tile_std, size=sst.shape)
        elif method == 'microwave':
            fill = tile_mean = 0.5 * (ir_sst[~ir_cloud].mean() + mw_sst[~mw_cloud].mean())
            fill = torch.where(mw_cloud, fill, mw_sst)
            fill = torch.stack([fill, fill])
        elif method == 'constant':
            fill = self.fill['value']
        sst = torch.where(cloud, fill, sst)
        return sst

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ir_sst, mw_sst = self.get_ir_mw_pair(self.sst_dir, row['ir'], row['mw'])
        ir_cloud, mw_cloud = self.get_ir_mw_pair(self.cloud_dir, row['cloud_ir'], row['cloud_mw'])
        ir_cloud = ir_cloud > 0
        mw_cloud = mw_cloud > 0
        ir_nan = ir_cloud | torch.isnan(ir_sst)
        mw_nan = mw_cloud | torch.isnan(mw_sst)
        mw_sst = mw_sst - mw_sst[~mw_nan].mean() + ir_sst[~ir_nan].mean()

        sst = torch.stack([ir_sst, mw_sst])
        mask = torch.stack([ir_cloud, mw_cloud])
        if self.transform is not None:
            sst = self.transform['sst'](sst)
            mask = self.transform['cloud'](mask)
        mask = mask | torch.isnan(sst)  # after the transform, check for nans
        ir_sst, mw_sst = sst[0], sst[1]
        sst_gt = torch.stack([mw_sst - ir_sst, mw_sst], dim=0)
        sst_input = self.init_gaps(sst, mask)
        # sst_gt = torch.where(torch.isnan(sst_gt), 0, sst_gt)
        return sst_input, sst_gt

    def get_ir_mw_pair(self, data_dir, ir_fname, mw_fname):
        _get_da = lambda fname: self.get_tile(data_dir, fname).sst.astype('float32')
        mw_da = _get_da(mw_fname)
        ir_da = _get_da(ir_fname)
        mw_da = mw_da.interp_like(ir_da)
        ir = torch.from_numpy(ir_da.values)
        mw = torch.from_numpy(mw_da.values)

        # Some tiles are (112, 113)
        ir = ir[:112, :112]
        mw = mw[:112, :112]
        return ir, mw
