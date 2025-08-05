import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize, RandomHorizontalFlip, RandomVerticalFlip

from .smooth_fill import smooth_fill


DS_CACHE = {}


def add_channel_dim(x):
    if x.ndim < 2 or x.ndim > 3:
        raise IndexError(f'Input has {x.ndim} dims, expected 2 or 3')
    elif x.ndim == 2:
        x = x.unsqueeze(0)
    return x


def downsample(ir, cloud_mask=None):
    if cloud_mask is None:
        pool = torch.nn.AvgPool2d(12, count_include_pad=False, padding=4)
        return pool(ir)
    pool = torch.nn.AvgPool2d(12, divisor_override=1, padding=4)
    data_pool = pool(torch.where(cloud_mask, 0, ir))
    mask_pool = pool((~cloud_mask).float())
    nonpad = pool(torch.ones_like(ir))
    frac_valid_px = mask_pool / nonpad
    return torch.where(
        frac_valid_px > 0.5,
        data_pool / mask_pool,
        np.nan
    )


def upsample(mw):
    up = Resize(112, interpolation=InterpolationMode.BICUBIC)
    return up(mw)


def masked_mean(sst, cloud):
    return sst[~cloud].mean()


def debias_mw(ir_sst, mw_sst, ir_cloud, mw_cloud):
    ir_nan = ir_cloud | torch.isnan(ir_sst)
    mw_nan = mw_cloud | torch.isnan(mw_sst)
    ir_nan = downsample(ir_nan.float()) > 0
    ir_sst = downsample(ir_sst)
    both_clear = (~ir_nan) & (~mw_nan)

    # Debias the MW data
    mw_sst = mw_sst - mw_sst[both_clear].mean() + ir_sst[both_clear].mean()
    return mw_sst


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

        self._generate_random_samples(K)
        self._generate_lat_lon_pairs()

        self.hflip = RandomHorizontalFlip(1)
        self.vflip = RandomVerticalFlip(1)

    def _load_csv(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'split.csv'))
        df = df[df['split'] == self.split]
        df = df[['mw', 'ir']]
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

        self.df['cloud_ir'] = random_cloud_df['ir']
        self.df['cloud_mw'] = random_cloud_df['mw']

    def _generate_lat_lon_pairs(self):
        pat = '([a-z0-9]+)_([a-z0-9]+)_([a-z0-9-.]+)_([a-z0-9-.]+).nc'
        coord_df = self.df['mw'].str.split(pat, regex=True, expand=True)
        coord_df = coord_df.iloc[:, 3:5].astype(float)
        self.df['mw_point'] = list(zip(coord_df.iloc[:, 0], coord_df.iloc[:, 1]))

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
        elif method == 'microwave':
            fill = kwargs['microwave']
        sst = torch.where(cloud, fill, sst)
        return sst

    def _get_random_flip(self):
        vflip = torch.rand(1).item() > 0.5
        hflip = torch.rand(1).item() > 0.5

        def _flip(ir, mw):
            if self.split == 'train':
                if vflip:
                    # Random vertical flip
                    ir = self.vflip(ir)
                    mw = self.vflip(mw)
                if hflip:
                    # Random horizontal flip
                    ir = self.hflip(ir)
                    mw = self.hflip(mw)
            return ir, mw
        return _flip

    def _transform_data(self, ir_sst, mw_sst, ir_cloud, mw_cloud):
        if self.transform is not None:
            if 'sst' in self.transform:
                ir_sst = self.transform['sst'](ir_sst)
                mw_sst = self.transform['sst'](mw_sst)
            if 'cloud' in self.transform:
                ir_cloud = self.transform['cloud'](ir_cloud)
                mw_cloud = self.transform['cloud'](mw_cloud)
        return ir_sst, mw_sst, ir_cloud, mw_cloud

    def __getitem__(self, i):
        row = self.df.iloc[i]
        flip_sst = self._get_random_flip()
        flip_cloud = self._get_random_flip()
        return self._get_data_by_row(row, flip_sst, flip_cloud)

    def _get_data_by_row(self, row, flip_sst=None, flip_cloud=None):
        ir_sst, mw_sst = self.get_ir_mw_pair(self.sst_dir, row['ir'], row['mw'])
        ir_cloud, mw_cloud = self.get_ir_mw_pair(self.cloud_dir, row['cloud_ir'], row['cloud_mw'])
        ir_cloud = ir_cloud > 0
        mw_cloud = mw_cloud > 0

        # Custom random flip, so that the flip is applied to both IR and MW
        if flip_sst is not None:
            ir_sst, mw_sst = flip_sst(ir_sst, mw_sst)
        if flip_cloud is not None:
            ir_cloud, mw_cloud = flip_cloud(ir_cloud, mw_cloud)

        # Add nan regions to cloud mask after flipping SST
        ir_cloud = ir_cloud | torch.isnan(ir_sst)
        mw_cloud = mw_cloud | torch.isnan(mw_sst)

        # After this point, all tensors have shape (C, H, W)
        ir_sst, mw_sst, ir_cloud, mw_cloud = [
            add_channel_dim(x) for x in (ir_sst, mw_sst, ir_cloud, mw_cloud)
        ]

        # mw_sst = debias_mw(ir_sst, mw_sst, ir_cloud, mw_cloud)
        ir_sst, mw_sst, ir_cloud, mw_cloud = self._transform_data(
            ir_sst, mw_sst, ir_cloud, mw_cloud,
        )
        input_ir_base = downsample(ir_sst, ir_cloud)
        ir_base_cloud = torch.isnan(input_ir_base)
        input_ir_base = self.init_gaps(input_ir_base, ir_base_cloud)
        input_mw_base = self.init_gaps(mw_sst, mw_cloud)
        if (input_ir_base is None) and (input_mw_base is None):
            raise ValueError('Both MW and IR are completely cloudy')
        if input_ir_base is None:
            input_base = input_mw_base
        elif input_mw_base is None:
            input_base = input_ir_base
        else:
            input_base = (input_ir_base + input_mw_base) * 0.5
        input_base = upsample(input_base)

        input_ir = self.init_gaps(ir_sst, ir_cloud, method='microwave', microwave=input_base)
        return {
            'input_ir': input_ir, 'input_base': input_base,
            'gt_ir': ir_sst, 'cloud_ir': ir_cloud,
        }

    def get_ir_mw_pair(self, data_dir, ir_fname, mw_fname):
        _get_da = lambda fname: self.get_tile(data_dir, fname).sst.astype('float32')
        mw_da = _get_da(mw_fname)
        ir_da = _get_da(ir_fname)
        # mw_da = mw_da.interp_like(ir_da)
        ir = torch.from_numpy(ir_da.values)
        mw = torch.from_numpy(mw_da.values)

        # Some tiles are (112, 113)
        ir = ir[:112, :112]
        # mw = mw[:112, :112]
        return ir, mw


def get_input_target(data):
    input_base = data['input_base']
    return data['input_ir'] - input_base, data['gt_ir'] - input_base


def get_recon_target(data, pred):
    input_base = data['input_base']
    return pred + input_base, data['gt_ir']
