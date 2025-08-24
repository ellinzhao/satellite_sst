import numpy as np
import pandas as pd

from .sst_dataset import SSTDataset


def index_to_col(df):
    return df.index.to_series().reset_index(drop=True)


class IRTripletDataset(SSTDataset):

    def __init__(
        self, var, sst_dir, cloud_dir, split, K=10, transform=None,
        fill={'method': 'constant', 'value': 0}, fnd_sst_path=None,
        return_coord=False, preload=True,
    ):
        super().__init__(
            var, sst_dir, cloud_dir, split, K=K, transform=transform, fill=fill,
            fnd_sst_path=fnd_sst_path, return_coord=return_coord, preload=preload,
        )

    def _generate_random_samples(self, K):
        N = len(self.sst_df)

        # Repeat each location K times
        self.df = self.sst_df.loc[self.sst_df.index.repeat(K)]
        idx = np.arange(len(self.df))
        self.df.set_index(idx, inplace=True)

        rng = np.random.default_rng()
        cloud_indices = np.hstack([
            rng.choice(len(self.cloud_df), replace=False, size=(2, K))  # shuffle=False should provide slight speedup?
            for _ in range(N)
        ])

        neg_sst_idx = []
        for i in range(N):
            ref_lat, ref_lon = self.sst_df.iloc[i][['lat_start', 'lon_start']]
            dist = np.sqrt((self.sst_df['lat_start'] - ref_lat)**2 + (self.sst_df['lon_start'] - ref_lon)**2)
            far_df = self.sst_df[dist > 5]  # 0.01° ~ 1km -> 5° ~ 500km
            neg_sst_idx.append(far_df.sample(n=K).index.to_numpy())
        neg_sst_idx = np.concat(neg_sst_idx)

        cols = ['ir', 'lat_start', 'lon_start']
        multiindex = pd.MultiIndex.from_product([['sst1', 'sst2', 'cloud1', 'cloud2'], cols])
        sst1 = self.df
        sst2 = self.sst_df.iloc[neg_sst_idx].set_index(idx)
        cloud1 = self.cloud_df.iloc[cloud_indices[0]].set_index(idx)
        cloud2 = self.cloud_df.iloc[cloud_indices[1]].set_index(idx)

        # The cloud location doesn't matter, but we add it so the multi-index is valid
        for df in (cloud1, cloud2):
            df['lat_start'] = [np.nan] * len(df)
            df['lon_start'] = [np.nan] * len(df)
        triplet_df = pd.concat(
            [df_[cols] for df_ in (sst1, sst2, cloud1, cloud2)], axis=1
        )
        triplet_df.columns = multiindex
        self.df = triplet_df

    def _get_triplet_row(self, sst, cloud, row):
        # sst is one of ['sst1', 'sst2'] i.e. the positive and negative labels
        ks = [
            (sst, 'ir'), (cloud, 'ir'), (sst, 'lat_start'), (sst, 'lon_start')
        ]
        # These are the columns that the parent class expects to receive
        new_cols = ['ir', 'cloud', 'lat_start', 'lon_start']
        row = row.loc[ks]
        row.index = new_cols
        return row

    def __getitem__(self, i):
        row = self.df.iloc[i]

        # can add more flips for sst1 and sst2, rather than applying the same flip for sst1 and sst2
        flip_sst = self._get_random_flip()
        flip_cloud = self._get_random_flip()
        rows = [
            ('x1_m1', self._get_triplet_row('sst1', 'cloud1', row)),
            ('x1_m2', self._get_triplet_row('sst1', 'cloud2', row)),
            ('x2_m1', self._get_triplet_row('sst2', 'cloud1', row)),
        ]
        return {
            k: self._get_data_by_row(row, flip_sst, flip_cloud)
            for k, row in rows
        }
