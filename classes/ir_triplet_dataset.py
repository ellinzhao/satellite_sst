import numpy as np
import pandas as pd

from .ir_dataset import IRDataset


def index_to_col(df):
    return df.index.to_series().reset_index(drop=True)


class IRTripletDataset(IRDataset):

    def __init__(
        self, sst_dir, cloud_dir, split, preload=True, transform=None,
        K=10, fill={'method': 'constant', 'value': 0}, return_coord=False,
    ):
        super().__init__(sst_dir, cloud_dir, split, preload, transform, K, fill, return_coord)

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
        idx_repeat = np.vstack([np.arange(N) for _ in range(K)]).ravel(order='F')

        neg_sst_idx = (rng.integers(50, N - 50, size=K * N) + idx_repeat) % N
        # random idx that is +/- 50 away from each row

        cols = ['ir', 'mw', 'mw_point']
        multiindex = pd.MultiIndex.from_product([['sst1', 'sst2', 'cloud1', 'cloud2'], cols])
        sst1 = self.df
        sst2 = self.sst_df.iloc[neg_sst_idx].set_index(idx)
        cloud1 = self.cloud_df.iloc[cloud_indices[0]].set_index(idx)
        cloud2 = self.cloud_df.iloc[cloud_indices[1]].set_index(idx)
        # The cloud location doesn't matter, but we add it so the multi-index is valid
        cloud1['mw_point'] = [(np.nan, np.nan), ] * len(cloud1)
        cloud2['mw_point'] = [(np.nan, np.nan), ] * len(cloud1)
        triplet_df = pd.concat(
            [df_[cols] for df_ in (sst1, sst2, cloud1, cloud2)], axis=1
        )
        triplet_df.columns = multiindex
        self.df = triplet_df

    def _get_triplet_row(self, sst, cloud, row):
        ks = [
            (sst, 'ir'), (sst, 'mw'), (sst, 'mw_point'), (cloud, 'ir'), (cloud, 'mw')
        ]
        # These are the columns that the parent class expects to receive
        new_cols = ['ir', 'mw', 'mw_point', 'cloud_ir', 'cloud_mw']
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
