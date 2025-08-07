from .ir_dataset import IRDataset
import numpy as np


def index_to_col(df):
    return df.index.to_series().reset_index(drop=True)


class IRTripletSSTDataset(IRDataset):

    def __init__(
        self, sst_dir, cloud_dir, split, preload=True, transform=None,
        K=10, fill={'method': 'constant', 'value': 0},
    ):
        super().__init__(sst_dir, cloud_dir, split, preload, transform, K, fill)

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
        # random idx that is +/- 2 away from each row

        # TODO: use a pandas multi-index here
        # Cloud idx 1
        random_cloud_df = self.cloud_df.iloc[cloud_indices[0]].set_index(idx)
        self.df['pos_cloud_ir'] = random_cloud_df['ir']
        self.df['pos_cloud_mw'] = random_cloud_df['mw']

        # Cloud idx 2
        random_cloud_df = self.cloud_df.iloc[cloud_indices[1]].set_index(idx)
        self.df['neg_cloud_ir'] = random_cloud_df['ir']
        self.df['neg_cloud_mw'] = random_cloud_df['mw']

        # SST
        random_sst_df = self.sst_df.iloc[neg_sst_idx].set_index(idx)
        self.df['neg_sst_ir'] = random_sst_df['ir']
        self.df['neg_sst_mw'] = random_sst_df['mw']

    def __getitem__(self, i):
        row = self.df.iloc[i]

        # can add more flips for sst1 and sst2, rather than applying the same flip for sst1 and sst2
        flip_sst = self._get_random_flip()
        flip_cloud = self._get_random_flip()

        x1_m1_row = row[['ir', 'mw', 'pos_cloud_ir', 'pos_cloud_mw']]
        x1_m1_row = x1_m1_row.rename({'pos_cloud_ir': 'cloud_ir', 'pos_cloud_mw': 'cloud_mw'})

        x1_m2_row = row[['ir', 'mw', 'neg_cloud_ir', 'neg_cloud_mw']]
        x1_m2_row = x1_m2_row.rename({'neg_cloud_ir': 'cloud_ir', 'neg_cloud_mw': 'cloud_mw'})

        x2_m1_row = row[['neg_sst_ir', 'neg_sst_mw', 'pos_cloud_ir', 'pos_cloud_mw']]
        x2_m1_row = x2_m1_row.rename({'neg_sst_ir': 'ir', 'neg_sst_mw': 'mw', 'pos_cloud_ir': 'cloud_ir', 'pos_cloud_mw': 'cloud_mw'})
        return {
            # Despite init calls working with only `super()...`, this call
            # requires the Python 2 syntax for calling parent methods
            k: self._get_data_by_row(row, flip_sst, flip_cloud)
            for k, row in zip(('x1_m1', 'x1_m2', 'x2_m1'), (x1_m1_row, x1_m2_row, x2_m1_row))
        }
