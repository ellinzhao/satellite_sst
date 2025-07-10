from .sst_dataset import SSTDataset
from geopy import distance
import pandas as pd


def index_to_col(df):
    return df.index.to_series().reset_index(drop=True)


class TripletSSTDatset(SSTDataset):

    def __init__(
        self, sst_dir, cloud_dir, split, preload=True, transform=None,
        K=10, fill={'method': 'constant', 'value': 0},
    ):
        super().__init__(sst_dir, cloud_dir, split, preload, transform, K, fill)

        # circumference of Earth ~40,000 km
        self.triplet_ranges = {
            'positive': (10, 800),
            'negative': (2000, 50000),
        }
        self.triplet_df = self._match_triplets()

    def _match_triplets(self):
        mw_locations = self.df['mw'].unique()
        dataset_triplets = []

        # Uses `great circle` distance calculation, which assumes the earth is a perfect sphere
        # This distance calculation is not as accurate at geodesic distance, but it ~8x faster

        for loc in mw_locations:
            skip = False
            loc_df = self.df[self.df['mw'] == loc]
            other_df = self.df[self.df['mw'] != loc]
            n_loc = len(loc_df)

            loc_point = loc_df.iloc[0]['mw_point']
            dist = other_df.apply(
                lambda row: distance.great_circle(row['mw_point'], loc_point).km,
                axis=1
            )

            idx_data = {'anchor': index_to_col(loc_df)}
            for k, dist_range in self.triplet_ranges.items():
                _df = other_df[(dist >= dist_range[0]) & (dist < dist_range[1])]

                if len(_df) == 0:
                    skip = True
                    break

                _df = _df.sample(n=n_loc, replace=n_loc > len(_df))
                idx_data[k] = index_to_col(_df)

            if skip:
                continue
            dataset_triplets.append(pd.DataFrame(idx_data))

        dataset_df = pd.concat(dataset_triplets, axis='index')
        return dataset_df.reset_index(drop=True)

    def __len__(self):
        return len(self.triplet_df)

    def __getitem__(self, i):
        row = self.triplet_df.iloc[i]
        return {
            # Despite init calls working with only `super()...`, this call
            # requires the Python 2 syntax for calling parent methods
            k: super(TripletSSTDatset, self).__getitem__(row[k])
            for k in ('anchor', 'positive', 'negative')
        }
