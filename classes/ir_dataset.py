import torch

from .sst_dataset import SSTDataset, add_channel_dim


class IRDataset(SSTDataset):

    def __init__(
        self, sst_dir, cloud_dir, split, preload=True, transform=None,
        K=10, fill={'method': 'constant', 'value': 0},
    ):
        assert fill['method'] not in ['microwave', 'smooth'], 'Cannot use MW-based fill for the IR only dataset'
        super().__init__(
            sst_dir, cloud_dir, split, preload=preload,
            transform=transform, K=K, fill=fill,
        )

    def preload_tiles(self, data_dir, df):
        for _, row in df.iterrows():
            self.get_tile(data_dir, row['ir'], save_to_cache=True)

    def init_gaps(self, sst, cloud, method=None, **kwargs):
        assert method not in ['microwave', 'smooth'], 'Cannot use MW-based fill for the IR only dataset'
        return super().init_gaps(sst, cloud, method=method, **kwargs)

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

    def _transform_data(self, ir_sst, ir_cloud):
        if self.transform is not None:
            if 'sst' in self.transform:
                ir_sst = self.transform['sst'](ir_sst)
            if 'cloud' in self.transform:
                ir_cloud = self.transform['cloud'](ir_cloud)
        return ir_sst, ir_cloud

    def __getitem__(self, i):
        row = self.df.iloc[i]
        flip_sst = self._get_random_flip()
        flip_cloud = self._get_random_flip()
        return self._get_data_by_row(row, flip_sst, flip_cloud)

    def _get_data_by_row(self, row, flip_sst=None, flip_cloud=None):
        ir_sst, _ = self.get_ir_mw_pair(self.sst_dir, row['ir'], row['mw'])
        ir_cloud, _ = self.get_ir_mw_pair(self.cloud_dir, row['cloud_ir'], row['cloud_mw'])
        ir_cloud = ir_cloud > 0

        # Custom random flip, so that the flip is applied to both IR and MW
        if flip_sst is not None:
            ir_sst = flip_sst(ir_sst)
        if flip_cloud is not None:
            ir_cloud = flip_cloud(ir_cloud)

        # Add nan regions to cloud mask after flipping SST
        ir_cloud = ir_cloud | torch.isnan(ir_sst)

        # After this point, all tensors have shape (C, H, W)
        ir_sst, ir_cloud = [add_channel_dim(x) for x in (ir_sst, ir_cloud)]

        ir_sst, ir_cloud = self._transform_data(ir_sst, ir_cloud)
        input_ir = self.init_gaps(ir_sst, ir_cloud)
        return {
            'input_ir': input_ir, 'gt_ir': ir_sst, 'cloud_ir': ir_cloud,
        }


def get_input_target(data):
    return data['input_ir'], data['gt_ir']
