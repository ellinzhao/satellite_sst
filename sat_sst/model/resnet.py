import torch
import torch.nn as nn
import torchvision  # noqa: F401


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity
        nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_in')
        nn.init.constant_(self.conv.bias, 0.0)
        nn.init.normal_(self.bn.weight, 1.0, 0.02)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(
        self, in_channels, out_channels, upsampling_method='conv_transpose',
    ):
        super().__init__()

        if upsampling_method == 'conv_transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif upsampling_method == 'bilinear':
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(out_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class Resnet50Encoder(nn.Module):

    CHS = [64, 256, 512, 1024, 2048]

    def __init__(self, in_ch, n_layers, use_mask=False, mask_token_init_fn=lambda dim: torch.randn(dim)):
        super().__init__()
        assert n_layers <= 4 and n_layers > 0
        resnet = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
        resnet_children = list(resnet.children())

        self.input_block = ConvBlock(in_ch, self.CHS[0])
        self.mask_token = nn.Parameter(mask_token_init_fn(self.CHS[0])[None, :, None, None])
        self.input_pool = resnet_children[3]  # TODO(ellin): check if perf is worse without this
        # if this pool op is removed, add a pool op before the final conv

        down_blocks = []
        for bottleneck in resnet_children[4:]:
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks[:n_layers - 1])
        self.n_layers = n_layers
        self.in_ch = in_ch
        self.chs = Resnet50Encoder.CHS[:n_layers]
        self.use_mask = use_mask

    def forward(self, x, mask=None):
        if self.use_mask and mask is None:
            raise ValueError('model is setup to use a mask token but no mask is provided')
        if not self.use_mask and mask is not None:
            print('Mask provided but will not be used!')

        x = self.input_block(x)
        if self.use_mask:
            x = torch.where(mask, self.mask_token, x)

        x = self.input_pool(x)
        for block in self.down_blocks:
            x = block(x)
        return x


class Decoder(nn.Module):

    def __init__(self, chs, out_ch):
        super().__init__()
        up_blocks = []
        for i in range(len(chs) - 1):
            up_blocks.append(UpBlock(chs[i], chs[i + 1]))
        self.up_blocks = nn.ModuleList(up_blocks)  # TODO(ellin): change to nn.Sequential
        self.out = nn.Conv2d(chs[-1], out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        for block in self.up_blocks:
            x = block(x)
        x = self.out(x)
        return x


class ReconModel(nn.Module):

    def __init__(self, n_enc_layers=3, dec_data_chs=[384, 256, 64], dec_mask_chs=[128, 64, 32]):
        super().__init__()
        self.enc = Resnet50Encoder(1, n_enc_layers, use_mask=True)
        # can't change the chs of the resnet [64, 256, 512, 1024, 2048]

        self.enc_feat_dim = self.enc.chs[-1]
        self.data_feat_dim = dec_data_chs[0]
        self.mask_feat_dim = dec_mask_chs[0]

        assert self.data_feat_dim + self.mask_feat_dim == self.enc_feat_dim

        self.dec_data = Decoder(dec_data_chs, 1)
        self.dec_mask = Decoder(dec_mask_chs, 2)

    def forward(self, x, return_z=False, mask=None):
        z = self.enc(x, mask=mask)
        z_data = z[:, :self.data_feat_dim]
        z_mask = z[:, self.data_feat_dim:]
        data = self.dec_data(z_data)
        mask = self.dec_mask(z_mask)
        out = {'pred_sst': data, 'pred_mask': mask}
        if return_z:
            out['z_sst'] = z_data
            out['z_mask'] = z_mask
        return out
