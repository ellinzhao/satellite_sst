import warnings
from collections import OrderedDict

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, dilation=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 3, dilation=2, padding=2)

    def forward(self, x):
        return self.double_conv(x) + self.residual_conv(x)


class Upsample(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.conv_T = nn.ConvTranspose2d(ch, ch, 2, 2)

    def forward(self, x):
        return self.conv_T(x)


class Downsample(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)


class UNet(nn.Module):

    def __init__(
        self, in_ch, in_proj_ch, out_ch, chs=[8, 16, 32, 64], use_loc=False,
        mask_token_init_fn=lambda dim: torch.randn(dim),
    ):
        super().__init__()
        self.chs = chs
        self.use_loc = use_loc
        self.mask_token = nn.Parameter(mask_token_init_fn(in_proj_ch)[None, :, None, None])
        # nn.Parameter(torch.randn(1, in_proj_ch, 1, 1))
        self.in_proj = nn.Conv2d(in_ch, in_proj_ch, 1)
        self.bottleneck_conv = ConvBlock(chs[-1], 2 * chs[-1])
        self.upsample = nn.Upsample(scale_factor=7, mode='nearest')

        enc_dict = OrderedDict()
        enc_chs = [in_proj_ch] + chs
        for i in range(len(enc_chs) - 1):
            enc_dict[f'conv{i}'] = ConvBlock(enc_chs[i], enc_chs[i + 1])
            enc_dict[f'down{i}'] = Downsample()

        data_dec_dict = OrderedDict()
        mask_dec_dict = OrderedDict()
        dec_chs = chs[::-1] + [out_ch]
        for i in range(len(dec_chs) - 1):
            for dct in [data_dec_dict, mask_dec_dict]:
                dct[f'up{i}'] = Upsample(dec_chs[i])
                dct[f'conv{i}'] = ConvBlock(dec_chs[i], dec_chs[i + 1])
        self.encoder = nn.Sequential(enc_dict)
        self.data_decoder = nn.Sequential(data_dec_dict)
        self.mask_decoder = nn.Sequential(mask_dec_dict)

    def forward(self, x, mask=None, return_z=False, loc_emb=None):
        x = self.in_proj(x)
        if mask is not None:
            x = torch.where(mask, self.mask_token, x)

        x = self.encoder(x)
        x = self.bottleneck_conv(x)  # Output is (b, n_z, h_z, w_z)
        # Half of the feats are mask, half are data

        feat_ch = self.chs[-1]
        x_data = z_data = x[:, :feat_ch]
        x_mask = z_mask = x[:, feat_ch:]

        if self.use_loc:
            assert loc_emb is not None
            x_data = x_data + self.upsample(loc_emb[..., None, None])
        else:
            if loc_emb is not None:
                warnings.warn('`loc_emb` is supplied, but the model is not setup to use the variable. Initialize the model with `use_loc=True`.')

        x_data = self.data_decoder(x_data)
        x_mask = self.data_decoder(x_mask)

        if return_z:
            return x_data, x_mask, z_data, z_mask

        return x_data, x_mask
