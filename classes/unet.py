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
        self, in_ch, out_ch, chs=[8, 16, 32, 64],
        in_proj_kwargs={}, out_proj_kwargs={},
    ):
        super().__init__()
        n_layers = len(chs)
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.in_proj = nn.Conv2d(in_ch, in_ch, 1, **in_proj_kwargs)
        self.bottleneck_conv = ConvBlock(chs[-2], chs[-1], **out_proj_kwargs)
        self.out_proj = nn.Conv2d(chs[0], out_ch, 1)

        enc_chs = [in_ch] + chs[:-1]
        for i in range(len(enc_chs) - 1):
            conv_i = ConvBlock(enc_chs[i], enc_chs[i + 1])
            down_i = Downsample()
            self.encoder_blocks.extend([conv_i, down_i])

        for i in range(n_layers - 1):
            conv_i = ConvBlock(chs[i] + chs[i + 1], chs[i])
            up_i = Upsample(chs[i + 1])
            self.decoder_blocks.extend([conv_i, up_i])

    def forward(self, x, return_z=False):
        x = self.in_proj(x)

        enc_feats = []

        for i in range(0, len(self.encoder_blocks), 2):
            conv_i = self.encoder_blocks[i]
            down_i = self.encoder_blocks[i + 1]
            x = conv_i(x)
            enc_feats.append(x)
            x = down_i(x)

        x = z = self.bottleneck_conv(x)

        for i in range(len(self.decoder_blocks), 0, -2):
            conv_i = self.decoder_blocks[i]
            up_i = self.decoder_blocks[i + 1]
            enc_feat_i = enc_feats[i // 2]
            x = up_i(x)
            x = torch.cat([x, enc_feat_i], dim=1)
            x = conv_i(x)

        x = self.out_proj(x)
        if return_z:
            return x, z

        return x
