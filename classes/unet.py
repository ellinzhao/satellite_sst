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
        self.encoder_convs = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()

        self.decoder_convs = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()

        self.in_proj = nn.Conv2d(in_ch, in_ch, 1, **in_proj_kwargs)
        self.bottleneck_conv = ConvBlock(chs[-2], chs[-1], **out_proj_kwargs)
        self.out_proj = nn.Conv2d(chs[0], out_ch, 1)

        enc_chs = [in_ch] + chs[:-1]
        for i in range(len(enc_chs) - 1):
            self.encoder_convs.append(ConvBlock(enc_chs[i], enc_chs[i + 1]))
            self.encoder_downs.append(Downsample())
        for i in range(n_layers - 1):
            self.decoder_convs.append(ConvBlock(chs[i] + chs[i + 1], chs[i]))
            self.decoder_ups.append(Upsample(chs[i + 1]))
        self.decoder_convs = self.decoder_convs[::-1]
        self.decoder_ups = self.decoder_ups[::-1]

    def forward(self, x, return_z=False):
        x = self.in_proj(x)

        enc_feats = []
        for (conv, down) in zip(self.encoder_convs, self.encoder_downs):
            x = conv(x)
            enc_feats.append(x)
            x = down(x)
        x = z = self.bottleneck_conv(x)

        for up, conv, enc_feat in zip(self.decoder_ups, self.decoder_convs, enc_feats[::-1]):
            x = up(x)
            x = torch.cat([x, enc_feat], dim=1)
            x = conv(x)

        x = self.out_proj(x)
        if return_z:
            return x, z
        return x
