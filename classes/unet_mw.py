import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pad=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=pad),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 3, padding=pad)

    def forward(self, x):
        return self.double_conv(x) + self.residual_conv(x)


class UNet(nn.Module):

    def __init__(self, in_ch=2, n_class=1, chs=[16, 24, 32]):
        super().__init__()
        self.dconv_down1 = ConvBlock(in_ch, chs[0], pad=2)
        self.dconv_down2 = ConvBlock(chs[0], chs[1])
        self.dconv_down3 = ConvBlock(chs[1], chs[2])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample3 = nn.ConvTranspose2d(chs[2], chs[2], 2, 2)
        self.upsample2 = nn.ConvTranspose2d(chs[1], chs[1], 2, 2)

        self.dconv_up2 = ConvBlock(chs[1] + chs[2], chs[1])
        self.dconv_up1 = ConvBlock(chs[1] + chs[0], chs[0])

        self.conv_last = nn.Conv2d(chs[0], n_class, 3)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
