import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.double_conv(x) + self.residual_conv(x)


class UNet(nn.Module):

    def __init__(self, in_ch=2, n_class=1, chs=[24, 32, 48, 64]):
        super().__init__()
        self.dconv_down1 = ConvBlock(in_ch, chs[0])
        self.dconv_down2 = ConvBlock(chs[0], chs[1])
        self.dconv_down3 = ConvBlock(chs[1], chs[2])
        self.dconv_down4 = ConvBlock(chs[2], chs[3])

        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.ConvTranspose2d(chs[3], chs[3], 2, 2)
        self.upsample3 = nn.ConvTranspose2d(chs[2], chs[2], 2, 2)
        self.upsample2 = nn.ConvTranspose2d(chs[1], chs[1], 2, 2)

        self.dconv_up3 = ConvBlock(chs[2] + chs[3], chs[2])
        self.dconv_up2 = ConvBlock(chs[1] + chs[2], chs[1])
        self.dconv_up1 = ConvBlock(chs[1] + chs[0], chs[0])

        self.conv_last = nn.Conv2d(chs[0], n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample4(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out
