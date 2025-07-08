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


class Decoder(nn.Module):

    def __init__(self, proj_ch, out_ch, chs):
        super().__init__()
        self.proj_conv = nn.Conv2d(proj_ch, chs[3], 3, dilation=2, padding=2)

        self.upsample4 = nn.ConvTranspose2d(chs[3], chs[3], 2, 2)
        self.upsample3 = nn.ConvTranspose2d(chs[2], chs[2], 2, 2)
        self.upsample2 = nn.ConvTranspose2d(chs[1], chs[1], 2, 2)

        self.dconv_up3 = ConvBlock(chs[3], chs[2])
        self.dconv_up2 = ConvBlock(chs[2], chs[1])
        self.dconv_up1 = ConvBlock(chs[1], chs[0])

        self.conv_last = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        x = self.proj_conv(x)
        x = self.upsample4(x)

        x = self.dconv_up3(x)
        x = self.upsample3(x)

        x = self.dconv_up2(x)
        x = self.upsample2(x)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


class Encoder(nn.Module):

    def __init__(self, in_ch, proj_ch, chs):
        super().__init__()
        self.conv_down1 = ConvBlock(in_ch, chs[0])
        self.conv_down2 = ConvBlock(chs[0], chs[1])
        self.conv_down3 = ConvBlock(chs[1], chs[2])
        self.conv_down4 = ConvBlock(chs[2], chs[3])
        self.maxpool = nn.MaxPool2d(2)
        self.proj_conv = nn.Conv2d(chs[3], proj_ch, 3, dilation=2, padding=2)

    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)

        x = self.conv_down4(x)

        # Project the features to the desired shape
        x = self.proj_conv(x)
        return x


class MMRecon(nn.Module):

    def __init__(self, chs=[24, 32, 48, 64], feat_ch=32):
        super().__init__()
        self.feat_ch = feat_ch
        self.mw_encoder = Encoder(1, feat_ch, chs)
        self.ir_encoder = Encoder(1, feat_ch * 2, chs)

        self.lr_decoder = Decoder(feat_ch, 1, chs)
        self.hr_decoder = Decoder(feat_ch, 1, chs)

        self.lr_feat_combiner = ConvBlock(2 * feat_ch, feat_ch)

    def forward(self, x_ir, x_mw):
        ir_feat = self.ir_encoder(x_ir)
        mw_feat = self.mw_encoder(x_mw)

        lr_feat = torch.cat([ir_feat[:, :self.feat_ch], mw_feat], dim=1)
        hr_feat = ir_feat[:, self.feat_ch:]
        lr_feat = self.lr_feat_combiner(lr_feat)

        x_lr = self.lr_decoder(lr_feat)
        x_hr = self.hr_decoder(hr_feat)
        return x_hr, x_lr
