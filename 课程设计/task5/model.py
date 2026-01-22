import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=gn_groups, num_channels=out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=gn_groups, num_channels=out_c)

        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 推荐：双线性上采样 + 1x1对齐通道，减少棋盘格伪影
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1x1 = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv1x1(x)

        # 保险：万一尺寸对不上就插值到skip尺寸（一般不会发生，但加上更稳）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.e1 = encoder_block(in_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b  = conv_block(512, 1024)

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # Deep Supervision heads
        self.out1 = nn.Conv2d(512, 1, kernel_size=1)  # from d1
        self.out2 = nn.Conv2d(256, 1, kernel_size=1)  # from d2
        self.out3 = nn.Conv2d(128, 1, kernel_size=1)  # from d3
        self.out4 = nn.Conv2d(64,  1, kernel_size=1)  # from d4 (final)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b,  s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        y1 = self.out1(d1)
        y2 = self.out2(d2)
        y3 = self.out3(d3)
        y4 = self.out4(d4)

        # 全部对齐到输入大小
        size = inputs.shape[2:]
        y1 = F.interpolate(y1, size=size, mode="bilinear", align_corners=False)
        y2 = F.interpolate(y2, size=size, mode="bilinear", align_corners=False)
        y3 = F.interpolate(y3, size=size, mode="bilinear", align_corners=False)
        # y4 已经是原尺寸

        return [y1, y2, y3, y4]
