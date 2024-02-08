import torch
from torch import nn

__all__ = ['UNet', 'NestedUNet','NestedUNet7','DPUNet','NestedDPUNet']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size, padding=int(kernel_size/2))
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size, padding=int(kernel_size/2))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], 7)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], 5)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], 3)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], 3)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], 3)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], 3)
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], 3)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


# 本研究のネットワーク：7層U-Net++
class NestedUNet7(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024, 2048, 4096]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], 3)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], 3)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], 3)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], 3)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], 3)
        self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5], 3)
        self.conv6_0 = VGGBlock(nb_filter[5], nb_filter[6], nb_filter[6], 3)
        self.conv7_0 = VGGBlock(nb_filter[6], nb_filter[7], nb_filter[7], 3)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], 3)
        self.conv4_1 = VGGBlock(nb_filter[4]+nb_filter[5], nb_filter[4], nb_filter[4], 3)
        self.conv5_1 = VGGBlock(nb_filter[5]+nb_filter[6], nb_filter[5], nb_filter[5], 3)
        self.conv6_1 = VGGBlock(nb_filter[6]+nb_filter[7], nb_filter[6], nb_filter[6], 3)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv3_2 = VGGBlock(nb_filter[3]*2+nb_filter[4], nb_filter[3], nb_filter[3], 3)
        self.conv4_2 = VGGBlock(nb_filter[4]*2+nb_filter[5], nb_filter[4], nb_filter[4], 3)
        self.conv5_2 = VGGBlock(nb_filter[5]*2+nb_filter[6], nb_filter[5], nb_filter[5], 3)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_3 = VGGBlock(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv3_3 = VGGBlock(nb_filter[3]*3+nb_filter[4], nb_filter[3], nb_filter[3], 3)
        self.conv4_3 = VGGBlock(nb_filter[4]*3+nb_filter[5], nb_filter[4], nb_filter[4], 3)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_4 = VGGBlock(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_4 = VGGBlock(nb_filter[2]*4+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv3_4 = VGGBlock(nb_filter[3]*4+nb_filter[4], nb_filter[3], nb_filter[3], 3)

        self.conv0_5 = VGGBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_5 = VGGBlock(nb_filter[1]*5+nb_filter[2], nb_filter[1], nb_filter[1], 3)

        self.conv2_5 = VGGBlock(nb_filter[2]*5+nb_filter[3], nb_filter[2], nb_filter[2], 3)

        self.conv0_6 = VGGBlock(nb_filter[0]*6+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_6 = VGGBlock(nb_filter[1]*6+nb_filter[2], nb_filter[1], nb_filter[1], 3)

        self.conv0_7 = VGGBlock(nb_filter[0]*7+nb_filter[1], nb_filter[0], nb_filter[0], 3)

        # Deep supervision
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final6 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final7 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        # Deep supervisionなし
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        x6_0 = self.conv6_0(self.pool(x5_0))
        x5_1 = self.conv5_1(torch.cat([x5_0, self.up(x6_0)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.up(x4_2)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.up(x1_5)], 1))

        x7_0 = self.conv7_0(self.pool(x6_0))
        x6_1 = self.conv6_1(torch.cat([x6_0, self.up(x7_0)], 1))
        x5_2 = self.conv5_2(torch.cat([x5_0, x5_1, self.up(x6_1)], 1))
        x4_3 = self.conv4_3(torch.cat([x4_0, x4_1, x4_2, self.up(x5_2)], 1))
        x3_4 = self.conv3_4(torch.cat([x3_0, x3_1, x3_2, x3_3, self.up(x4_3)], 1))
        x2_5 = self.conv2_5(torch.cat([x2_0, x2_1, x2_2, x2_3, x2_4, self.up(x3_4)], 1))
        x1_6 = self.conv1_6(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, self.up(x2_5)], 1))
        x0_7 = self.conv0_7(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, self.up(x1_6)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output5 = self.final5(x0_5)
            output6 = self.final6(x0_6)
            output7 = self.final7(x0_7)
            return [output1, output2, output3, output4, output5, output6, output7]

        else:
            output = self.final(x0_7)
            return output


# 既存研究のネットワーク
class MainBranch(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out
    

class RefrainBranch(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out
    

class DownSamplingBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.prelu = nn.PReLU()
        self.conv = nn.Conv2d(in_channels, in_channels, 2, stride=2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = self.prelu(conv_out)
        conv_out = self.bn(conv_out)

        ref_out = self.pool(x)

        out = torch.cat([conv_out,ref_out],1)

        return out

class UpSamplingBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(in_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        out = self.up(x)
        out = self.prelu(out)
        out = self.bn(out)

        return out


# IVUS-Netのネットワーク
class DPUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512]

        self.prelu = nn.PReLU()

        # Encoder input 64ch
        self.main0_0 = MainBranch(input_channels, nb_filter[0], nb_filter[0])
        self.ref0_0  = RefrainBranch(input_channels, nb_filter[0], nb_filter[0])

        # input 128ch
        self.down1_0 = DownSamplingBranch(nb_filter[0])
        self.main1_0 = MainBranch(nb_filter[1], nb_filter[1], nb_filter[1])
        self.ref1_0  = RefrainBranch(nb_filter[1], nb_filter[1], nb_filter[1])

        # input 256ch
        self.down2_0 = DownSamplingBranch(nb_filter[1])
        self.main2_0 = MainBranch(nb_filter[2], nb_filter[2], nb_filter[2])
        self.ref2_0  = RefrainBranch(nb_filter[2], nb_filter[2], nb_filter[2])

        # input 512ch
        self.down3_0 = DownSamplingBranch(nb_filter[2])
        self.main3_0 = MainBranch(nb_filter[3], nb_filter[3], nb_filter[3])
        self.ref3_0  = RefrainBranch(nb_filter[3], nb_filter[3], nb_filter[3])


        # Decoder
        self.up2_1   = UpSamplingBranch(nb_filter[3])
        self.main2_1 = MainBranch(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.ref2_1  = RefrainBranch(nb_filter[3], nb_filter[2], nb_filter[2])

        self.up1_2   = UpSamplingBranch(nb_filter[2])
        self.main1_2 = MainBranch(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.ref1_2  = RefrainBranch(nb_filter[2], nb_filter[1], nb_filter[1])

        self.up0_3   = UpSamplingBranch(nb_filter[1])
        self.main0_3 = MainBranch(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.ref0_3  = RefrainBranch(nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.prelu(torch.add(self.main0_0(input), self.ref0_0(input)))
        x1_0 = self.down1_0(x0_0)
        x1_0 = self.prelu(torch.add(self.main1_0(x1_0), self.ref1_0(x1_0)))
        x2_0 = self.down2_0(x1_0)
        x2_0 = self.prelu(torch.add(self.main2_0(x2_0), self.ref2_0(x2_0)))
        x3_0 = self.down3_0(x2_0)
        x3_0 = self.prelu(torch.add(self.main3_0(x3_0), self.ref3_0(x3_0)))

        x2_1 = self.up2_1(x3_0)
        x2_1 = self.prelu(torch.add(self.main2_1(torch.cat([x2_0, x2_1], 1)), self.ref2_1(x2_1)))
        x1_2 = self.up1_2(x2_1)
        x1_2 = self.prelu(torch.add(self.main1_2(torch.cat([x1_0, x1_2], 1)), self.ref1_2(x1_2)))
        x0_3 = self.up0_3(x1_2)
        x0_3 = self.prelu(torch.add(self.main0_3(torch.cat([x0_0, x0_3], 1)), self.ref0_3(x0_3)))

        output = self.final(x0_3)
        return output


# IVUS-U-Net++のネットワーク
class NestedDPUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.prelu = nn.PReLU()

        # Encoder
        self.main0_0 = MainBranch(input_channels, nb_filter[0], nb_filter[0])
        self.ref0_0  = RefrainBranch(input_channels, nb_filter[0], nb_filter[0])

        self.down1_0 = DownSamplingBranch(nb_filter[0])
        self.main1_0 = MainBranch(nb_filter[1], nb_filter[1], nb_filter[1])
        self.ref1_0  = RefrainBranch(nb_filter[1], nb_filter[1], nb_filter[1])

        self.down2_0 = DownSamplingBranch(nb_filter[1])
        self.main2_0 = MainBranch(nb_filter[2], nb_filter[2], nb_filter[2])
        self.ref2_0  = RefrainBranch(nb_filter[2], nb_filter[2], nb_filter[2])

        self.down3_0 = DownSamplingBranch(nb_filter[2])
        self.main3_0 = MainBranch(nb_filter[3], nb_filter[3], nb_filter[3])
        self.ref3_0  = RefrainBranch(nb_filter[3], nb_filter[3], nb_filter[3])

        self.down4_0 = DownSamplingBranch(nb_filter[3])
        self.main4_0 = MainBranch(nb_filter[4], nb_filter[4], nb_filter[4])
        self.ref4_0  = RefrainBranch(nb_filter[4], nb_filter[4], nb_filter[4])

        # Nest1
        self.up0_1   = UpSamplingBranch(nb_filter[1])
        self.main0_1 = MainBranch(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.ref0_1  = RefrainBranch(nb_filter[1], nb_filter[0], nb_filter[0])

        self.up1_1   = UpSamplingBranch(nb_filter[2])
        self.main1_1 = MainBranch(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.ref1_1  = RefrainBranch(nb_filter[2], nb_filter[1], nb_filter[1])

        self.up2_1   = UpSamplingBranch(nb_filter[3])
        self.main2_1 = MainBranch(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.ref2_1  = RefrainBranch(nb_filter[3], nb_filter[2], nb_filter[2])

        self.up3_1   = UpSamplingBranch(nb_filter[4])
        self.main3_1 = MainBranch(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.ref3_1  = RefrainBranch(nb_filter[4], nb_filter[3], nb_filter[3])

        # Nest2
        self.up0_2   = UpSamplingBranch(nb_filter[1])
        self.main0_2 = MainBranch(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.ref0_2  = RefrainBranch(nb_filter[1], nb_filter[0], nb_filter[0])

        self.up1_2   = UpSamplingBranch(nb_filter[2])
        self.main1_2 = MainBranch(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.ref1_2  = RefrainBranch(nb_filter[2], nb_filter[1], nb_filter[1])

        self.up2_2   = UpSamplingBranch(nb_filter[3])
        self.main2_2 = MainBranch(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.ref2_2  = RefrainBranch(nb_filter[3], nb_filter[2], nb_filter[2])

        # Nest3
        self.up0_3   = UpSamplingBranch(nb_filter[1])
        self.main0_3 = MainBranch(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.ref0_3  = RefrainBranch(nb_filter[1], nb_filter[0], nb_filter[0])

        self.up1_3   = UpSamplingBranch(nb_filter[2])
        self.main1_3 = MainBranch(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.ref1_3  = RefrainBranch(nb_filter[2], nb_filter[1], nb_filter[1])

        # Nest4
        self.up0_4   = UpSamplingBranch(nb_filter[1])
        self.main0_4 = MainBranch(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.ref0_4  = RefrainBranch(nb_filter[1], nb_filter[0], nb_filter[0])

        # Feature Pyramid Network(FPN)
        if self.deep_supervision:
            self.upf0 = nn.Upsample(scale_factor=2**(len(nb_filter)-1), mode='bilinear', align_corners=True)
            self.upf1 = nn.Upsample(scale_factor=2**(len(nb_filter)-2), mode='bilinear', align_corners=True)
            self.upf2 = nn.Upsample(scale_factor=2**(len(nb_filter)-3), mode='bilinear', align_corners=True)
            self.upf3 = nn.Upsample(scale_factor=2**(len(nb_filter)-4), mode='bilinear', align_corners=True)

            self.smooth0 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=3, padding=1)
            self.smooth1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=3, padding=1)
            self.smooth2 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=3, padding=1)
            self.smooth3 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=3, padding=1)

            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        # FPNなし
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)



    def forward(self, input):
        x0_0 = self.prelu(torch.add(self.main0_0(input), self.ref0_0(input)))
        x1_0 = self.down1_0(x0_0)
        x1_0 = self.prelu(torch.add(self.main1_0(x1_0), self.ref1_0(x1_0)))
        x0_1 = self.up0_1(x1_0)
        x0_1 = self.prelu(torch.add(self.main0_1(torch.cat([x0_0, x0_1], 1)), self.ref0_1(x0_1)))

        x2_0 = self.down2_0(x1_0)
        x2_0 = self.prelu(torch.add(self.main2_0(x2_0), self.ref2_0(x2_0)))
        x1_1 = self.up1_1(x2_0)
        x1_1 = self.prelu(torch.add(self.main1_1(torch.cat([x1_0,x1_1], 1)), self.ref1_1(x1_1)))
        x0_2 = self.up0_2(x1_1)
        x0_2 = self.prelu(torch.add(self.main0_2(torch.cat([x0_0, x0_1, x0_2], 1)), self.ref0_2(x0_2)))

        x3_0 = self.down3_0(x2_0)
        x3_0 = self.prelu(torch.add(self.main3_0(x3_0), self.ref3_0(x3_0)))
        x2_1 = self.up2_1(x3_0)
        x2_1 = self.prelu(torch.add(self.main2_1(torch.cat([x2_0,x2_1], 1)), self.ref2_1(x2_1)))
        x1_2 = self.up1_2(x2_1)
        x1_2 = self.prelu(torch.add(self.main1_2(torch.cat([x1_0, x1_1, x1_2], 1)), self.ref1_2(x1_2)))
        x0_3 = self.up0_3(x1_2)
        x0_3 = self.prelu(torch.add(self.main0_3(torch.cat([x0_0, x0_1, x0_2, x0_3], 1)), self.ref0_3(x0_3)))

        x4_0 = self.down4_0(x3_0)
        x4_0 = self.prelu(torch.add(self.main4_0(x4_0), self.ref4_0(x4_0)))
        x3_1 = self.up3_1(x4_0)
        x3_1 = self.prelu(torch.add(self.main3_1(torch.cat([x3_0,x3_1], 1)), self.ref3_1(x3_1)))
        x2_2 = self.up2_2(x3_1)
        x2_2 = self.prelu(torch.add(self.main2_2(torch.cat([x2_0, x2_1, x2_2], 1)), self.ref2_2(x2_2)))
        x1_3 = self.up1_3(x2_2)
        x1_3 = self.prelu(torch.add(self.main1_3(torch.cat([x1_0, x1_1, x1_2, x1_3], 1)), self.ref1_3(x1_3)))
        x0_4 = self.up0_4(x1_3)
        x0_4 = self.prelu(torch.add(self.main0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], 1)), self.ref0_4(x0_4)))


        if self.deep_supervision:
            fpn_0 = self.smooth0(self.upf0(x4_0))
            fpn_1 = self.smooth1(self.upf1(x3_1))
            fpn_2 = self.smooth2(self.upf2(x2_2))
            fpn_3 = self.smooth3(self.upf3(x1_3))

            output1 = self.final1(fpn_0)
            output2 = self.final2(fpn_1)
            output3 = self.final3(fpn_2)
            output4 = self.final4(fpn_3)
            output5 = self.final5(x0_4)
            
            return [output1, output2, output3, output4, output5]
        
        else:
            output = self.final(x0_4)
            return output
