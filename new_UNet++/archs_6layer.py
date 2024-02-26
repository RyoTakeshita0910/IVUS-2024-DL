import torch
from torch import nn

__all__ = ['NestedUNet']


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


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024, 2048, 4096]

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

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], 3)
        self.conv4_1 = VGGBlock(nb_filter[4]+nb_filter[5], nb_filter[4], nb_filter[4], 3)
        self.conv5_1 = VGGBlock(nb_filter[5]+nb_filter[6], nb_filter[5], nb_filter[5], 3)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv3_2 = VGGBlock(nb_filter[3]*2+nb_filter[4], nb_filter[3], nb_filter[3], 3)
        self.conv4_2 = VGGBlock(nb_filter[4]*2+nb_filter[5], nb_filter[4], nb_filter[4], 3)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_3 = VGGBlock(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2], 3)
        self.conv3_3 = VGGBlock(nb_filter[3]*3+nb_filter[4], nb_filter[3], nb_filter[3], 3)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_4 = VGGBlock(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1], 3)
        self.conv2_4 = VGGBlock(nb_filter[2]*4+nb_filter[3], nb_filter[2], nb_filter[2], 3)

        self.conv0_5 = VGGBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0], 3)
        self.conv1_5 = VGGBlock(nb_filter[1]*5+nb_filter[2], nb_filter[1], nb_filter[1], 3)

        self.conv0_6 = VGGBlock(nb_filter[0]*6+nb_filter[1], nb_filter[0], nb_filter[0], 3)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final6 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
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

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output5 = self.final5(x0_5)
            output6 = self.final6(x0_6)
            return [output1, output2, output3, output4, output5, output6]

        else:
            output = self.final(x0_6)
            return output
