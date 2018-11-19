import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv2dLeaky(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2dLeaky, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, 0.1, inplace=True)


class RRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_size = 256

        bottleneck_size = [256, 128]
        self.reduce_dim_z = BasicConv2d(input_size * 2, bottleneck_size[0], kernel_size=1, padding=0)
        self.s_atten_z = nn.Sequential(
            nn.Conv2d(1, bottleneck_size[1], kernel_size=8, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_size[1], 64, kernel_size=1, padding=0, bias=False))
        self.c_atten_z = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(bottleneck_size[0], input_size, kernel_size=1, padding=0, bias=False))

    def generate_attention_z(self, x):
        z = self.reduce_dim_z(x)
        atten_s = self.s_atten_z(z.mean(dim=1, keepdim=True)).view(z.size(0), 1, z.size(2), z.size(3))
        atten_c = self.c_atten_z(z)
        z = F.sigmoid(atten_s * atten_c)
        return z, 1 - z

    def forward(self, x):
        '''

        :param x: raw features (B, T, C, H, W)
        :return: refined features  (B, C, T, H, W)
        '''

        if x.dim() == 4:
            x = x.view((1,) + x.size())
        assert x.dim() == 5

        video_num = x.size(0)
        depth = x.size(1)

        res = torch.cat((x[:, 0].contiguous().view((x.size(0), 1) + x.size()[2:]), x), dim=1)
        res = res[:, :-1]
        res = x - res

        h = x[:, 0]
        output = []
        for t in range(depth):
            con_fea = torch.cat((h - x[:, t], res[:, t]), dim=1)
            z_p, z_r = self.generate_attention_z(con_fea)
            h = z_r * h + z_p * x[:, t]
            output.append(h)

        fea = torch.stack(output, dim=2)
        return fea


class RRUV2(RRU):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_size = 256

        bottleneck_size = [256, 128]
        self.reduce_dim_z = BasicConv2dLeaky(input_size * 2, bottleneck_size[0], kernel_size=1, padding=0)
        self.s_atten_z = nn.Sequential(
            BasicConv2dLeaky(1, bottleneck_size[1], kernel_size=8, padding=0),
            nn.Conv2d(bottleneck_size[1], 64, kernel_size=1, padding=0, bias=False))
        self.c_atten_z = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(bottleneck_size[0], input_size, kernel_size=1, padding=0, bias=False))



