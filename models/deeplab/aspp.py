import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, image_lvl_feat):
        super(ASPP, self).__init__()

        inplanes = {
            'mobilenet': 320,
            'resnet10': 512,
            'resnet18': 512,
            'resnet10-bi': 2560,
            'resnet50': 2048,
            'resnet101': 2048,
        }[backbone]

        dilations = {
            8: [1, 12, 24, 36],
            16: [1, 6, 12, 18]
        }[output_stride]

        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU()) if image_lvl_feat else None

        planes_conv1 = 256 * (5 if image_lvl_feat else 4)
        self.ConvBnRelu = nn.Sequential(
            nn.Conv2d(planes_conv1, 256, 1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x_s = [
            self.aspp1(x),
            self.aspp2(x),
            self.aspp3(x),
            self.aspp4(x),
        ]

        if self.global_avg_pool is not None:
            out_shape = x_s[0].shape[2:]
            pool = self.global_avg_pool(x)
            x_s.append(F.interpolate(pool, size=out_shape, mode='bilinear', align_corners=True))

        x = torch.cat(x_s, dim=1)
        x = self.ConvBnRelu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPPModule(nn.Sequential):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPPModule, self).__init__(
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            BatchNorm(planes),
            nn.ReLU(),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

