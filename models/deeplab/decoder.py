import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        low_level_inplanes = {
            'resnet101': 256,
            'resnet50': 256,
            'resnet18': 64,
            'resnet10': 64,
            'mobilenet': 24
        }[backbone]

        self.ConvBnRelu = nn.Sequential(
            nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
            BatchNorm(48),
            nn.ReLU()
        )

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.ConvBnRelu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()