from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplab_cwm.aspp import ASPP
from models.deeplab_cwm.decoder import Decoder
from models.deeplab_cwm.backbone import build_backbone
from models.slim_cwm_ops import CWMConv2d

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLabCWM(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                 norm_layer=None, freeze_bn=False, image_lvl_feat=True):
        super(DeepLabCWM, self).__init__()

        self.backbone = build_backbone(backbone, output_stride, norm_layer)
        self.aspp = ASPP(backbone, output_stride, norm_layer, image_lvl_feat)
        self.decoder = Decoder(num_classes, backbone, norm_layer)
        self.freeze_bn = freeze_bn

    def forward(self, inputs):
        x, low_level_feat = self.backbone(inputs)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        return x

    def train(self, *args, **kwargs):
        super(DeepLabCWM, self).train(*args, **kwargs)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                    m.eval()

    def fine_tune_params(self):
        modules = [self.backbone]
        return self.get_params(modules)

    def random_init_params(self):
        modules = [self.aspp, self.decoder]
        return self.get_params(modules)

    def get_params(self, modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                    for p in m.parameters():
                        if p.requires_grad:
                            yield p

    def reset(self):
        self.apply(lambda m: m.reset() if isinstance(m, CWMConv2d) else None)

    def set_width_mult(self, width_mult):
        self.apply(lambda m: setattr(m, "width_mult", width_mult) if hasattr(m, "width_mult") else None)

    def set_stepper(self, stepper):
        self.apply(lambda m: setattr(m, "stepper", stepper) if hasattr(m, "stepper") else None)
    
    def set_mask_generator(self, mask_generator):
        self.apply(lambda m: setattr(m, "mask_generator", mask_generator) if hasattr(m, "mask_generator") else None)
        self.apply(lambda m: m.init_masks() if hasattr(m, "mask_generator") else None)


if __name__ == "__main__":
    model = DeepLabCWM(backbone='mobilenet', output_stride=16)
    model.eval()
    inputs = torch.rand(1, 3, 513, 513)
    output = model(inputs)
    print(output.size())

    
