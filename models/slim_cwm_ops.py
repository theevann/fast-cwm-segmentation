import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint


#################################################
# # Building blocks for ResNet
# Including CWMConv2d

class SlimBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        in_ch = input.size(1)
        output = F.batch_norm(input, self.running_mean[:in_ch], self.running_var[:in_ch], self.weight[:in_ch], self.bias[:in_ch], self.training, self.momentum, self.eps)
        return output


class CWMConv2d(nn.Conv2d):    
    def __init__(self, *args, slim=True, temporal=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.last = None
        self.masks = None
        self.stepper = None
        self.mask_generator = None
        self.width_mult = 1.0
        self.slim = slim
        self.temporal = temporal

    def init_masks(self):
        assert self.masks is None
        assert self.mask_generator is not None
        self.masks = self.mask_generator.get_masks(self.out_channels_active)
        
    @property
    def out_channels_active(self):
        return int(self.out_channels * self.width_mult) if self.slim else self.out_channels

    def reset(self):
        self.last = None
    
    def forward(self, x):
        assert self.mask_generator is not None
        step = self.stepper.value % len(self.masks)
        mask = self.masks[step]

        in_ch = x.size(1)
        out_ch = self.out_channels_active
        weight = self.weight[:out_ch, :in_ch] if self.slim else self.weight[:, :in_ch]
        bias = (self.bias[:out_ch] if self.slim else self.bias) if self.bias is not None else None

        if not self.temporal:
            return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        if self.last is not None:
            assert self.stepper.value > 0
            if self.training:
                self.last = self.last.clone()

            weight = weight[mask]
            bias = bias[mask] if bias is not None else None
            self.last[:, mask] = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            assert self.stepper.value == 0
            self.last = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        return self.last


#################################################
# Stepper and Mask Generators

class Stepper():
    def __init__(self):
        self.value = 0
    
    def step(self):
        self.value += 1
    
    def reset(self):
        self.value = 0


class BiStepMaskGen():
    def __init__(self, n_channel, proportions):
        self.n_channel = n_channel
        self.proportions = torch.Tensor(proportions)

        assert sum(proportions) == 1
        assert len(proportions) == 2
        
        self.masks = self._create_all_masks()
        self.num_masks = len(self.masks)

    def __getitem__(self, step):
        return self.masks[step % self.num_masks]

    def __len__(self):
        return self.num_masks

    def _create_all_masks(self):
        n_ch_1 = self.proportions[0] * self.n_channel
        start = torch.div(self.n_channel - n_ch_1, 2, rounding_mode='floor')
        stop = torch.div(self.n_channel + n_ch_1, 2, rounding_mode='floor')
        
        return [
            slice(0, int(stop.item()), None),
            slice(int(start.item()), None, None)
        ]


class RandMaskGen():
    def __init__(self, n_channel, proportions):
        self.n_channel = n_channel
        self.size = int(proportions*n_channel)
        assert 0 < proportions <= 1

    def __len__(self):
        return 1

    def __getitem__(self, step):
        start = randint(0, self.n_channel - self.size)
        return slice(start, start + self.size, None)


class MaskGenerator():
    def __init__(self, proportions, generator="bistep"):
        self.proportions = proportions
        self.generator = {"bistep": BiStepMaskGen, "rand": RandMaskGen}[generator]
    
    def get_masks(self, n_channel):
        return self.generator(n_channel, self.proportions)


