import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo

from itertools import chain

from models.slim_cwm_ops import CWMConv2d, SlimBatchNorm2d


def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=False)


def reset(m):
    if isinstance(m, CWMConv2d):
        m.reset()


class BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1, slim=True, temporal=True):
        super(BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', SlimBatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm))
        self.add_module('conv', CWMConv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=(k // 2), bias=bias, dilation=dilation, slim=slim, temporal=temporal))


class Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(Upsample, self).__init__()
        self.bottleneck = BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i), BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse', BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, group=1, dilation=1):
    """3x3 convolution with padding"""
    return CWMConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=dilation, groups=group, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return CWMConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _bn_function_factory(conv, norm, relu=None):
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SlimBatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SlimBatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, efficient=True, use_bn=True):
        super(Bottleneck, self).__init__()
        norm_layer = SlimBatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.use_bn = use_bn
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width) if self.use_bn else None
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width) if self.use_bn else None
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        identity = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = _bn_function_factory(self.conv3, self.bn3)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        relu = self.relu(out)

        return relu, out


class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=False, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.conv1 = CWMConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, temporal=False)
        self.bn1 = SlimBatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        upsamples = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [Upsample(num_features, self.inplanes,
                                num_features, use_bn=self.use_bn, k=k_up)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [Upsample(num_features, self.inplanes,
                                num_features, use_bn=self.use_bn, k=k_up)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [Upsample(num_features, self.inplanes,
                                num_features, use_bn=self.use_bn, k=k_up)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [self.conv1, self.maxpool,
                          self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn)

        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [CWMConv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=stride, bias=False, temporal=False)]
            if self.use_bn:
                layers += [SlimBatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample,
                        efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes,
                             efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        # features += [skip]
        return features

    def forward_up(self, features):
        features = features[::-1]

        x = features[0]

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, image):
        # return self.forward_up(self.forward_down(image))
        x = self.forward_down(image)
        x = self.forward_up(x)
        return x


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    r"""Constructs a ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


class SwiftNetCWM(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(SwiftNetCWM, self).__init__()
        self.backbone = {
            "resnet18": resnet18,
            "resnet50": resnet50,
            "resnet101": resnet101,
        }[backbone]()
        self.num_classes = num_classes
        self.logits = BNReluConv(
            self.backbone.num_features, self.num_classes, batch_norm=use_bn, temporal=False, slim=False)

    def forward(self, batch):
        image_size = batch.shape[-2:]
        target_size = image_size[0] // 4, image_size[1] // 4
        feats, _ = self.backbone(batch)
        features = upsample(feats, target_size)
        logits = self.logits.forward(features)
        return upsample(logits, image_size)

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def reset(self):
        self.apply(reset)

    def set_width_mult(self, width_mult):
        self.apply(lambda m: setattr(m, "width_mult", width_mult) if hasattr(m, "width_mult") else None)

    def set_stepper(self, stepper):
        self.apply(lambda m: setattr(m, "stepper", stepper) if hasattr(m, "stepper") else None)
    
    def set_mask_generator(self, mask_generator):
        self.apply(lambda m: setattr(m, "mask_generator", mask_generator) if hasattr(m, "mask_generator") else None)
        self.apply(lambda m: m.init_masks() if hasattr(m, "mask_generator") else None)