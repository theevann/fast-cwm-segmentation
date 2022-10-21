import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.slim_cwm_ops import CWMConv2d, SlimBatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return CWMConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, slim=True, temporal=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = SlimBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = SlimBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = CWMConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SlimBatchNorm2d(planes)
        self.conv2 = CWMConv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = SlimBatchNorm2d(planes)
        self.conv3 = CWMConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = SlimBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, multigrid=True, pretrained=True, url=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        layer4_fctn = self._make_layer
        if multigrid:
            layers[3] = [1, 2, 4]
            layer4_fctn = self._make_MG_unit
        
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = CWMConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, temporal=False)
        self.bn1 = SlimBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = layer4_fctn(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self._init_weight()

        if pretrained:
            self.url = url
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                CWMConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, temporal=False),
                SlimBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        ''' MultiGrid Unit '''
        # assert False, "Should not be here"
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                CWMConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, temporal=False),
                SlimBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(self.url)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ResNet_10(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, bi=False):
        self.inplanes = 64
        super(ResNet_10, self).__init__()

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        double = 5 if bi else 1

        # Modules
        self.conv1 = CWMConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, temporal=False)
        self.bn1 = SlimBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], double=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], double=double)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], double=double)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], double=double)
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, double=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                CWMConv2d(self.inplanes * double, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False, temporal=False),
                SlimBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes * double, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x

        end.record()
        torch.cuda.synchronize()
        print("llf", start.elapsed_time(end))

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, multigrid=True, pretrained=pretrained, url=model_urls['101'])
    return model


def ResNet50(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, multigrid=True, pretrained=pretrained, url=model_urls['50'])
    return model


def ResNet18(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, multigrid=False, pretrained=pretrained, url=model_urls['18'])
    return model


def ResNet10(output_stride, BatchNorm, pretrained=False, bi=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        print('No pretrained model for ResNet10')
    model = ResNet_10(BasicBlock, [1, 1, 1, 1], output_stride, BatchNorm, bi=bi)
    return model


model_urls = {
    '18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

if __name__ == "__main__":
    import torch
    model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
