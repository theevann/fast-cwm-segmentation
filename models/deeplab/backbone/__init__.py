from models.deeplab.backbone import resnet, mobilenet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm)
    elif backbone == 'resnet18':
        return resnet.ResNet18(output_stride, BatchNorm)
    elif backbone == 'resnet10':
        return resnet.ResNet10(output_stride, BatchNorm)
    elif backbone == 'resnet10-bi':
        return resnet.ResNet10(output_stride, BatchNorm, bi=True)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
