import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms.functional as TF


def deconstr(sample):
    return sample['image'], sample['label']


def constr(img, mask):
    return {'image': img, 'label': mask}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, mask = deconstr(sample)
        img = TF.normalize(img, self.mean, self.std)
        return constr(img, mask)


class ToTensor(object):
    """Convert ndarrays is not None to Tensors."""

    def __call__(self, sample):
        img, mask = deconstr(sample)
        img = TF.to_tensor(img)
        mask = torch.from_numpy(np.array(mask, np.float32)).float()
        return constr(img, mask)


class SetSeed(object):
    def __call__(self, sample):
        if 'seed' in sample:
            random.seed(sample['seed'])
        return constr(sample['image'], sample['label'])


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img, mask = deconstr(sample)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return constr(img, mask)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img, mask = deconstr(sample)

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return constr(img, mask)


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img, mask = deconstr(sample)

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return constr(img, mask)


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill):
        self.base_size = base_size * (3-len(base_size))
        self.crop_size = crop_size * (3-len(crop_size))
        self.fill = fill

    def __call__(self, sample):
        img, mask = deconstr(sample)

        # random scale (short edge)
        w, h = img.size
        if h > w:
            short_size = random.randint(
                int(self.base_size[0] * 0.75), int(self.base_size[0] * 1.5))
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            short_size = random.randint(
                int(self.base_size[1] * 0.75), int(self.base_size[1] * 1.5))
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        padh = self.crop_size[1] - oh if oh < self.crop_size[1] else 0
        padw = self.crop_size[0] - ow if ow < self.crop_size[0] else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img.crop(
            (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop(
            (x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        return constr(img, mask)


class CustomTranslate(object):
    def __init__(self, translations, fill=(0,0,0)):
        self.translations = translations
        self.fill = fill

    def __call__(self, img1):
        translations = []
        for translate in self.translations:
            im_tr = TF.affine(img1, translate=translate, angle=0, scale=1, shear=0, fillcolor=self.fill)
            translations.append(im_tr)
        return translations