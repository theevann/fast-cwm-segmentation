import os
import random
from logging import getLogger
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF

from pathlib import Path
import dataloaders.custom_transforms as tr


class CityscapesBase(data.Dataset):
    num_classes = 19
    total_classes = 35
    ignore_index = 255
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                    'motorcycle', 'bicycle']
    class_colors = torch.Tensor([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
    ])
    base_size = (2048, 1024)
    mean = torch.Tensor([0.2869, 0.3252, 0.2839])
    std = torch.Tensor([0.1869, 0.1901, 0.1872])
    
    def __init__(self, root, split, crop_size, frame_numbers=[19], augmentation=None):

        self.root = Path(root)
        self.split = split
        self.frame_numbers = frame_numbers
        self.crop_size = crop_size

        self.images_base = self.root / 'leftImg8bit' / self.split
        self.sequence_base = self.root / 'leftImg8bit_sequence' / self.split
        self.annotations_base = self.root / 'gtFine' / self.split
        self.files = list(Path(self.images_base).rglob('*.png'))

        self.class_map = np.ones(self.total_classes) * self.ignore_index
        self.class_map[self.valid_classes] = np.arange(self.num_classes)

        self.augmentation = augmentation if augmentation is not None else (self.split == 'train')
        self.transform = self.create_transforms()
        self.seed = 0

        if not self.files:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        else:
            getLogger("train_code").debug("Found %d %s images", len(self.files), split)
               
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = {}
        seed = (idx + self.seed) if len(self.frame_numbers) > 1 else None

        for i, num in enumerate(self.frame_numbers):
            sample = self.get_image_frame(idx, num, seed)
            item['image_%d' % i] = sample['image']
        item['label'] = sample['label']

        return item

    def get_image_frame(self, image_number, frame_number, seed=None):
        img_path = self.files[image_number]
        lbl_path = self.annotations_base / str(img_path.relative_to(self.images_base))\
            .replace('leftImg8bit', 'gtFine_labelIds')

        city, prefix, index, suffix = str(img_path.name).split('_')
        index = int(index) - 19

        img_path = self.sequence_base / city / '{}_{}_{:0>6}_{}'.format(
            city, prefix, index + frame_number, suffix)

        img = Image.open(img_path).convert('RGB')
        label = np.array(Image.open(lbl_path), dtype=np.uint8)
        target = Image.fromarray(self.encode_segmap(label))

        sample = { 'image': img, 'label': target }
        if seed is not None:
            sample['seed'] = seed
        sample = self.transform(sample)

        img.close()
        return {'image': sample['image'], 'label': sample['label'].long(), 'index': frame_number}

    def encode_segmap(self, mask):
        return self.class_map[mask]

    def decode_segmap(self, mask):
        mask[mask == self.ignore_index] = self.num_classes
        return (self.class_colors[mask] / 255.).permute(2, 0, 1)
            
    def create_transforms(self):
        random_transforms = []
        standard_transforms = [
            tr.ToTensor(),
            tr.Normalize(self.mean, self.std),
        ]

        if self.augmentation:
            random_transforms = [
                tr.SetSeed(),
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=self.base_size,
                                   crop_size=self.crop_size,
                                   fill=self.ignore_index),
                tr.RandomGaussianBlur()
            ]

        return transforms.Compose(random_transforms + standard_transforms)

    def unnormalise_img(self, image):
        return image * self.std.view(3, 1, 1) + self.mean.view(3, 1, 1)

    def show(self, image, mode, scale=0.25):
        return TF.to_pil_image(F.interpolate(image[None, ...], scale_factor=scale, mode=mode).squeeze())

    def show_img(self, image, **kwargs):
        return self.show(image * self.std.view(3,1,1) + self.mean.view(3,1,1), mode='bilinear', **kwargs)

    def show_map(self, segmap, **kwargs):
        return self.show(self.decode_segmap(segmap.long()), mode='nearest', **kwargs)

    def show_both(self, image, segmap, alpha, **kwargs):
        image = image * self.std.view(3, 1, 1) + self.mean.view(3, 1, 1)
        segmap = self.decode_segmap(segmap.long()).squeeze().float()
        return self.show(image * alpha + segmap * (1-alpha), mode='bilinear', **kwargs)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__