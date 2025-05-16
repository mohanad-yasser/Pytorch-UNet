import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import random

class JointTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.8),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.7,
                contrast=0.7,
                saturation=0.6,
                hue=0.3
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.35, 0.35),
                scale=(0.6, 1.4),
                shear=35
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5)
            ], p=0.7),
            transforms.Lambda(self.add_random_noise),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
            transforms.Lambda(self.cutout),
        ])

    def add_random_noise(self, img):
        if random.random() < 0.7:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 15, arr.shape)
            arr = np.clip(arr + noise, 0, 255)
            return Image.fromarray(arr.astype(np.uint8))
        return img

    def cutout(self, img, n_holes=1, length=32):
        arr = np.array(img)
        h, w = arr.shape[:2]
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            arr[y1:y2, x1:x2] = 0
        return Image.fromarray(arr.astype(np.uint8))

    def __call__(self, img, mask):
        if random.random() < 0.7:
            img, mask = self.elastic_deform(img, mask)
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask

    def elastic_deform(self, img, mask, alpha=40, sigma=6):
        img_np = np.array(img)
        mask_np = np.array(mask)
        shape = img_np.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        img_deformed = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape)
        mask_deformed = map_coordinates(mask_np, indices, order=0, mode='reflect').reshape(shape)
        return Image.fromarray(img_deformed.astype(np.uint8)), Image.fromarray(mask_deformed.astype(np.uint8))

from scipy.ndimage import gaussian_filter, map_coordinates 