import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import SimpleITK as sitk
from torchvision.transforms import functional as TF
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
from utils.augmentations import JointTransform

import cv2


def apply_clahe(image_np):
    """
    Apply CLAHE to a grayscale image (2D NumPy array).
    If input is 3D (1, H, W) or (H, W, 1), it squeezes it automatically.
    """
    # ✅ Squeeze extra dimensions if any
    if image_np.ndim == 3:
        image_np = np.squeeze(image_np)

    # ✅ Confirm image is now 2D
    if image_np.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image for CLAHE, got shape {image_np.shape}")

    # ✅ Make sure it is 8-bit
    image_uint8 = np.uint8(np.clip(image_np, 0, 255))

    # ✅ Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # ✅ Apply CLAHE
    enhanced = clahe.apply(image_uint8)

    return enhanced.astype(np.float32)



def apply_bias_correction(np_img):
    sitk_image = sitk.GetImageFromArray(np_img.astype(np.float32))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrected_image = corrector.Execute(sitk_image, mask_image)
    corrected_np = sitk.GetArrayFromImage(corrected_image)
    return corrected_np

def elastic_deform_2d(image_np, alpha=34, sigma=4):
    random_state = np.random.RandomState(None)

    shape = image_np.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

    return map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)


def joint_augment(image, mask):
    img_np = np.asarray(image).astype(np.float32)
    mask_np = np.asarray(mask).astype(np.uint8)

    # ✅ Elastic deformation (image + mask)
    if random.random() < 0.5:
        img_np = elastic_deform_2d(img_np, alpha=34, sigma=4)
        mask_np = elastic_deform_2d(mask_np, alpha=34, sigma=4)
        image = Image.fromarray(np.uint8(np.clip(img_np, 0, 255)))
        mask = Image.fromarray(np.uint8(np.clip(mask_np, 0, 255)))

    # ✅ Horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # ✅ Vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # ✅ Rotation
    angle = random.choice([0, 90, 180, 270])
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)

    # ✅ Contrast adjustment (only on image)
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.9, 1.1)
        image = TF.adjust_contrast(image, contrast_factor)

    # ✅ Random Zoom (affine scaling) — 0.9x to 1.1x
    if random.random() < 0.5:
        scale_factor = random.uniform(0.9, 1.1)
        image = TF.affine(image, angle=0, translate=[0, 0], scale=scale_factor, shear=[0.0, 0.0])
        mask = TF.affine(mask, angle=0, translate=[0, 0], scale=scale_factor, shear=[0.0, 0.0])

    return image, mask


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename)).convert('L')
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy()).convert('L')
    else:
        return Image.open(filename).convert('L')



def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, transform=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        # Compose both JointTransform and joint_augment
        self.transform = transform if transform is not None else lambda img, mask: joint_augment(*JointTransform()(img, mask))

        # Only include files that start with 'volume_1_'
        self.ids = []
        for ext in ['.png', '.jpg', '.tif']:
            self.ids.extend([file.stem for file in sorted(self.images_dir.glob(f'volume_1_*{ext}'))])

        # Separate into empty and non-empty masks
        empty_ids = []
        nonempty_ids = []
        for idx in self.ids:
            mask_files = list(self.mask_dir.glob(f'{idx}_mask.*'))
            if len(mask_files) == 1:
                mask = Image.open(mask_files[0]).convert('L')
                if mask.getextrema()[1] == 0:
                    empty_ids.append(idx)
                else:
                    nonempty_ids.append(idx)
        # 1:1 ratio
        n_nonempty = len(nonempty_ids)
        sampled_empty = random.sample(empty_ids, min(n_nonempty, len(empty_ids)))
        self.ids = nonempty_ids + sampled_empty
        random.shuffle(self.ids)
        print(f"Loaded {len(self.ids)} images: {len(nonempty_ids)} non-empty, {len(sampled_empty)} empty (1:1 ratio)")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(f'{name}_mask.*'))[0]
        img_file = list(self.images_dir.glob(f'{name}.*'))[0]

        mask = Image.open(mask_file).convert('L')
        img = Image.open(img_file).convert('L')

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        # Always apply transform
        if img.shape[0] == 1:
            img = img.squeeze(0)
        img = Image.fromarray((img * 255).astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        img, mask = self.transform(img, mask)
        img = np.array(img) / 255.0
        mask = np.array(mask)
        if img.ndim == 3 and img.shape[2] == 3:
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).convert('L')) / 255.0
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]
        mask = (mask > 0).astype(np.float32)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1.0, transform=None):
        super().__init__(images_dir, mask_dir, scale, transform)
        self.mask_suffix = '_mask'
