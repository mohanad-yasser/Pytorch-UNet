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

def apply_clahe(image_np):
    """
    Apply CLAHE to a grayscale image (2D NumPy array).
    """
    image_uint8 = np.uint8(np.clip(image_np, 0, 255))  # Ensure 8-bit input
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
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
    # ✅ Convert to NumPy
    img_np = np.asarray(image).astype(np.float32)
    mask_np = np.asarray(mask).astype(np.uint8)

    # ✅ 50% chance to apply elastic deformation
    if random.random() < 0.5:
        img_np = elastic_deform_2d(img_np, alpha=34, sigma=4)
        mask_np = elastic_deform_2d(mask_np, alpha=34, sigma=4)

    # ✅ Convert back to PIL
    image = Image.fromarray(img_np).convert('L')
    mask = Image.fromarray(mask_np)

    # 🔁 Optional flips & rotations
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    angle = random.choice([0, 90, 180, 270])
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)

    return image, mask

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


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
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        img = apply_clahe(img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            if img.ndim == 2:
                # ✅ Step 1: Apply bias correction BEFORE Z-score
                img = apply_bias_correction(img)

                # ✅ Step 2: Convert to shape (1, H, W)
                img = img[np.newaxis, ...]

                # ✅ Step 3: Apply Brain-Only Z-score normalization
                brain_mask = img > 0
                brain_values = img[brain_mask]

                mean = brain_values.mean()
                std = brain_values.std()
                if std == 0: std = 1.0
                img = (img - mean) / std

            else:
                raise ValueError("Expected grayscale image (2D), got color image.")

            return img


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        img, mask = joint_augment(img, mask)    

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
