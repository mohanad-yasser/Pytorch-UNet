import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import random
import torchvision.transforms.functional as TF


# === Load grayscale MRI image ===
def load_grayscale_image(path):
    return np.array(Image.open(path).convert('L')).astype(np.float32)


# === N4 Bias Field Correction ===
def apply_bias_correction(np_img):
    sitk_image = sitk.GetImageFromArray(np_img)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrected_image = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected_image)


# === CLAHE (Contrast Limited AHE) ===
def apply_clahe(image_np):
    image_uint8 = np.uint8(np.clip(image_np, 0, 255))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(image_uint8).astype(np.float32)


# === Z-Score Normalization ===
def z_score_normalize(img):
    brain_mask = img > 0
    brain_pixels = img[brain_mask]
    mean, std = brain_pixels.mean(), brain_pixels.std()
    std = std if std > 0 else 1.0
    norm = (img - mean) / std
    return np.clip(norm, -3.0, 3.0)


# === Elastic Deformation ===
def elastic_deform(image_np, alpha=34, sigma=4):
    random_state = np.random.RandomState(None)
    shape = image_np.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    return map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)


# === Horizontal and Vertical Flips ===
def apply_flips(img):
    pil = Image.fromarray(img.astype(np.uint8))
    if random.random() > 0.5:
        pil = TF.hflip(pil)
    if random.random() > 0.5:
        pil = TF.vflip(pil)
    return np.array(pil).astype(np.float32)


# === Random Rotation (0, 90, 180, 270) ===
def random_rotate(img):
    pil = Image.fromarray(img.astype(np.uint8))
    angle = random.choice([0, 90, 180, 270])
    return np.array(TF.rotate(pil, angle)).astype(np.float32)


# === Random Zoom (Scaling) ===
def random_zoom(img):
    pil = Image.fromarray(img.astype(np.uint8))
    scale = random.uniform(0.9, 1.1)
    return np.array(TF.affine(pil, angle=0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])).astype(np.float32)


# === Contrast Adjustment ===
def random_contrast(img):
    pil = Image.fromarray(img.astype(np.uint8))
    factor = random.uniform(0.9, 1.1)
    return np.array(TF.adjust_contrast(pil, factor)).astype(np.float32)


# === Show Original + Processed side by side ===
def show_side_by_side(title, original, processed):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(processed, cmap='gray')
    axs[1].set_title(title)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show(block=True)  # Wait until user closes window


# === MAIN ===
if __name__ == "__main__":
    img_path = "C:/Users/mohan/OneDrive/Desktop/Pytorch-UNet/data/imgs/volume_1_slice_58.png"
    original = load_grayscale_image(img_path)

    steps = [
        ("CLAHE", apply_clahe(original)),
        ("Bias Field Corrected", apply_bias_correction(original)),
        ("Z-Score Normalized", z_score_normalize(apply_bias_correction(original))),
        ("Elastic Deformation", elastic_deform(original)),
        ("Random Flips", apply_flips(original)),
        ("Rotation", random_rotate(original)),
        ("Zoom", random_zoom(original)),
        ("Contrast Adjustment", random_contrast(original))
    ]

    for title, processed in steps:
        show_side_by_side(title, original, processed)
