import os
import h5py
import numpy as np
import imageio
from PIL import Image

# === FOLDERS ===
INPUT_FOLDER = 'data/BraTS2020_training_data/content/data'
IMG_OUT_DIR = 'data/imgs_t1_only'
MASK_OUT_DIR = 'data/masks_t1_only'

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

# Extract only T1 modality (index 0) and volume 1 files
volume_1_count = 0
for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith('.h5'):
        continue
    
    # Filter for volume 1 only
    if not filename.startswith('volume_1_'):
        continue

    filepath = os.path.join(INPUT_FOLDER, filename)
    try:
        with h5py.File(filepath, 'r') as f:
            # Get all modalities
            image = f['image'][()]  # Shape: (H, W, 4) for 4 modalities
            mask = f['mask'][()]    # Shape: (H, W, 3) for RGB mask

            # Extract only T1 modality (index 0 in the last dimension)
            t1_image = image[:, :, 0]  # T1 is at index 0 in the last dimension
            t1_image = normalize(t1_image)
            
            # Create filename
            base_name = os.path.splitext(filename)[0]
            img_out_path = os.path.join(IMG_OUT_DIR, f"{base_name}.png")
            mask_out_path = os.path.join(MASK_OUT_DIR, f"{base_name}_mask.png")

            # Save T1 image as grayscale using PIL
            t1_pil = Image.fromarray(t1_image, mode='L')
            t1_pil.save(img_out_path)
            
            # Save mask - convert RGB to binary
            # Take any non-zero value as foreground
            mask_binary = (mask.sum(axis=2) > 0).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_binary, mode='L')
            mask_pil.save(mask_out_path)

        volume_1_count += 1
        print(f"✓ Extracted T1: {filename}")
    except Exception as e:
        print(f"✗ Failed to process {filename}: {e}")
        import traceback
        traceback.print_exc()

print(f"\nExtraction complete!")
print(f"Processed {volume_1_count} volume 1 files")
print(f"T1 images saved to: {IMG_OUT_DIR}")
print(f"Masks saved to: {MASK_OUT_DIR}") 