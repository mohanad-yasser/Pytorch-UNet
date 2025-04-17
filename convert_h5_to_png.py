import os
import h5py
import numpy as np
import imageio

# === FOLDERS ===
INPUT_FOLDER = 'data/BraTS2020_training_data/content/data'
IMG_OUT_DIR = 'data/imgs'
MASK_OUT_DIR = 'data/masks'

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith('.h5'):
        continue

    filepath = os.path.join(INPUT_FOLDER, filename)
    try:
        with h5py.File(filepath, 'r') as f:
            image = f['image'][()]
            mask = f['mask'][()]

            image = normalize(image)
            mask = (mask > 0).astype(np.uint8) * 255  # Binary mask

            # Remove .h5 extension
            base_name = os.path.splitext(filename)[0]

            img_out_path = os.path.join(IMG_OUT_DIR, f"{base_name}.png")
            mask_out_path = os.path.join(MASK_OUT_DIR, f"{base_name}_mask.png")

            imageio.imwrite(img_out_path, image)
            imageio.imwrite(mask_out_path, mask)

            print(f"✓ Converted: {filename}")
    except Exception as e:
        print(f"✗ Failed to process {filename}: {e}")
