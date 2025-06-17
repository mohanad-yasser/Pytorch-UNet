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

# Modality mapping
MODALITIES = {
    0: 't1',
    1: 't1ce',
    2: 't2',
    3: 'flair'
}

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith('.h5'):
        continue

    filepath = os.path.join(INPUT_FOLDER, filename)
    try:
        with h5py.File(filepath, 'r') as f:
            # Get all modalities
            image = f['image'][()]  # Shape: (4, H, W) for 4 modalities
            mask = f['mask'][()]

            # Process each modality separately
            for modality_idx, modality_name in MODALITIES.items():
                # Get the specific modality
                modality_image = image[modality_idx]
                modality_image = normalize(modality_image)
                
                # Create filename with modality information
                base_name = os.path.splitext(filename)[0]
                img_out_path = os.path.join(IMG_OUT_DIR, f"{base_name}_{modality_name}.png")
                mask_out_path = os.path.join(MASK_OUT_DIR, f"{base_name}_mask.png")

                # Save image and mask
                imageio.imwrite(img_out_path, modality_image)
                if modality_idx == 0:  # Save mask only once
                    mask = (mask > 0).astype(np.uint8) * 255  # Binary mask
                    imageio.imwrite(mask_out_path, mask)

            print(f"✓ Converted: {filename}")
    except Exception as e:
        print(f"✗ Failed to process {filename}: {e}")
