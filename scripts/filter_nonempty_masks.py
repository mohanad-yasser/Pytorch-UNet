import os
from PIL import Image
import numpy as np
from pathlib import Path
import shutil

# Directories
img_dir = Path('data/imgs_t1_only')
mask_dir = Path('data/masks_t1_only')
out_img_dir = Path('data/imgs_t1_tumor_only')
out_mask_dir = Path('data/masks_t1_tumor_only')

out_img_dir.mkdir(exist_ok=True)
out_mask_dir.mkdir(exist_ok=True)

num_copied = 0
for mask_file in mask_dir.iterdir():
    if mask_file.name.startswith('.'):
        continue
    mask = np.array(Image.open(mask_file))
    if np.any(mask > 0):  # Non-empty mask
        # Adjust if your naming differs
        img_file = img_dir / mask_file.name.replace('_mask', '')
        if img_file.exists():
            shutil.copy(str(img_file), str(out_img_dir / img_file.name))
            shutil.copy(str(mask_file), str(out_mask_dir / mask_file.name))
            num_copied += 1

print(f"Filtered dataset created with {num_copied} non-empty masks.") 