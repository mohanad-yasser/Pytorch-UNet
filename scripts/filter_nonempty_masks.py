import os
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import argparse

def filter_nonempty_masks(img_dir, mask_dir, out_img_dir, out_mask_dir):
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

def main():
    parser = argparse.ArgumentParser(description='Filter non-empty masks and copy corresponding images/masks.')
    parser.add_argument('--img_dir', type=str, required=True, help='Input images directory')
    parser.add_argument('--mask_dir', type=str, required=True, help='Input masks directory')
    parser.add_argument('--out_img_dir', type=str, required=True, help='Output images directory')
    parser.add_argument('--out_mask_dir', type=str, required=True, help='Output masks directory')
    args = parser.parse_args()
    filter_nonempty_masks(Path(args.img_dir), Path(args.mask_dir), Path(args.out_img_dir), Path(args.out_mask_dir))

if __name__ == '__main__':
    main() 