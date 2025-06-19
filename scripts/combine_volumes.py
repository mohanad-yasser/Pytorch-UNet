import os
import shutil
from pathlib import Path
import random
import argparse

def combine_volumes_and_split(
    volume1_img_dir, volume1_mask_dir,
    volume200_img_dir, volume200_mask_dir,
    volume300_img_dir, volume300_mask_dir,
    output_img_dir, output_mask_dir,
    train_img_dir, train_mask_dir,
    val_img_dir, val_mask_dir,
    val_split=0.2,
    seed=42
):
    """
    Combine volumes 1, 200, and 300, then split into train/validation sets
    """
    # Set random seed for reproducible splits
    random.seed(seed)
    
    # Create output directories
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_mask_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all files from all volumes
    all_files = []
    
    # Volume 1 files
    if volume1_img_dir.exists():
        for img_file in volume1_img_dir.iterdir():
            if img_file.name.startswith('.'):
                continue
            # Mask file has _mask suffix
            mask_file = volume1_mask_dir / f"{img_file.stem}_mask{img_file.suffix}"
            if mask_file.exists():
                all_files.append(('volume1', img_file, mask_file))
    
    # Volume 200 files
    if volume200_img_dir.exists():
        for img_file in volume200_img_dir.iterdir():
            if img_file.name.startswith('.'):
                continue
            # Mask file has _mask suffix
            mask_file = volume200_mask_dir / f"{img_file.stem}_mask{img_file.suffix}"
            if mask_file.exists():
                all_files.append(('volume200', img_file, mask_file))
    
    # Volume 300 files
    if volume300_img_dir.exists():
        for img_file in volume300_img_dir.iterdir():
            if img_file.name.startswith('.'):
                continue
            # Mask file has _mask suffix
            mask_file = volume300_mask_dir / f"{img_file.stem}_mask{img_file.suffix}"
            if mask_file.exists():
                all_files.append(('volume300', img_file, mask_file))
    
    print(f"Found {len(all_files)} total files:")
    volume_counts = {}
    for volume, _, _ in all_files:
        volume_counts[volume] = volume_counts.get(volume, 0) + 1
    for volume, count in volume_counts.items():
        print(f"  {volume}: {count} files")
    
    # Shuffle files
    random.shuffle(all_files)
    
    # Split into train and validation
    n_val = int(len(all_files) * val_split)
    n_train = len(all_files) - n_val
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:]
    
    print(f"\nSplit: {n_train} training, {n_val} validation")
    
    # Copy files to combined directory first
    print("\nCopying files to combined directory...")
    for i, (volume, img_file, mask_file) in enumerate(all_files):
        # Create unique names to avoid conflicts
        new_name = f"{volume}_{img_file.name}"
        shutil.copy2(img_file, output_img_dir / new_name)
        shutil.copy2(mask_file, output_mask_dir / new_name)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(all_files)} files")
    
    # Copy to train/val directories
    print("\nCopying to training directory...")
    for i, (volume, img_file, mask_file) in enumerate(train_files):
        new_name = f"{volume}_{img_file.name}"
        shutil.copy2(img_file, train_img_dir / new_name)
        shutil.copy2(mask_file, train_mask_dir / new_name)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(train_files)} training files")
    
    print("\nCopying to validation directory...")
    for i, (volume, img_file, mask_file) in enumerate(val_files):
        new_name = f"{volume}_{img_file.name}"
        shutil.copy2(img_file, val_img_dir / new_name)
        shutil.copy2(mask_file, val_mask_dir / new_name)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(val_files)} validation files")
    
    print(f"\n✅ Successfully combined and split {len(all_files)} files:")
    print(f"  Combined: {len(list(output_img_dir.iterdir()))} files")
    print(f"  Training: {len(list(train_img_dir.iterdir()))} files")
    print(f"  Validation: {len(list(val_img_dir.iterdir()))} files")

def main():
    parser = argparse.ArgumentParser(description='Combine volumes and split for training/validation')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')
    args = parser.parse_args()
    
    # Define directories
    volume1_img_dir = Path('./data/imgs_t1_tumor_only/')
    volume1_mask_dir = Path('./data/masks_t1_tumor_only/')
    volume200_img_dir = Path('./data/imgs_volume_200_t1_tumor_only/')
    volume200_mask_dir = Path('./data/masks_volume_200_t1_tumor_only/')
    volume300_img_dir = Path('./data/imgs_volume_300_t1_tumor_only/')
    volume300_mask_dir = Path('./data/masks_volume_300_t1_tumor_only/')
    
    # Output directories
    output_img_dir = Path('./data/imgs_combined_t1_tumor_only/')
    output_mask_dir = Path('./data/masks_combined_t1_tumor_only/')
    train_img_dir = Path('./data/imgs_train_t1_tumor_only/')
    train_mask_dir = Path('./data/masks_train_t1_tumor_only/')
    val_img_dir = Path('./data/imgs_val_t1_tumor_only/')
    val_mask_dir = Path('./data/masks_val_t1_tumor_only/')
    
    combine_volumes_and_split(
        volume1_img_dir, volume1_mask_dir,
        volume200_img_dir, volume200_mask_dir,
        volume300_img_dir, volume300_mask_dir,
        output_img_dir, output_mask_dir,
        train_img_dir, train_mask_dir,
        val_img_dir, val_mask_dir,
        val_split=args.val_split,
        seed=args.seed
    )

if __name__ == '__main__':
    main() 