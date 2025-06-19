import h5py
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse

def extract_t1_from_slice_files(h5_dir, output_dir, volume_name):
    """
    Extract T1 modality from individual slice H5 files and save as PNG images
    """
    h5_dir = Path(h5_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    img_dir = output_dir / f'imgs_{volume_name}_t1'
    mask_dir = output_dir / f'masks_{volume_name}_t1'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all H5 files for the specified volume
    h5_files = list(h5_dir.glob(f'{volume_name}_slice_*.h5'))
    
    if not h5_files:
        print(f"No H5 files found for {volume_name} in {h5_dir}")
        return 0
    
    print(f"Found {len(h5_files)} slice files for {volume_name}")
    
    num_processed = 0
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                # Get the T1 modality (first channel)
                t1_data = f['image'][:, :, 0]  # T1 is the first channel
                mask_data = f['mask'][:, :, :]  # RGB mask
                
                # Normalize T1 to 0-255
                t1_data = ((t1_data - t1_data.min()) / (t1_data.max() - t1_data.min()) * 255).astype(np.uint8)
                
                # Convert RGB mask to binary (any non-zero pixel is foreground)
                mask_binary = (mask_data.sum(axis=2) > 0).astype(np.uint8) * 255
                
                # Extract slice number from filename
                slice_num = h5_file.stem.split('_')[-1]  # Get the slice number
                
                # Save T1 image
                img_filename = f'{volume_name}_slice_{slice_num}.png'
                img_path = img_dir / img_filename
                Image.fromarray(t1_data, mode='L').save(img_path)
                
                # Save mask
                mask_filename = f'{volume_name}_slice_{slice_num}_mask.png'
                mask_path = mask_dir / mask_filename
                Image.fromarray(mask_binary, mode='L').save(mask_path)
                
                num_processed += 1
                
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
    
    print(f"Successfully processed {num_processed} slices for {volume_name}")
    return num_processed

def main():
    parser = argparse.ArgumentParser(description='Extract T1 modality from H5 slice files')
    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing H5 files')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for extracted images')
    parser.add_argument('--volume', type=str, required=True, help='Volume number (e.g., 200, 300)')
    
    args = parser.parse_args()
    
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    volume_name = f'volume_{args.volume}'
    
    if not h5_dir.exists():
        print(f"H5 directory not found: {h5_dir}")
        return
    
    try:
        num_slices = extract_t1_from_slice_files(h5_dir, output_dir, volume_name)
        print(f"Successfully extracted {num_slices} T1 slices from {volume_name}")
    except Exception as e:
        print(f"Error processing {volume_name}: {e}")

if __name__ == '__main__':
    main() 