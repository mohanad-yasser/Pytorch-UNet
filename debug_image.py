import logging
from pathlib import Path
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def debug_image_loading():
    # Test with a single image
    img_path = Path('./data/imgs_t1_only/volume_1_slice_0.png')
    mask_path = Path('./data/masks_t1_only/volume_1_slice_0_mask.png')
    
    print(f"Testing image: {img_path}")
    print(f"Testing mask: {mask_path}")
    
    if img_path.exists():
        # Load image
        img = Image.open(img_path)
        print(f"✓ Image loaded successfully")
        print(f"  - Mode: {img.mode}")
        print(f"  - Size: {img.size}")
        print(f"  - Format: {img.format}")
        
        # Convert to numpy
        img_array = np.asarray(img)
        print(f"  - Array shape: {img_array.shape}")
        print(f"  - Array dtype: {img_array.dtype}")
        print(f"  - Min/Max values: {img_array.min()}/{img_array.max()}")
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img_gray = img.convert('L')
            img_gray_array = np.asarray(img_gray)
            print(f"  - After grayscale conversion:")
            print(f"    - Array shape: {img_gray_array.shape}")
            print(f"    - Array dtype: {img_gray_array.dtype}")
            print(f"    - Min/Max values: {img_gray_array.min()}/{img_gray_array.max()}")
    else:
        print(f"✗ Image file not found: {img_path}")
    
    if mask_path.exists():
        # Load mask
        mask = Image.open(mask_path)
        print(f"\n✓ Mask loaded successfully")
        print(f"  - Mode: {mask.mode}")
        print(f"  - Size: {mask.size}")
        print(f"  - Format: {mask.format}")
        
        # Convert to numpy
        mask_array = np.asarray(mask)
        print(f"  - Array shape: {mask_array.shape}")
        print(f"  - Array dtype: {mask_array.dtype}")
        print(f"  - Unique values: {np.unique(mask_array)}")
    else:
        print(f"✗ Mask file not found: {mask_path}")

if __name__ == '__main__':
    debug_image_loading() 