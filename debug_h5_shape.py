import os
import h5py
import numpy as np

# === FOLDERS ===
INPUT_FOLDER = 'data/BraTS2020_training_data/content/data'

# Check the first few volume 1 files
for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith('.h5'):
        continue
    
    # Filter for volume 1 only
    if not filename.startswith('volume_1_'):
        continue

    filepath = os.path.join(INPUT_FOLDER, filename)
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\nFile: {filename}")
            print(f"Keys: {list(f.keys())}")
            
            image = f['image'][()]
            mask = f['mask'][()]
            
            print(f"Image shape: {image.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Mask dtype: {mask.dtype}")
            
            # Check T1 modality specifically
            t1_image = image[0]
            print(f"T1 image shape: {t1_image.shape}")
            print(f"T1 image dtype: {t1_image.dtype}")
            print(f"T1 min/max: {t1_image.min()}/{t1_image.max()}")
            
            # Only check first 3 files
            if filename in ['volume_1_slice_0.h5', 'volume_1_slice_1.h5', 'volume_1_slice_2.h5']:
                break
                
    except Exception as e:
        print(f"✗ Failed to process {filename}: {e}")
        import traceback
        traceback.print_exc()
        break 