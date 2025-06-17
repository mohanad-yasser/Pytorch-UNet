import logging
from pathlib import Path
from utils.volume1_dataset import Volume1Dataset
from multiprocessing import freeze_support

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Test the dataset
    dir_img = Path('./data/imgs_t1_only/')
    dir_mask = Path('./data/masks_t1_only/')

    print("Testing Volume1Dataset...")
    print(f"Images directory: {dir_img}")
    print(f"Masks directory: {dir_mask}")

    try:
        # Create dataset
        dataset = Volume1Dataset(dir_img, dir_mask, scale=0.5)
        
        print(f"✓ Dataset created successfully!")
        print(f"✓ Dataset size: {len(dataset)}")
        
        # Test loading first item
        first_item = dataset[0]
        print(f"✓ First item loaded successfully!")
        print(f"✓ Image shape: {first_item['image'].shape}")
        print(f"✓ Mask shape: {first_item['mask'].shape}")
        print(f"✓ Image dtype: {first_item['image'].dtype}")
        print(f"✓ Mask dtype: {first_item['mask'].dtype}")
        
        # Test a few more items
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            print(f"✓ Item {i}: Image {item['image'].shape}, Mask {item['mask'].shape}")
        
        print("\n✓ All tests passed! The extracted files will be used correctly.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    freeze_support()
    main() 