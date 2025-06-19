import h5py
import numpy as np
import matplotlib.pyplot as plt

def inspect_h5_structure(h5_file_path):
    """
    Inspect the structure of an H5 file
    """
    with h5py.File(h5_file_path, 'r') as f:
        print(f"File: {h5_file_path}")
        print("Keys:", list(f.keys()))
        
        for key in f.keys():
            data = f[key]
            print(f"\nKey: {key}")
            print(f"Shape: {data.shape}")
            print(f"Type: {data.dtype}")
            print(f"Min: {data[:].min()}")
            print(f"Max: {data[:].max()}")
            
            # Show a small sample
            if len(data.shape) == 3:
                print(f"Sample slice 0: shape {data[0].shape}")
            elif len(data.shape) == 2:
                print(f"Sample: shape {data.shape}")

if __name__ == '__main__':
    # Inspect a volume 200 file
    h5_file = "data/BraTS2020_training_data/content/data/volume_200_slice_0.h5"
    inspect_h5_structure(h5_file)

# Path to your H5 file
h5_path = 'data/BraTS2020_training_data/content/data/volume_1_slice_53.h5'

with h5py.File(h5_path, 'r') as f:
    print("Keys in the file:", list(f.keys()))
    image = f['image'][()]  # Should be shape (4, H, W)
    mask = f['mask'][()]
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    # Show all modalities for this slice
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    for i, name in enumerate(modality_names):
        plt.figure()
        plt.title(f"{name} modality")
        plt.imshow(image[i], cmap='gray')
        plt.axis('off')
        plt.show()

    # Show the mask
    plt.figure()
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show() 