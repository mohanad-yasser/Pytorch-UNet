import h5py
import numpy as np
import matplotlib.pyplot as plt

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