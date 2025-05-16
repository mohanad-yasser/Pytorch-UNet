import torch
import matplotlib.pyplot as plt
from unet.hybrid_unet_model import HybridUNet
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np

# --- Config (should match train.py) ---
config = {
    'n_channels': 1,
    'n_classes': 1,
    'bilinear': True,
    'img_scale': 1.0,
    'batch_size': 4,  # for visualization
}

# --- Paths ---
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks_binary/')
checkpoint_path = 'checkpoints/checkpoint_epoch53.pth'

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model ---
model = HybridUNet(
    n_channels=config['n_channels'],
    n_classes=config['n_classes'],
    bilinear=config['bilinear']
)
model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Dataset & DataLoader (validation set) ---
dataset = BasicDataset(dir_img, dir_mask, scale=config['img_scale'])
# For demo, just use the first N images
val_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

# --- Get a batch ---
batch = next(iter(val_loader))
images = batch['image'].to(device)
true_masks = batch['mask'].to(device)

# --- Predict ---
with torch.no_grad():
    pred = model(images)
    pred_mask = (torch.sigmoid(pred) > 0.5).float()

# --- Visualize ---
for i in range(min(10, images.shape[0])):
    plt.figure(figsize=(12,4))
    axs = plt.subplot(1,3,1), plt.subplot(1,3,2), plt.subplot(1,3,3)
    for ax in axs:
        ax.axis('off')
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Ground Truth')
    axs[2].title.set_text('Prediction')

    input_img = images[i, 0].cpu().numpy()
    gt_mask = true_masks[i, 0].cpu().numpy()
    pred_mask_np = pred_mask[i, 0].cpu().numpy()

    # Lower threshold
    pred_bin = (pred_mask_np > 0.5).astype(np.uint8)

    # Post-processing: morphological opening and closing, remove small objects
    kernel = np.ones((3, 3), np.uint8)
    pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
    pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
    min_area = 50  # Remove small objects
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            pred_bin[labels == label] = 0

    axs[0].imshow(input_img, cmap='gray')
    axs[1].imshow(gt_mask, cmap='gray')
    axs[2].imshow(pred_bin, cmap='gray')
    plt.show()