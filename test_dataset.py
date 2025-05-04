import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import BCEWithLogitsLoss
from utils.dice_score import dice_loss
from utils.data_loading import BasicDataset

# Load raw image and mask manually
img_path = './data/imgs/volume_1_slice_47.png'
mask_path = './data/masks_binary/volume_1_slice_47_mask.png'

# Load raw files as grayscale PIL images
raw_img = Image.open(img_path).convert('L')
raw_mask = Image.open(mask_path).convert('L')

# Initialize dataset to access preprocessing
dataset = BasicDataset('./data/imgs/', './data/masks_binary/', scale=1.0)

# ✅ Preprocess and convert to torch tensors
# Just ONE unsqueeze for batch dimension (1, 1, H, W)
image_tensor = torch.tensor(dataset.preprocess(raw_img, scale=1.0, is_mask=False)).unsqueeze(0)  # (1, 1, H, W)
mask_tensor = torch.tensor(dataset.preprocess(raw_mask, scale=1.0, is_mask=True)).unsqueeze(0)   # (1, 1, H, W)


# ✅ Simulate model output (fuzzy logits around ground truth)
pred_logits = model(image_tensor)  # Ensure model is loaded and in eval() mode


# ✅ Compute losses
bce_fn = BCEWithLogitsLoss()
bce = bce_fn(pred_logits, mask_tensor)
dice = dice_loss(torch.sigmoid(pred_logits), mask_tensor, multiclass=False)
combo = 0.7 * bce + 0.3 * dice

# ✅ Display metrics
print(f"📊 Combo Loss: {combo.item()}")
print(f"✅ BCE Loss: {bce.item()}")
print(f"✅ Dice Loss: {dice.item()}")
print(f"Mask unique values: {mask_tensor.unique()}")
print(f"Mask shape: {mask_tensor.shape}")

# ✅ Convert predictions to binary mask for visualization
with torch.no_grad():
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > 0.5).float()

# ✅ Plot results
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(mask_tensor.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title("Ground Truth Mask")

axes[1].imshow(pred_binary.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title("Predicted Binary Mask")

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
