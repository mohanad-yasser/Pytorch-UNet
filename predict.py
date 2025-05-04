import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from utils.data_loading import BasicDataset
from unet.hybrid_unet_model import HybridUNet
from pathlib import Path

# 🛠 Setup paths
input_folder = Path('data/imgs/')
mask_folder = Path('data/masks_binary/')  
checkpoint_path = Path('checkpoints/checkpoint_epoch9.pth')
save_folder = Path('predictions/')
save_folder.mkdir(parents=True, exist_ok=True)

# 🚀 Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridUNet(n_channels=1, n_classes=1, bilinear=True)
state_dict = torch.load(checkpoint_path, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 🔥 Load filenames ONLY for volume_1
volume1_images = sorted([f for f in input_folder.glob('volume_1_slice_*.png')])

# 🚀 Predict and Save
for img_path in volume1_images:
    name = img_path.stem  # e.g., "volume_1_slice_0"

    # Load input image and true mask
    img = Image.open(img_path).convert('L')
    mask_path = mask_folder / f"{name}_mask.png"
    mask = Image.open(mask_path).convert('L')

    # Preprocess input image correctly
    img_arr = BasicDataset.preprocess(img, scale=1.0, is_mask=False)
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).to(device=device, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        pred = pred.squeeze().cpu().numpy()

    # 🖼 Resize to 240x240
    img_resized = img.resize((240, 240))
    mask_resized = mask.resize((240, 240))
    pred_img = Image.fromarray((pred * 255).astype(np.uint8)).resize((240, 240))

    # 🧩 Merge input, true mask, prediction side-by-side
    combined_width = img_resized.width * 3
    combined_img = Image.new('L', (combined_width, img_resized.height))
    combined_img.paste(img_resized, (0, 0))
    combined_img.paste(mask_resized, (img_resized.width, 0))
    combined_img.paste(pred_img, (img_resized.width * 2, 0))

    # Save combined result
    combined_img.save(save_folder / f"{name}_comparison.png")
    print(f"✅ Saved: {save_folder}/{name}_comparison.png")
