import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import logging
from torch.utils.data import DataLoader, random_split
from unet import ResUNet
from utils.data_loading import BasicDataset
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def predict_and_visualize(checkpoint_path, num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Use combined dataset directories
    dir_img = Path('./data/imgs_val_t1_tumor_only/')
    dir_mask = Path('./data/masks_val_t1_tumor_only/')
    dataset = BasicDataset(dir_img, dir_mask, scale=0.5)
    
    # Use all validation data (no random split needed)
    val_loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=0, pin_memory=True)
    
    # Create model
    model = ResUNet(n_channels=1, n_classes=2, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    mask_values = state_dict.pop('mask_values', None)
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()
    
    logging.info(f'Model loaded from {checkpoint_path}')
    
    # Generate predictions for a few samples
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            if sample_count >= num_samples:
                break
                
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            
            # Get predictions
            masks_pred = model(images)
            
            # Convert predictions to probabilities
            if model.n_classes == 1:
                masks_pred = F.sigmoid(masks_pred)
                masks_pred = (masks_pred > 0.5).float()
            else:
                masks_pred = F.softmax(masks_pred, dim=1)
                masks_pred = masks_pred.argmax(dim=1)
            
            # Convert to numpy for visualization
            img_np = images.cpu().numpy()[0, 0]  # Remove batch and channel dims
            true_mask_np = true_masks.cpu().numpy()[0]
            pred_mask_np = masks_pred.cpu().numpy()[0]
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_np, cmap='gray')
            axes[0].set_title('Original T1 Image')
            axes[0].axis('off')
            
            # True mask
            axes[1].imshow(true_mask_np, cmap='gray')
            axes[1].set_title('True Mask')
            axes[1].axis('off')
            
            # Predicted mask
            axes[2].imshow(pred_mask_np, cmap='gray')
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            sample_count += 1
    
    logging.info(f'Shown {sample_count} prediction visualizations interactively.')
    return sample_count

if __name__ == '__main__':
    checkpoint_path = './checkpoints/checkpoint_epoch15.pth'
    num_shown = predict_and_visualize(checkpoint_path, num_samples=5)
    print(f'✅ Shown {num_shown} prediction images interactively.') 