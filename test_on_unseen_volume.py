import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import logging
from torch.utils.data import DataLoader
from unet import UNet
from utils.volume_dataset import VolumeDataset
import torch.nn.functional as F
from evaluate import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_on_unseen_volume(checkpoint_path, volume_name, num_samples=5):
    """
    Test the trained model on an unseen volume
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Use the unseen volume dataset
    dir_img = Path(f'./data/imgs_{volume_name}_t1/')
    dir_mask = Path(f'./data/masks_{volume_name}_t1/')
    
    if not dir_img.exists() or not dir_mask.exists():
        logging.error(f"Dataset directories not found: {dir_img} or {dir_mask}")
        logging.info("Please run: python extract_t1_from_h5.py --h5_dir <path> --volume <number>")
        return
    
    # Create dataset for the unseen volume
    dataset = VolumeDataset(dir_img, dir_mask, volume_name, scale=0.5)
    
    # Create data loader
    test_loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=1, num_workers=0, pin_memory=True)
    
    # Create model
    model = UNet(n_channels=1, n_classes=2, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    mask_values = state_dict.pop('mask_values', None)
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()
    
    logging.info(f'Model loaded from {checkpoint_path}')
    logging.info(f'Testing on {volume_name} with {len(dataset)} samples')
    
    # Evaluate the model
    dice_score = evaluate(model, test_loader, device, amp=False)
    logging.info(f'Dice Score on {volume_name}: {dice_score:.4f}')
    
    # Generate visualizations for a few samples
    sample_count = 0
    with torch.no_grad():
        for batch in test_loader:
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
            axes[0].set_title(f'Original T1 Image ({volume_name})')
            axes[0].axis('off')
            
            # True mask
            axes[1].imshow(true_mask_np, cmap='gray')
            axes[1].set_title('True Mask')
            axes[1].axis('off')
            
            # Predicted mask
            axes[2].imshow(pred_mask_np, cmap='gray')
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            plt.suptitle(f'Model Prediction on {volume_name} - Sample {sample_count+1}')
            plt.tight_layout()
            plt.show()
            sample_count += 1
    
    logging.info(f'Shown {sample_count} prediction visualizations for {volume_name}')
    return dice_score

if __name__ == '__main__':
    # Test on volume 2 (unseen data)
    checkpoint_path = './checkpoints/checkpoint_epoch10.pth'  # Use the best checkpoint
    volume_name = 'volume_2'
    
    dice_score = test_on_unseen_volume(checkpoint_path, volume_name, num_samples=5)
    print(f'✅ Tested model on {volume_name}')
    print(f'📊 Dice Score: {dice_score:.4f}') 