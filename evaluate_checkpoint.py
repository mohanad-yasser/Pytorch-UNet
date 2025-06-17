import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from unet import UNet
from utils.volume1_dataset import Volume1Dataset
from evaluate import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_checkpoint(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Create dataset
    dir_img = Path('./data/imgs_t1_only/')
    dir_mask = Path('./data/masks_t1_only/')
    dataset = Volume1Dataset(dir_img, dir_mask, scale=0.5)
    
    # Split into train / validation partitions
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # Create validation data loader
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, num_workers=4, pin_memory=True)
    
    # Create model
    model = UNet(n_channels=1, n_classes=2, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    mask_values = state_dict.pop('mask_values', None)
    model.load_state_dict(state_dict)
    model.to(device=device)
    
    logging.info(f'Model loaded from {checkpoint_path}')
    logging.info(f'Model has {model.n_channels} input channels and {model.n_classes} output classes')
    
    # Evaluate
    val_score = evaluate(model, val_loader, device, amp=False)
    logging.info(f'Validation Dice score: {val_score}')
    
    return val_score

if __name__ == '__main__':
    checkpoint_path = './checkpoints/checkpoint_epoch4.pth'
    score = evaluate_checkpoint(checkpoint_path)
    print(f'Final Dice Score: {score:.4f}') 