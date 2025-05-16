import numpy as np
import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.balanced_sampler import BalancedMaskSampler
from evaluate import evaluate
from unet.hybrid_unet_model import HybridUNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from PIL import Image
from losses import FocalTverskyLoss, CombinedLoss
from utils.augmentations import JointTransform
import math



dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks_binary/')
dir_checkpoint = Path('./checkpoints/')

# 🚀 Hyperparameter Configuration
config = {
    'epochs': 80,
    'batch_size': 1,
    'learning_rate': 2e-5,  # Lowered learning rate for better stability
    'val_percent': 0.2,
    'img_scale': 1.0,
    'amp': True,
    'bilinear': True,
    'n_channels': 1,
    'n_classes': 1,
    'weight_decay': 1e-4,  # Reduced weight decay
    'gradient_clipping': 0.5,  # Reduced gradient clipping
    'accumulation_steps': 8,
    'pos_weight': 5.0,  # Reduced positive class weight
    'ft_weight': 0.4,  # Adjusted weights for better balance
    'dice_weight': 0.6,  # Increased dice weight
    'warmup_epochs': 5
}

train_transform = JointTransform()

def train_model(
        model,
        device,
        epochs: int = 80,
        batch_size: int = 1,
        learning_rate: float = 2e-5,  # Lowered default learning rate
        val_percent: float = 0.2,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = True,
        weight_decay: float = 1e-4,  # Updated default
        gradient_clipping: float = 0.5,  # Updated default
        accumulation_steps: int = 8,
        start_epoch: int = 1,
        checkpoint_path: str = None
):
    # 0) set up loss function with adjusted weights
    criterion = CombinedLoss(pos_weight=5.0, ft_weight=0.4, dice_weight=0.6)

    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✅ Loading checkpoint from: {checkpoint_path}")
        
        # Handle old checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"📈 Resuming from epoch {start_epoch}")

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, transform=train_transform)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, transform=train_transform)

    # 2. Balanced Train/Val Split (based on mask content)
    empty_idxs, non_empty_idxs = [], []
    print("📦 Classifying masks into empty and non-empty for balanced splitting...")
    for i in tqdm(range(len(dataset)), desc="🔍 Classifying masks"):
        mask = dataset[i]['mask'].numpy()
        if np.max(mask) == 0:
            empty_idxs.append(i)
        else:
            non_empty_idxs.append(i)

    # Ensure we have a good balance of empty and non-empty masks
    if len(empty_idxs) > len(non_empty_idxs) * 2:
        # If we have too many empty masks, sample them
        empty_idxs = np.random.choice(empty_idxs, size=len(non_empty_idxs) * 2, replace=False).tolist()

    val_percent_float = val_percent
    if len(empty_idxs) > 0:
        train_empty, val_empty = train_test_split(empty_idxs, test_size=val_percent_float, random_state=42)
    else:
        train_empty, val_empty = [], []

    if len(non_empty_idxs) > 0:
        train_non_empty, val_non_empty = train_test_split(non_empty_idxs, test_size=val_percent_float, random_state=42)
    else:
        train_non_empty, val_non_empty = [], []

    train_indices = [int(i) for i in train_empty + train_non_empty]
    val_indices = [int(i) for i in val_empty + val_non_empty]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"\n✅ Final dataset split:")
    print(f"🔹 Train set — {len(train_set)} samples (empty: {len(train_empty)}, non-empty: {len(train_non_empty)})")
    print(f"🔸 Val set   — {len(val_set)} samples (empty: {len(val_empty)}, non-empty: {len(val_non_empty)})\n")

    # 3. Create data loaders with balanced sampling
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    sampler = BalancedMaskSampler(train_set, empty_fraction=0.4)  # Keep 40% empty masks
    train_loader = DataLoader(train_set, sampler=sampler, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size, **loader_args)

    n_train = len(train_set)
    n_val = len(val_set)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler, and loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Cosine Annealing LR with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * config['warmup_epochs']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # More conservative cosine decay
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress)) * 0.8  # Added 0.8 factor for lower max LR
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Add early stopping
    best_val_dice = 0
    patience = 10
    patience_counter = 0
    best_epoch = 0

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)  # ✅ Initialize the gradient
    effective_batch = batch_size * accumulation_steps
    logging.info(
        f"Simulating effective batch size: {effective_batch} "
        f"(actual batch size {batch_size}, accumulation steps {accumulation_steps})"
    )

    # 5. Begin training
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        batch_count = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                
                # Only log detailed stats every 10 batches
                if batch_count % 10 == 0:
                    logging.info(f"🧠 True mask stats: min={true_masks.min()} max={true_masks.max()} mean={true_masks.float().mean():.4f}")

                assert images.shape[1] == model.n_channels, \
                    f'Expected {model.n_channels} input channels, got {images.shape[1]}'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # forward + loss
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    non_empty = true_masks.view(true_masks.size(0), -1).sum(dim=1) > 0
                    if non_empty.sum() == 0:
                        continue  # skip fully-empty batch

                    # Add loss value checks
                    loss = criterion(masks_pred[non_empty], true_masks[non_empty])
                    
                    # Check for invalid loss values
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"Invalid loss value detected: {loss.item()}. Skipping batch.")
                        continue
                    
                    # Clip loss value to prevent extreme values
                    loss = torch.clamp(loss, min=-100.0, max=100.0)

                    # log a quick Dice on train
                    with torch.no_grad():
                        bin_pred = (torch.sigmoid(masks_pred) > 0.5).float()
                        inter = (bin_pred * true_masks).sum()
                        uni   = bin_pred.sum() + true_masks.sum()
                        dice  = (2 * inter + 1e-6) / (uni + 1e-6)
                        epoch_dice += dice.item()

                pred_probs = torch.sigmoid(masks_pred)
                if global_step % 50 == 0:
                    ratio = (pred_probs > 0.5).float().mean().item()
                    logging.info(f"Predicted mask foreground %: {ratio*100:.2f}%")

                # — accumulate gradients —
                loss = loss / accumulation_steps
                grad_scaler.scale(loss).backward()

                # — optimizer step every `accumulation_steps` mini-batches —
                if (global_step + 1) % accumulation_steps == 0:
                    # Check for invalid gradients before unscaling
                    valid_gradients = True
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                valid_gradients = False
                                break
                    
                    if valid_gradients:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        scheduler.step()
                    else:
                        logging.warning("Invalid gradients detected. Skipping optimizer step.")
                        # Do not call grad_scaler.update() or modify its internals; just skip
                    
                    optimizer.zero_grad(set_to_none=True)

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                batch_count += 1
                
                # Update progress bar with more meaningful metrics
                pbar.set_postfix(**{
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

        # — end of epoch: run one validation pass —
        val_score = evaluate(model, val_loader, device, amp)
        avg_dice = epoch_dice / batch_count
        
        # Early stopping check
        if val_score > best_val_dice:
            best_val_dice = val_score
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            if save_checkpoint:
                best_model_path = str(dir_checkpoint / 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_loss/batch_count,
                    'dice': avg_dice,
                    'val_dice': val_score
                }, best_model_path)
                logging.info(f'Best model saved with validation Dice: {val_score:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered. Best validation Dice: {best_val_dice:.4f} at epoch {best_epoch}')
                break
        
        logging.info(
            f"Epoch {epoch} summary — "
            f"Loss: {epoch_loss/batch_count:.4f} | "
            f"Train Dice: {avg_dice:.4f} | "
            f"Val Dice: {val_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ✅ Save checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss/batch_count,
                'dice': avg_dice,
                'val_dice': val_score
            }
            
            # Save to temporary file first
            temp_path = str(dir_checkpoint / f'checkpoint_epoch{epoch}_temp.pth')
            final_path = str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth')
            
            try:
                # Save to temporary file
                torch.save(state_dict, temp_path)
                
                # If successful, rename to final file
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_path, final_path)
                logging.info(f'Checkpoint {epoch} saved!')
                
            except Exception as e:
                logging.error(f'Error saving checkpoint: {e}')
                # Try to clean up temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                # Try to save to backup location
                try:
                    backup_dir = Path(dir_checkpoint) / 'backup'
                    backup_dir.mkdir(exist_ok=True)
                    backup_path = str(backup_dir / f'checkpoint_epoch{epoch}_backup.pth')
                    torch.save(state_dict, backup_path)
                    logging.info(f'Checkpoint {epoch} saved to backup location!')
                except Exception as backup_e:
                    logging.error(f'Failed to save backup checkpoint: {backup_e}')

    model.train()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=2e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--resume', '-r', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--start-epoch', type=int, default=1, help='Start epoch number')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = HybridUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            start_epoch=args.start_epoch,
            checkpoint_path=args.resume
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            start_epoch=args.start_epoch,
            checkpoint_path=args.resume
        )

