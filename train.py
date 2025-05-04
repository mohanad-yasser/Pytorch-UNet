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


dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks_binary/')
dir_checkpoint = Path('./checkpoints/')

# 🚀 Hyperparameter Configuration
config = {
    'epochs': 25,
    'batch_size': 1,
    'learning_rate': 1e-4,
    'val_percent': 0.2,         # 10% validation split
    'img_scale': 1.0,           # full 240x240 images
    'amp': True,               # no mixed precision yet
    'bilinear': True,           # use bilinear upsampling
    'n_channels': 1,            # grayscale input (T1 MRI slices)
    'n_classes': 1,             # binary output (tumor or no tumor)
    'weight_decay': 1e-5,
    'gradient_clipping': 1.0,
    'accumulation_steps': 1
}
class ComboLoss(nn.Module):
    def __init__(self, weight=0.7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight = weight

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        dice = dice_loss(torch.sigmoid(preds), targets, multiclass=False)
        return self.weight * bce + (1 - self.weight) * dice

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = True,
        weight_decay: float = 1e-5,
        gradient_clipping: float = 1.0,
        accumulation_steps: int = 1
):
     # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Balanced Train/Val Split (based on mask content)
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    empty_idxs, non_empty_idxs = [], []
    print("📦 Classifying masks into empty and non-empty for balanced splitting...")
    for i in tqdm(range(len(dataset)), desc="🔍 Classifying masks"):
        mask = dataset[i]['mask'].numpy()
        if np.max(mask) == 0:
            empty_idxs.append(i)
        else:
            non_empty_idxs.append(i)

    val_percent_float = val_percent  # already float like 0.2
    train_empty, val_empty = train_test_split(empty_idxs, test_size=val_percent_float, random_state=42)
    train_non_empty, val_non_empty = train_test_split(non_empty_idxs, test_size=val_percent_float, random_state=42)

    train_indices = train_empty + train_non_empty
    val_indices = val_empty + val_non_empty

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"\n✅ Final dataset split:")
    print(f"🔹 Train set — {len(train_set)} samples (empty: {len(train_empty)}, non-empty: {len(train_non_empty)})")
    print(f"🔸 Val set   — {len(val_set)} samples (empty: {len(val_empty)}, non-empty: {len(val_non_empty)})\n")

    # 3. Create data loaders
    loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
    sampler = BalancedMaskSampler(train_set, empty_fraction=0.4)
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

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Weighted BCE to handle class imbalance

    global_step = 0
    optimizer.zero_grad(set_to_none=True)  # ✅ Initialize the gradient


    # ✅ Define once before the loop
    criterion = ComboLoss(weight=0.7)

    logging.info(f'Simulating effective batch size: {batch_size * accumulation_steps} '
             f'(actual batch size {batch_size}, accumulation steps {accumulation_steps})')

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                print(f"🧠 True mask stats: min={true_masks.min()} max={true_masks.max()} mean={true_masks.float().mean():.4f}")
                print("True mask unique values:", true_masks.unique())

                assert images.shape[1] == model.n_channels, \
                    f'Expected {model.n_channels} input channels, got {images.shape[1]}'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    with torch.no_grad():
                        bin_pred = (torch.sigmoid(masks_pred) > 0.5).float()
                        intersection = (bin_pred * true_masks).sum()
                        union = bin_pred.sum() + true_masks.sum()
                        dice = (2 * intersection + 1e-6) / (union + 1e-6)
                        logging.info(f'📏 Train Dice score: {dice.item():.4f}')

                pred_probs = torch.sigmoid(masks_pred)
                if global_step % 50 == 0:  # Every 50 steps
                    bin_mask = (pred_probs > 0.5).float()
                    ratio = bin_mask.mean().item()
                    logging.info(f"Predicted mask foreground %: {ratio*100:.2f}%")

                print(f"📊 Batch Loss: {loss.item():.6f}")
                print(f"✅ Prediction sigmoid stats: min={pred_probs.min().item():.6f} max={pred_probs.max().item():.6f}")
                print(f"🖼️ Input image stats: min={images.min().item():.6f}, max={images.max().item():.6f}")

                # ✅ Accumulate loss and gradients
                loss = loss / accumulation_steps
                grad_scaler.scale(loss).backward()

                if (global_step + 1) % accumulation_steps == 0:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                     # ✅ Log gradient norm here
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    logging.info(f"Gradient Norm: {total_norm:.4f}")
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)


                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # ✅ Evaluate after each epoch
                if global_step % 40 == 0:
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)
                    logging.info(f'Validation Dice score: {val_score:.4f}')
                if device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
                    reserved = torch.cuda.memory_reserved() / 1024 ** 2    # in MB
                    logging.info(f'📊 GPU Memory — Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB')

        logging.info(f"Epoch {epoch} summary — Loss: {epoch_loss:.4f} | Val Dice: {val_score:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")


         # ✅ Save checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), dir_checkpoint / f'checkpoint_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved!')
            


    
    

    # Load images
    input_image = Image.open(input_path).convert('L')
    mask = Image.open(mask_path).convert('L')

    # Preprocess
    img_tensor = TF.to_tensor(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        pred_np = pred.squeeze().cpu().numpy()

    # Plot input, ground truth mask, and prediction
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.array(input_image), cmap='gray')
    axes[0].set_title('Input Image')

    axes[1].imshow(np.array(mask), cmap='gray')
    axes[1].set_title('Ground Truth Mask')

    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title('Predicted Mask')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    model.train()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = HybridUNet(
    n_channels=config['n_channels'],
    n_classes=config['n_classes'],
    bilinear=config['bilinear']
)
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
            device=device,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            img_scale=config['img_scale'],
            val_percent=config['val_percent'],
            amp=config['amp'],
            weight_decay=config['weight_decay'],
            gradient_clipping=config['gradient_clipping'],
            accumulation_steps=4
        )

    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            device=device,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            img_scale=config['img_scale'],
            val_percent=config['val_percent'],
            amp=config['amp'],
            weight_decay=config['weight_decay'],
            gradient_clipping=config['gradient_clipping'],
            accumulation_steps=4
)

