import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet.hybrid_unet_model import HybridUNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks_binary/')
dir_checkpoint = Path('./checkpoints/')

# 🚀 Hyperparameter Configuration
config = {
    'epochs': 25,
    'batch_size': 4,
    'learning_rate': 1e-5,
    'val_percent': 0.2,         # 10% validation split
    'img_scale': 1.0,           # full 240x240 images
    'amp': False,               # no mixed precision yet
    'bilinear': True,           # use bilinear upsampling
    'n_channels': 1,            # grayscale input (T1 MRI slices)
    'n_classes': 1,             # binary output (tumor or no tumor)
    'weight_decay': 1e-8,
    'momentum': 0.999,
    'gradient_clipping': 1.0,
}

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

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
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Weighted BCE to handle class imbalance
    if model.n_classes == 1:
        pos_weight = torch.tensor([7.0], device=device)  # 5.0 is a good starting point
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    global_step = 0

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
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # ✅ ⬇️ ADD dynamic pos_weight calculation here (before forward pass)
                batch_size = true_masks.shape[0]
                total_pixels = torch.numel(true_masks)
                num_positives = true_masks.sum()
                num_negatives = total_pixels - num_positives

                if num_positives == 0:
                    dynamic_pos_weight = torch.tensor(1.0, device=device)
                else:
                    dynamic_pos_weight = num_negatives / num_positives

                # 🚨 Clamp pos_weight for safety
                dynamic_pos_weight = torch.clamp(dynamic_pos_weight, min=1.0, max=20.0)

                

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                   
                    # 🛑 DO NOT print raw output before activation unless for serious debugging
                    pred_probs = torch.sigmoid(masks_pred.squeeze(1))

                    # 🎯 Proper losses
                    if model.n_classes == 1:
                        loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks.float(), pos_weight=dynamic_pos_weight)
                        pred_probs = torch.sigmoid(masks_pred)  # <== move this inside
                        loss += dice_loss(pred_probs.squeeze(1), true_masks.float().squeeze(1), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # ✅ Print after exiting autocast
                print(f"📊 Batch Loss: {loss.item():.6f}")
                print(f"✅ Prediction sigmoid stats: min={pred_probs.min().item():.6f} max={pred_probs.max().item():.6f}")
                print(f"🧠 True mask unique values: {true_masks.unique(sorted=True)}")
                print(f"🖼️ Input image stats: min={images.min().item():.6f}, max={images.max().item():.6f}")

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            #state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            


    
    # Load the input slice
    input_path = "data/imgs/volume_1_slice_60.png"
    mask_path = "data/masks/volume_1_slice_60_mask.png"

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
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
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
            momentum=config['momentum'],
            gradient_clipping=config['gradient_clipping']
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
            momentum=config['momentum'],
            gradient_clipping=config['gradient_clipping']
)

