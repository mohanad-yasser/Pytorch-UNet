import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.hd95 import compute_hd95


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    hd95_score = 0
    n_samples = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred_bin = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred_bin, mask_true, reduce_batch_first=False)
                # HD95: convert to numpy, remove batch/channel dims
                for i in range(mask_true.shape[0]):
                    hd95_score += compute_hd95(
                        mask_true[i, 0].cpu().numpy(),
                        mask_pred_bin[i, 0].cpu().numpy()
                    )
                    n_samples += 1
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true_oh = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_oh = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred_oh[:, 1:], mask_true_oh[:, 1:], reduce_batch_first=False)
                # HD95 for foreground class only
                for i in range(mask_true.shape[0]):
                    hd95_score += compute_hd95(
                        mask_true_oh[i, 1].cpu().numpy(),
                        mask_pred_oh[i, 1].cpu().numpy()
                    )
                    n_samples += 1

    net.train()
    avg_dice = dice_score / max(num_val_batches, 1)
    avg_hd95 = hd95_score / max(n_samples, 1)
    return avg_dice, avg_hd95
