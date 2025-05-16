import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def focal_tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, gamma=1.33, eps=1e-7):
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()

    tp = (y_pred * y_true).sum(dim=(1, 2, 3))
    fp = ((1 - y_true) * y_pred).sum(dim=(1, 2, 3))
    fn = (y_true * (1 - y_pred)).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = (1 - tversky) ** gamma
    return loss.mean()


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, bce_smoothing=0.0, threshold=0.5):
    net.eval()

    # reproduce same weighted BCE as in training
    pos_frac   = 0.1
    pos_weight = torch.tensor((1.0 - pos_frac) / pos_frac, device=device)
    class SmoothedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
        def __init__(self, smoothing=0.0, pos_weight=None):
            super().__init__(pos_weight=pos_weight)
            self.smoothing = smoothing

        def forward(self, pred, target):
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing
            return super().forward(pred, target)

    bce_loss_fn = SmoothedBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=bce_smoothing)


    # combined focal-Tversky + BCE
    def combined_loss(pred, true):
        ft = focal_tversky_loss(pred, true)
        ce = bce_loss_fn(pred, true)
        return 0.5 * ft + 0.5 * ce

    num_val_batches = len(dataloader)
    total_loss = 0.0
    dice_score = 0.0
    valid_batches = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            mask_logits = net(image)
            non_empty = mask_true.view(mask_true.size(0), -1).sum(dim=1) > 0
            if non_empty.sum() == 0:
                continue

            valid_batches += 1
            total_loss += combined_loss(mask_logits[non_empty], mask_true[non_empty]).item()

            if net.n_classes == 1:
                prob = torch.sigmoid(mask_logits[non_empty])
                pred_bin = (prob > threshold).float().squeeze(1)
                true_flat = mask_true[non_empty].squeeze(1)
                dice_score += dice_coeff(pred_bin, true_flat, reduce_batch_first=False)
            else:
                true_onehot = F.one_hot(mask_true.long(), net.n_classes).permute(0, 3, 1, 2).float()
                pred_labels = mask_logits.argmax(dim=1)
                pred_onehot = F.one_hot(pred_labels, net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(pred_onehot[:,1:], true_onehot[:,1:], reduce_batch_first=False)

    avg_loss = total_loss / max(valid_batches, 1)
    avg_dice = dice_score  / max(valid_batches, 1)
    print(f"🧪 Validation — Avg Dice: {avg_dice:.4f} | Avg Loss: {avg_loss:.4f}")

    net.train()
    return avg_dice
