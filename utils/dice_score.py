import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Ensure shapes match
    assert input.size() == target.size(), f"Shape mismatch: input {input.shape}, target {target.shape}"

    # Support [B, C, H, W] or [B, H, W]
    if input.dim() == 4:
        sum_dim = (-1, -2)  # sum over H and W
    elif input.dim() == 3:
        sum_dim = (-1, -2)
    elif input.dim() == 2:
        sum_dim = (-1,)
    else:
        raise ValueError(f"Unsupported input dimension {input.dim()}")

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Flatten channel into batch dimension if multiclass
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Main Dice loss wrapper
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
