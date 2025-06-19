import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_hd95(mask_gt, mask_pred, spacing=(1.0, 1.0)):
    """
    Compute the 95th percentile Hausdorff Distance (HD95) between two binary masks.
    Lightweight implementation using distance transforms.
    
    mask_gt, mask_pred: numpy arrays of shape (H, W), binary (0/1 or bool)
    spacing: tuple, pixel spacing (default (1.0, 1.0))
    """
    mask_gt = np.asarray(mask_gt).astype(bool)
    mask_pred = np.asarray(mask_pred).astype(bool)
    
    # If either mask is empty, return a large distance
    if not np.any(mask_gt) or not np.any(mask_pred):
        return np.inf
    
    # Compute distance from prediction to ground truth
    dist_pred_to_gt = distance_transform_edt(~mask_gt, sampling=spacing)
    dist_pred_to_gt = dist_pred_to_gt[mask_pred]
    
    # Compute distance from ground truth to prediction
    dist_gt_to_pred = distance_transform_edt(~mask_pred, sampling=spacing)
    dist_gt_to_pred = dist_gt_to_pred[mask_gt]
    
    # Combine distances
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    
    # Compute 95th percentile
    if len(all_distances) > 0:
        hd95 = np.percentile(all_distances, 95)
    else:
        hd95 = np.inf
    
    return hd95 