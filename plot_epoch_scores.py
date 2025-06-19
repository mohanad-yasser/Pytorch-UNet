import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress():
    """Plot the training progress for ResUNet with augmentations"""
    
    epochs = list(range(1, 16))  # Epochs 1-15
    
    # ResUNet WITH augmentations
    dice_scores = [
        0.597,  # Epoch 1
        0.592,  # Epoch 2
        0.573,  # Epoch 3
        0.583,  # Epoch 4
        0.556,  # Epoch 5
        0.607,  # Epoch 6
        0.571,  # Epoch 7
        0.612,  # Epoch 8
        0.591,  # Epoch 9
        0.615,  # Epoch 10
        0.616,  # Epoch 11
        0.566,  # Epoch 12
        0.621,  # Epoch 13
        0.621,  # Epoch 14
        0.626   # Epoch 15
    ]
    
    hd95_scores = [
        15.90,  # Epoch 1
        15.97,  # Epoch 2
        17.66,  # Epoch 3
        17.84,  # Epoch 4
        18.13,  # Epoch 5
        16.39,  # Epoch 6
        18.37,  # Epoch 7
        16.26,  # Epoch 8
        18.08,  # Epoch 9
        16.11,  # Epoch 10
        15.34,  # Epoch 11
        18.60,  # Epoch 12
        15.81,  # Epoch 13
        14.98,  # Epoch 14
        14.35   # Epoch 15
    ]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Dice scores
    ax1.plot(epochs, dice_scores, 'b-o', linewidth=2, markersize=6, label='ResUNet (With Aug)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Validation Dice Score Progress (ResUNet with Augmentations)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight best Dice score
    best_dice_epoch = np.argmax(dice_scores) + 1
    best_dice_score = max(dice_scores)
    ax1.plot(best_dice_epoch, best_dice_score, 'ro', markersize=12, label=f'Best: {best_dice_score:.3f} (Epoch {best_dice_epoch})')
    ax1.legend()
    
    # Plot HD95 scores
    ax2.plot(epochs, hd95_scores, 'g-o', linewidth=2, markersize=6, label='ResUNet (With Aug)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('HD95 Score (pixels)')
    ax2.set_title('Validation HD95 Score Progress (ResUNet with Augmentations)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Highlight best HD95 score
    best_hd95_epoch = np.argmin(hd95_scores) + 1
    best_hd95_score = min(hd95_scores)
    ax2.plot(best_hd95_epoch, best_hd95_score, 'ro', markersize=12, label=f'Best: {best_hd95_score:.2f} (Epoch {best_hd95_epoch})')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("RESUNET TRAINING RESULTS (15 Epochs, WITH Augmentations)")
    print("=" * 60)
    print(f"Best Dice Score: {best_dice_score:.3f} (Epoch {best_dice_epoch})")
    print(f"Best HD95 Score: {best_hd95_score:.2f} pixels (Epoch {best_hd95_epoch})")
    print(f"Average Dice Score: {np.mean(dice_scores):.3f}")
    print(f"Average HD95 Score: {np.mean(hd95_scores):.2f} pixels")
    print(f"Final Dice Score: {dice_scores[-1]:.3f}")
    print(f"Final HD95 Score: {hd95_scores[-1]:.2f} pixels")
    
    # Performance analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Find epochs with Dice > 0.6
    high_dice_epochs = [i+1 for i, score in enumerate(dice_scores) if score > 0.6]
    print(f"Epochs with Dice > 0.6: {high_dice_epochs}")
    
    # Find epochs with HD95 < 15
    low_hd95_epochs = [i+1 for i, score in enumerate(hd95_scores) if score < 15]
    print(f"Epochs with HD95 < 15: {low_hd95_epochs}")
    
    # Stability analysis
    dice_std = np.std(dice_scores)
    hd95_std = np.std(hd95_scores)
    print(f"Dice Score Stability (std): {dice_std:.3f}")
    print(f"HD95 Score Stability (std): {hd95_std:.2f}")
    
    # Training progression
    print(f"\nTraining Progression:")
    print(f"Early (Epochs 1-5): Dice avg = {np.mean(dice_scores[:5]):.3f}, HD95 avg = {np.mean(hd95_scores[:5]):.2f}")
    print(f"Mid (Epochs 6-10): Dice avg = {np.mean(dice_scores[5:10]):.3f}, HD95 avg = {np.mean(hd95_scores[5:10]):.2f}")
    print(f"Late (Epochs 11-15): Dice avg = {np.mean(dice_scores[10:]):.3f}, HD95 avg = {np.mean(hd95_scores[10:]):.2f}")
    
    # Recommendation
    print(f"\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"✅ Use checkpoint from Epoch {best_dice_epoch} for best Dice performance")
    print(f"✅ Use checkpoint from Epoch {best_hd95_epoch} for best HD95 performance")
    print(f"✅ Overall best: Epoch {best_dice_epoch} (Dice: {best_dice_score:.3f}, HD95: {hd95_scores[best_dice_epoch-1]:.2f})")

if __name__ == '__main__':
    plot_training_progress() 