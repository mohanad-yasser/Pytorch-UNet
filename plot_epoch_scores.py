import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress():
    """Plot the training progress from the logs"""
    
    # Training results from your 15-epoch run with augmentations
    epochs = list(range(1, 16))  # Epochs 1-15
    
    # Dice scores from your training logs (with augmentations)
    dice_scores = [
        0.564,  # Epoch 1 (actual from log)
        0.584,  # Epoch 2 (actual from log)
        0.515,  # Epoch 3 (actual from log)
        0.592,  # Epoch 4 (actual from log)
        0.513,  # Epoch 5 (actual from log)
        0.538,  # Epoch 6 (actual from log)
        0.581,  # Epoch 7 (actual from log)
        0.604,  # Epoch 8 (actual from log)
        0.604,  # Epoch 9 (actual from log)
        0.538,  # Epoch 10 (actual from log)
        0.529,  # Epoch 11 (actual from log)
        0.618,  # Epoch 12 (actual from log)
        0.606,  # Epoch 13 (actual from log)
        0.602,  # Epoch 14 (actual from log)
        0.511   # Epoch 15 (actual from log)
    ]
    
    # HD95 scores from your training logs (with augmentations)
    hd95_scores = [
        17.76,  # Epoch 1 (actual from log)
        16.58,  # Epoch 2 (actual from log)
        19.83,  # Epoch 3 (actual from log)
        16.29,  # Epoch 4 (actual from log)
        19.29,  # Epoch 5 (actual from log)
        17.90,  # Epoch 6 (actual from log)
        17.21,  # Epoch 7 (actual from log)
        16.26,  # Epoch 8 (actual from log)
        14.90,  # Epoch 9 (actual from log)
        18.80,  # Epoch 10 (actual from log)
        19.06,  # Epoch 11 (actual from log)
        15.05,  # Epoch 12 (actual from log)
        15.11,  # Epoch 13 (actual from log)
        15.00,  # Epoch 14 (actual from log)
        19.69   # Epoch 15 (actual from log)
    ]
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot Dice scores
    ax1.plot(epochs, dice_scores, 'b-o', linewidth=2, markersize=6, label='Validation Dice')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Training Progress with Augmentations - Dice Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.45, 0.7)
    
    # Highlight best epoch
    best_dice_epoch = np.argmax(dice_scores) + 1
    best_dice_score = max(dice_scores)
    ax1.plot(best_dice_epoch, best_dice_score, 'ro', markersize=10, label=f'Best: {best_dice_score:.3f} (Epoch {best_dice_epoch})')
    ax1.legend()
    
    # Add value annotations
    for i, (epoch, score) in enumerate(zip(epochs, dice_scores)):
        ax1.annotate(f'{score:.3f}', (epoch, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # Plot HD95 scores
    ax2.plot(epochs, hd95_scores, 'r-o', linewidth=2, markersize=6, label='Validation HD95')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('HD95 (pixels)')
    ax2.set_title('Training Progress with Augmentations - HD95 Score (Lower is Better)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(10, 22)
    
    # Highlight best epoch for HD95
    best_hd95_epoch = np.argmin(hd95_scores) + 1
    best_hd95_score = min(hd95_scores)
    ax2.plot(best_hd95_epoch, best_hd95_score, 'ro', markersize=10, label=f'Best: {best_hd95_score:.2f} (Epoch {best_hd95_epoch})')
    ax2.legend()
    
    # Add value annotations
    for i, (epoch, score) in enumerate(zip(epochs, hd95_scores)):
        ax2.annotate(f'{score:.2f}', (epoch, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY (15 EPOCHS WITH AUGMENTATIONS)")
    print("="*60)
    print(f"Best Dice Score: {best_dice_score:.3f} (Epoch {best_dice_epoch})")
    print(f"Best HD95 Score: {best_hd95_score:.2f} (Epoch {best_hd95_epoch})")
    print(f"Average Dice Score: {np.mean(dice_scores):.3f}")
    print(f"Average HD95 Score: {np.mean(hd95_scores):.2f}")
    print(f"Final Dice Score: {dice_scores[-1]:.3f} (Epoch {epochs[-1]})")
    print(f"Final HD95 Score: {hd95_scores[-1]:.2f} (Epoch {epochs[-1]})")
    print("="*60)
    
    # Compare with previous run (no augmentations)
    print("\nCOMPARISON WITH PREVIOUS RUN (NO AUGMENTATIONS):")
    print("Previous Best Dice: 0.667 (Epoch 10)")
    print("Previous Best HD95: 13.10 (Epoch 6)")
    print("Current Best Dice:  {:.3f} (Epoch {})".format(best_dice_score, best_dice_epoch))
    print("Current Best HD95:  {:.2f} (Epoch {})".format(best_hd95_score, best_hd95_epoch))
    
    return best_dice_epoch, best_hd95_epoch

if __name__ == '__main__':
    best_dice_epoch, best_hd95_epoch = plot_training_progress()
    print(f"\nRecommended checkpoint for Dice: checkpoint_epoch{best_dice_epoch}.pth")
    print(f"Recommended checkpoint for HD95: checkpoint_epoch{best_hd95_epoch}.pth") 