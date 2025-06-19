import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress():
    """Plot the training progress for CBRDilatedUNet with conservative augmentations (20 epochs)"""
    
    epochs = list(range(1, 21))  # Epochs 1-20
    
    # CBRDilatedUNet WITH conservative augmentations (actual training data)
    dice_scores = [
        0.592,  # Epoch 1
        0.592,  # Epoch 2
        0.610,  # Epoch 3
        0.587,  # Epoch 4
        0.624,  # Epoch 5
        0.606,  # Epoch 6
        0.624,  # Epoch 7
        0.567,  # Epoch 8
        0.618,  # Epoch 9
        0.621,  # Epoch 10
        0.628,  # Epoch 11
        0.582,  # Epoch 12
        0.618,  # Epoch 13
        0.593,  # Epoch 14
        0.598,  # Epoch 15
        0.621,  # Epoch 16
        0.601,  # Epoch 17
        0.621,  # Epoch 18
        0.619,  # Epoch 19
        0.591   # Epoch 20
    ]
    
    hd95_scores = [
        15.36,  # Epoch 1
        15.36,  # Epoch 2
        13.88,  # Epoch 3
        18.10,  # Epoch 4
        15.27,  # Epoch 5
        15.24,  # Epoch 6
        15.21,  # Epoch 7
        15.46,  # Epoch 8
        15.40,  # Epoch 9
        15.48,  # Epoch 10
        15.28,  # Epoch 11
        17.67,  # Epoch 12
        15.53,  # Epoch 13
        float('inf'),  # Epoch 14 (inf)
        17.40,  # Epoch 15
        15.48,  # Epoch 16
        17.39,  # Epoch 17
        15.32,  # Epoch 18
        15.44,  # Epoch 19
        15.16   # Epoch 20
    ]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Dice scores
    ax1.plot(epochs, dice_scores, 'g-o', linewidth=2, markersize=6, label='CBRDilatedUNet (With Aug)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Validation Dice Score Progress (CBRDilatedUNet - 20 Epochs with Augmentations)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight best Dice score
    best_dice_epoch = np.argmax(dice_scores) + 1
    best_dice_score = max(dice_scores)
    ax1.plot(best_dice_epoch, best_dice_score, 'ro', markersize=12, 
             label=f'Best: {best_dice_score:.3f} (Epoch {best_dice_epoch})')
    ax1.legend()
    
    # Plot HD95 scores (handle inf values)
    hd95_plot = [x if x != float('inf') else 20 for x in hd95_scores]  # Replace inf with 20 for plotting
    ax2.plot(epochs, hd95_plot, 'g-o', linewidth=2, markersize=6, label='CBRDilatedUNet (With Aug)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('HD95 Score (pixels)')
    ax2.set_title('Validation HD95 Score Progress (CBRDilatedUNet - 20 Epochs with Augmentations)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Highlight best HD95 score (excluding inf)
    valid_hd95 = [x for x in hd95_scores if x != float('inf')]
    best_hd95_epoch = np.argmin(valid_hd95) + 1
    best_hd95_score = min(valid_hd95)
    ax2.plot(best_hd95_epoch, best_hd95_score, 'ro', markersize=12, 
             label=f'Best: {best_hd95_score:.2f} (Epoch {best_hd95_epoch})')
    ax2.legend()
    
    # Mark inf values with special marker
    inf_epochs = [i+1 for i, x in enumerate(hd95_scores) if x == float('inf')]
    if inf_epochs:
        ax2.plot(inf_epochs, [20]*len(inf_epochs), 'rx', markersize=10, 
                label=f'Inf values (Epochs {inf_epochs})')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=" * 70)
    print("CBRDILATEDUNET TRAINING RESULTS (20 Epochs, WITH Conservative Augmentations)")
    print("=" * 70)
    print(f"Best Dice Score: {best_dice_score:.3f} (Epoch {best_dice_epoch})")
    print(f"Best HD95 Score: {best_hd95_score:.2f} pixels (Epoch {best_hd95_epoch})")
    print(f"Average Dice Score: {np.mean(dice_scores):.3f}")
    print(f"Average HD95 Score: {np.mean(valid_hd95):.2f} pixels (excluding inf)")
    print(f"Final Dice Score: {dice_scores[-1]:.3f}")
    print(f"Final HD95 Score: {hd95_scores[-1]:.2f} pixels")
    
    # Performance analysis
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Find epochs with Dice > 0.60
    high_dice_epochs = [i+1 for i, score in enumerate(dice_scores) if score > 0.60]
    print(f"Epochs with Dice > 0.60: {high_dice_epochs}")
    
    # Find epochs with HD95 < 16
    low_hd95_epochs = [i+1 for i, score in enumerate(hd95_scores) if score != float('inf') and score < 16]
    print(f"Epochs with HD95 < 16: {low_hd95_epochs}")
    
    # Stability analysis
    dice_std = np.std(dice_scores)
    hd95_std = np.std(valid_hd95)
    print(f"Dice Score Stability (std): {dice_std:.3f}")
    print(f"HD95 Score Stability (std): {hd95_std:.2f}")
    
    # Training progression
    print(f"\nTraining Progression:")
    print(f"Early (Epochs 1-7): Dice avg = {np.mean(dice_scores[:7]):.3f}, HD95 avg = {np.mean([x for x in hd95_scores[:7] if x != float('inf')]):.2f}")
    print(f"Mid (Epochs 8-14): Dice avg = {np.mean(dice_scores[7:14]):.3f}, HD95 avg = {np.mean([x for x in hd95_scores[7:14] if x != float('inf')]):.2f}")
    print(f"Late (Epochs 15-20): Dice avg = {np.mean(dice_scores[14:]):.3f}, HD95 avg = {np.mean([x for x in hd95_scores[14:] if x != float('inf')]):.2f}")
    
    # Comparison with previous results
    print(f"\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS RESULTS (No Augmentations)")
    print("=" * 70)
    print(f"Previous Best Dice: 0.726 (Epoch 6) vs Current Best: {best_dice_score:.3f} (Epoch {best_dice_epoch})")
    print(f"Previous Best HD95: 9.09 (Epoch 6) vs Current Best: {best_hd95_score:.2f} (Epoch {best_hd95_epoch})")
    print(f"Dice Change: {best_dice_score - 0.726:+.3f}")
    print(f"HD95 Change: {best_hd95_score - 9.09:+.2f}")
    
    # Recommendation
    print(f"\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"✅ Use checkpoint from Epoch {best_dice_epoch} for best Dice performance")
    print(f"   Dice: {best_dice_score:.3f}, HD95: {hd95_scores[best_dice_epoch-1]:.2f}")
    print(f"⚠️  Augmentations decreased performance compared to no-aug version")
    print(f"⚠️  Consider reducing augmentation probability or removing augmentations")
    print(f"✅ Model shows more stable training with less overfitting")

if __name__ == '__main__':
    plot_training_progress() 