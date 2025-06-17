import matplotlib.pyplot as plt

# Validation Dice scores for epochs 1-10
val_dice = [
    0.8131, 0.9017, 0.8094, 0.8400, 0.9344,
    0.9394, 0.8808, 0.9360, 0.8970, 0.9368
]

epochs = list(range(1, len(val_dice) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_dice, marker='o', label='Validation Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.title('Validation Dice Score over Epochs')
plt.xticks(epochs)
plt.ylim(0.7, 1.0)
plt.grid(True)
plt.legend()
plt.show() 