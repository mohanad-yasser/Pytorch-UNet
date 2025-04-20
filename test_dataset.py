import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.data_loading import apply_bias_correction, apply_clahe

# ✅ Load and resize MRI image
img_path = 'data/imgs/volume_1_slice_46.png'
image = Image.open(img_path).convert('L').resize((256, 256))
img_np = np.asarray(image).astype(np.float32)

# ✅ Apply CLAHE → Bias Correction
img_clahe = apply_clahe(img_np)
img_bias = apply_bias_correction(img_clahe)

# ✅ Regular Z-score (all pixels)
img_all = img_bias.copy()
img_all = img_all[np.newaxis, ...]
mean_all = img_all.mean()
std_all = img_all.std()
img_z_all = (img_all - mean_all) / (std_all if std_all != 0 else 1.0)

# ✅ Brain-only Z-score (exclude zeros)
img_brain = img_bias.copy()
img_brain = img_brain[np.newaxis, ...]
brain_mask = img_brain > 0
mean_brain = img_brain[brain_mask].mean()
std_brain = img_brain[brain_mask].std()
img_z_brain = (img_brain - mean_brain) / (std_brain if std_brain != 0 else 1.0)

# ✅ Plot all
plt.figure(figsize=(12, 3))

plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap='gray')
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(img_z_all.squeeze(0), cmap='gray')
plt.title("Regular Z-Score")

plt.subplot(1, 3, 3)
plt.imshow(img_z_brain.squeeze(0), cmap='gray')
plt.title("Brain-Only Z-Score")

plt.tight_layout()
plt.show()
