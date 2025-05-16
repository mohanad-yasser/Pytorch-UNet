from torch.utils.data import Sampler
import numpy as np
from tqdm import tqdm

class BalancedMaskSampler(Sampler):
    def __init__(self, dataset, empty_fraction=0.2):
        self.empty_fraction = empty_fraction
        self.dataset = dataset
        self.empty_idxs = []
        self.non_empty_idxs = []

        total = len(dataset)
        print(f"\n📦 Total masks found in dataset: {total}")
        print("⏳ Scanning dataset to classify masks...")

        progress = tqdm(total=total, desc="🔍 Scanning masks", unit="mask")

        for i in range(total):
            mask = dataset[i]['mask'].numpy()
            if np.max(mask) == 0:
                self.empty_idxs.append(i)
            else:
                self.non_empty_idxs.append(i)

            progress.set_postfix({
                "scanned": i + 1,
                "left": total - (i + 1)
            })
            progress.update(1)

        progress.close()

        print(f"\n📊 Summary:")
        print(f"🟤 Empty masks               : {len(self.empty_idxs)}")
        print(f"🟢 Non-empty masks           : {len(self.non_empty_idxs)}")
        print(f"🎯 Sampling {self.empty_fraction*100:.1f}% of empty masks per epoch\n")

    def __iter__(self):
        sampled_empty = np.random.choice(
            self.empty_idxs,
            size=int(len(self.empty_idxs) * self.empty_fraction),
            replace=False
        ).astype(np.int32)  # Ensures integer indices

        combined = np.concatenate([self.non_empty_idxs, sampled_empty])
        np.random.shuffle(combined)
        return iter(combined.tolist())


    def __len__(self):
        return int(len(self.non_empty_idxs) + len(self.empty_idxs) * self.empty_fraction)
