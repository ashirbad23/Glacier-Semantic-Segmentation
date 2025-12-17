import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from transform import GlacierAugment


class GlacierDataset(Dataset):
    def __init__(self, base_path: str, patch_size=256):
        self.base_path = base_path
        self.band_folders = os.listdir(base_path)[:-1]
        self.labels_folder = os.listdir(base_path)[-1]
        self.patch_size = patch_size
        self.transform = GlacierAugment()

        # Extract image IDs from the first band folder
        self.image_ids = [fname.split("_")[-2] + "_" + fname.split("_")[-1].split(".")[0]
                          for fname in os.listdir(os.path.join(self.base_path, self.band_folders[0]))]

        self.samples = self._generate_samples()

    def __len__(self):
        return len(self.samples)

    def _generate_samples(self):
        samples = []
        random_patches_per_image = 5  # you can increase if you want more random patches

        for img_id in self.image_ids:
            # --- Load all bands & label once ---
            bands_list = []
            for folder in self.band_folders:
                band_path = os.path.join(self.base_path, folder,
                                         [f for f in os.listdir(os.path.join(self.base_path, folder)) if img_id in f][
                                             0])
                band_img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
                bands_list.append(band_img)
            bands_stack = np.stack(bands_list, axis=0)  # shape: [bands, H, W]

            label_path = os.path.join(self.base_path, self.labels_folder,
                                      [f for f in os.listdir(os.path.join(self.base_path, self.labels_folder)) if
                                       img_id in f][0])
            label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            H, W = label_img.shape

            # --- 1. Generate 16-grid patches with 50% overlap ---
            stride = self.patch_size // 2
            for y in range(0, H - self.patch_size + 1, stride):
                for x in range(0, W - self.patch_size + 1, stride):
                    patch_label = label_img[y:y + self.patch_size, x:x + self.patch_size]
                    patch_bands = bands_stack[:, y:y + self.patch_size, x:x + self.patch_size]

                    if patch_bands.sum() == 0:
                        continue  # skip empty patches

                    for t_id in self.transform.transform_list:
                        aug_bands, aug_label = self.transform.apply_offline(patch_bands, patch_label, t_id)
                        if aug_bands.sum() > 0:
                            samples.append((img_id, x, y, self.patch_size, self.patch_size, t_id))

            # --- 2. Add random non-zero patches ---
            if self.patch_size != 512:
                non_zero_coords = np.argwhere(label_img > 0)  # where glacier pixels exist
                for _ in range(random_patches_per_image):
                    if len(non_zero_coords) == 0:
                        break
                    cy, cx = non_zero_coords[np.random.randint(0, len(non_zero_coords))]
                    y = max(0, min(cy - self.patch_size // 2, H - self.patch_size))
                    x = max(0, min(cx - self.patch_size // 2, W - self.patch_size))

                    patch_label = label_img[y:y + self.patch_size, x:x + self.patch_size]
                    patch_bands = bands_stack[:, y:y + self.patch_size, x:x + self.patch_size]
                    if patch_bands.sum() > 0:
                        for t_id in self.transform.transform_list:
                            aug_bands, aug_label = self.transform.apply_offline(patch_bands, patch_label, t_id)
                            if aug_bands.sum() > 0:
                                samples.append((img_id, x, y, self.patch_size, self.patch_size, t_id))

        return samples

    def get_image_ids(self):
        return [s[0] for s in self.samples]

    def __getitem__(self, idx):
        img_id, x, y, h, w, t_id = self.samples[idx]

        bands = []
        # Read all band images using OpenCV
        for folder in self.band_folders:
            band_dir = os.path.join(self.base_path, folder)
            band_file = [f for f in os.listdir(band_dir) if img_id in f][0]
            band_path = os.path.join(band_dir, band_file)
            img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)  # 16-bit unchanged
            bands.append(img[y:y + h, x:x + w])
        bands = np.stack(bands, axis=0).astype(np.float32)

        # Read label using OpenCV
        label_dir = os.path.join(self.base_path, self.labels_folder)
        label_file = [f for f in os.listdir(label_dir) if img_id in f][0]
        label_path = os.path.join(label_dir, label_file)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = label[y:y + h, x:x + w]

        # Apply transform if given (e.g., augmentations)
        bands, label = self.transform.apply_offline(bands, label, t_id)

        for i in range(bands.shape[0]):
            bmin, bmax = bands[i].min(), bands[i].max()
            bands[i] = (bands[i] - bmin) / (bmax - bmin + 1e-6)

        label = label // 85

        return torch.tensor(bands, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class GlacierDatasetOnline(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bands, label = self.samples[idx]

        # Convert to NumPy for Albumentations
        if isinstance(bands, torch.Tensor):
            bands = bands.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()

        bands = np.moveaxis(bands, 0, -1)  # (C,H,W) → (H,W,C)

        # Apply Albumentations transform (image augmentations)
        augmented = self.transform(image=bands, mask=label)
        bands = augmented['image']
        label = augmented['mask']

        # ✅ Normalize dtype after augmentation (some transforms cast to float)
        if isinstance(label, np.ndarray):
            label = np.round(label).astype(np.int64)
        elif isinstance(label, torch.Tensor):
            label = torch.round(label).long()

        # Albumentations outputs images as (H,W,C), need (C,H,W)
        bands = torch.as_tensor(bands, dtype=torch.float32).clone().detach()
        label = torch.as_tensor(label, dtype=torch.long).clone().detach()

        return bands, label


if __name__ == "__main__":
    base_path = "../data/Train"
    band_folders = os.listdir(base_path)[:-1]

    # Toggle normalize on/off here
    dataset = GlacierDataset(base_path=base_path, patch_size=128)

    # Test one sample
    bands, label = dataset[0]
    bands, label = bands.numpy(), label.numpy()
    print("Bands shape:", bands.shape)  # [5, H, W]
    print("Label shape:", label.shape)  # [H, W]
    print("Bands max:", bands.max())
    print("Label unique:", np.unique(label))
    print(len(dataset))

    # Plot comparison for first 3 bands
    plt.figure(figsize=(15, 6))
    for i in range(5):
        # Raw band
        plt.subplot(1, 6, i + 1)
        plt.imshow(bands[i], cmap='gray')
        plt.title(f"Raw Band {i + 1}\nMax: {bands[i].max():.0f}")
        plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(label, cmap='grey')
    plt.title(f"Raw Label \n{label.max()}")

    plt.tight_layout()
    plt.show()

    # class_count = np.zeros(4)
    #
    # for _, labels in dataset:
    #     for c in range(4):
    #         class_count[c] += np.sum(labels.numpy() == c)
    #
    # print(class_count)
