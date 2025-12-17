import numpy as np
from scipy.ndimage import rotate
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GlacierAugment:
    def __init__(self):
        # Fixed transform IDs:
        # 0: original, 1: rot90, 2: rot180, 3: rot270, 4: hflip, 5: vflip,
        # 6: rot45, 7: rot135, 8: rot225, 9: rot315
        self.transform_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def apply_offline(self, bands, label, transform_id):
        if transform_id == 1:  # rot90
            bands = np.rot90(bands, k=1, axes=(1, 2))
            label = np.rot90(label, k=1)
        elif transform_id == 2:  # rot180
            bands = np.rot90(bands, k=2, axes=(1, 2))
            label = np.rot90(label, k=2)
        elif transform_id == 3:  # rot270
            bands = np.rot90(bands, k=3, axes=(1, 2))
            label = np.rot90(label, k=3)
        elif transform_id == 4:  # hflip
            bands = np.flip(bands, axis=2)
            label = np.flip(label, axis=1)
        elif transform_id == 5:  # vflip
            bands = np.flip(bands, axis=1)
            label = np.flip(label, axis=0)
        elif transform_id in [6, 7, 8, 9]:  # 45°, 135°, 225°, 315°
            angle_map = {6: 45, 7: 135, 8: 225, 9: 315}
            angle = angle_map[transform_id]

            bands_rot = np.zeros_like(bands)
            for i in range(bands.shape[0]):
                bands_rot[i] = rotate(bands[i], angle, reshape=False, order=1, mode='reflect')
            label = rotate(label, angle, reshape=False, order=0, mode='reflect')
            bands = bands_rot

        return bands.copy(), label.copy()

    @staticmethod
    def get_train_augmentations():
        """
        Light, realistic online augmentations for glacier patches.
        You already normalized pixel values beforehand, so no Normalize().
        """
        return A.Compose([
            A.Affine(scale=(1 - 0.03, 1 + 0.03), translate_percent=(0.02, 0.02), rotate=(-5, 5), p=0.5),
            ToTensorV2(),
        ])

    @staticmethod
    def get_val_augmentations():
        """
        No augmentation for validation — only convert to tensor.
        """
        return A.Compose([
            ToTensorV2(),
        ])
