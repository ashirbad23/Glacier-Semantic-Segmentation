# torch torchvision numpy opencv-python matplotlib scikit-learn pillow segmentation-models-pytorch

import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Model Definition ----------------
class ConvBlock(nn.Module):
    """(Conv => BN => GeLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, deep_supervision=False, base_filters=32, fusion=False):
        super(UNetPP, self).__init__()
        self.deep_supervision = deep_supervision
        self.fusion = fusion

        nb_filter = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder (nested)
        self.up1_0 = Up(nb_filter[1], nb_filter[0])
        self.up2_0 = Up(nb_filter[2], nb_filter[1])
        self.up3_0 = Up(nb_filter[3], nb_filter[2])
        self.up4_0 = Up(nb_filter[4], nb_filter[3])

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[0], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[1], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[2], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[3], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[0], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[1], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[2], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[0], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[1], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[0], nb_filter[0])

        # Deep supervision heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder path (nested connections)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_0(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_0(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_0(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)], 1))

        if self.fusion:
            return x0_4
        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


def get_tile_id(filename):
    match = re.search(r"img(\d+)\.tif", filename)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else None


def maskgeration(imagepath, model_path, patch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    band_tile_map = {b: {} for b in imagepath}

    for band, folder in imagepath.items():
        for fname in os.listdir(folder):
            if fname.endswith(".tif"):
                tid = get_tile_id(fname)
                if tid:
                    band_tile_map[band][tid] = os.path.join(folder, fname)

    ref_band = sorted(imagepath.keys())[0]
    tile_ids = sorted(band_tile_map[ref_band].keys())
    results = {}

    model = UNetPP(5, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for tid in tile_ids:
            bands = []
            for band in sorted(imagepath.keys()):
                arr = cv2.imread(band_tile_map[band][tid], cv2.IMREAD_UNCHANGED).astype(np.float32)
                bands.append(arr)
            bands = np.stack(bands, 0)

            # Normalize
            for i in range(bands.shape[0]):
                bmin, bmax = bands[i].min(), bands[i].max()
                bands[i] = (bands[i] - bmin) / (bmax - bmin + 1e-6)

            C, H, W = bands.shape
            mask = np.zeros((H, W), np.uint8)

            for y in range(0, bands.shape[1], patch_size):
                for x in range(0, bands.shape[2], patch_size):
                    patch = bands[:, y:y + patch_size, x:x + patch_size]
                    if patch.sum() == 0:
                        continue
                    inp = torch.from_numpy(patch).unsqueeze(0).to(device)
                    out = model(inp)
                    pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
                    mask[y:y + patch_size, x:x + patch_size] = pred[:patch_size, :patch_size]

            results[tid] = (mask * 85).astype(np.uint8)

    return results


# ---------------- Main ----------------
def main():
    import time
    import matplotlib.pyplot as plt
    from sklearn.metrics import matthews_corrcoef

    # ---- Helper for MCC ----
    def mcc_score(y_true, y_pred):
        y_true = (y_true // 85).astype(np.int32)
        y_pred = (y_pred // 85).astype(np.int32)
        return matthews_corrcoef(y_true.ravel(), y_pred.ravel())

    # ---- Paths ----
    image_path = {
        "B1": "../data/Train/Band1",
        "B2": "../data/Train/Band2",
        "B3": "../data/Train/Band3",
        "B4": "../data/Train/Band4",
        "B5": "../data/Train/Band5",
    }
    mask_path = "../data/Train/labels"
    model_path = "weights/UnetPP_Normal/model.pth"
    output_dir = "evaluation"
    results_dir = "results"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    patch_size = 128

    # ---- Inference ----
    print("🚀 Running inference...")
    t1 = time.time()
    preds = maskgeration(image_path, model_path, patch_size)
    print(f"✅ Inference completed in {time.time() - t1:.2f}s")

    # ---- Evaluation ----
    mcc_total = 0
    num_evaluated = 0

    for tid, mask in preds.items():
        # Save predicted mask
        save_path = os.path.join(results_dir, f"{tid}.png")
        cv2.imwrite(save_path, mask)

        # Find corresponding ground truth
        labels = [img for img in os.listdir(mask_path) if tid in img]
        if not labels:
            continue

        ref_img = cv2.imread(os.path.join(mask_path, labels[0]), cv2.IMREAD_UNCHANGED)
        mcc = mcc_score(ref_img, mask)
        mcc_total += mcc
        num_evaluated += 1

        # Visualization: Ground Truth vs Prediction
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(ref_img, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title(f"Ground Truth {tid}")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap='gray', vmin=0, vmax=255)
        ax[1].set_title(f"Predicted {tid}\nMCC: {mcc:.4f}")
        ax[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{tid}.png"), dpi=200)
        plt.close(fig)

    if num_evaluated > 0:
        print(f"\n📊 Average MCC: {mcc_total / num_evaluated:.4f}")
    else:
        print("⚠️ No matching ground truth masks found for evaluation.")


if __name__ == "__main__":
    main()
