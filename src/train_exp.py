import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import numpy as np
from dataset import GlacierDataset, GlacierDatasetOnline
from loss import FocalDiceLoss
from model import UNetPP
from model_2en import UNetPP_Multimodal
from tqdm import tqdm
import pickle
import os
from transform import GlacierAugment

# =====================
# Setup
# =====================
os.makedirs("../weights", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Hyperparameters
EPOCHS = 100  # Start with 100; expand later if needed
BATCH_SIZE = 2
PATCH_SIZE = 128
LR = 1e-3
DECAY = 1e-5
PATIENCE = 10
Early_Stopping = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "../data/Train"


# =====================
# Metrics MCC
# =====================
def mcc_score(y_true, y_pred):
    """Compute MCC after applying softmax + argmax."""
    probs = torch.softmax(y_pred, dim=1)
    preds = torch.argmax(probs, dim=1)

    preds_flat = preds.cpu().numpy().ravel()
    labels_flat = y_true.cpu().numpy().ravel()

    return matthews_corrcoef(labels_flat, preds_flat)


# =====================
# Training Loop
# =====================
def train_one_loop(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mcc = 0, 0
    pbar = tqdm(loader, desc='Training', leave=False)

    for bands, labels in pbar:
        bands, labels = bands.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(bands)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_mcc = mcc_score(labels, outputs)
        total_loss += loss.item()
        total_mcc += batch_mcc

        pbar.set_postfix({
            "Batch loss": f"{loss.item():.4f}",
            "Batch MCC": f"{batch_mcc:.4f}"
        })

    return total_loss / len(loader), total_mcc / len(loader)


# =====================
# Validation Loop
# =====================
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mcc = 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for bands, labels in pbar:
            bands, labels = bands.to(device), labels.to(device)
            outputs = model(bands)

            loss = criterion(outputs, labels)
            batch_mcc = mcc_score(labels, outputs)

            total_loss += loss.item()
            total_mcc += batch_mcc

            pbar.set_postfix({
                "Batch loss": f"{loss.item():.4f}",
                "Batch MCC": f"{batch_mcc:.4f}"
            })

    return total_loss / len(loader), total_mcc / len(loader)


# =====================
# Main Training
# =====================
def main():
    print("📂 Loading dataset...")
    dataset = GlacierDataset(base_path=BASE_PATH, patch_size=PATCH_SIZE)

    # Simple 80/20 split
    all_img_ids = list(set(dataset.get_image_ids()))
    train_img_ids, val_img_ids = train_test_split(all_img_ids, test_size=0.2, random_state=seed, shuffle=True)
    print(val_img_ids)

    # now map patch indices based on which image they belong to
    train_idx = [i for i, s in enumerate(dataset.samples) if s[0] in train_img_ids]
    val_idx = [i for i, s in enumerate(dataset.samples) if s[0] in val_img_ids]

    print(f"✅ Dataset split: {len(train_idx)} train | {len(val_idx)} val")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_dataset = GlacierDatasetOnline(train_subset, GlacierAugment.get_train_augmentations())
    val_dataset = GlacierDatasetOnline(val_subset, GlacierAugment.get_val_augmentations())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)

    model = UNetPP_Multimodal(5, 4)
    model = model.to(DEVICE)
    criterion = FocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_mcc = -1
    patience_counter = 0
    best_model_path = None

    train_losses, val_losses, val_mccs = [], [], []

    print("\n🚀 Starting training...\n")

    for epoch in range(EPOCHS):
        train_loss, train_mcc = train_one_loop(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_mcc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.4f} | Train MCC: {train_mcc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MCC: {val_mcc:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mccs.append(val_mcc)

        # Save model if MCC improves
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0

            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_model_path = f"../weights/best_mcc_{best_mcc:.4f}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved with MCC {best_mcc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if Early_Stopping:
            if patience_counter >= PATIENCE:
                print(f"⏹ Early stopping at epoch {epoch + 1} — no MCC improvement for {PATIENCE} epochs.")
                break

    # Save metrics
    with open(f"../outputs/metrics_backup.pkl", "wb") as f:
        pickle.dump({
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_mcc": val_mccs
        }, f)

    print("\n🏁 Training complete.")
    print(f"📈 Best validation MCC: {best_mcc:.4f}")

    torch.cuda.empty_cache()
    del model, optimizer, criterion, train_loader, val_loader


if __name__ == "__main__":
    main()
