import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
import numpy as np
from loss import FocalDiceLoss
from model import UNetPP
from dataset import GlacierDataset
from tqdm import tqdm

# =====================
# Setup
# =====================
os.makedirs("../weights", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 2
LR = 5e-5  # for fine-tuning
PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "../data/Train"


# =====================
# Metrics MCC (Binary)
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
def train_one_loop(epoch, model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_mcc = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

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
            "Batch Loss": f"{loss.item():.4f}",
            "Batch MCC": f"{batch_mcc:.4f}"
        })

    avg_loss = total_loss / len(loader)
    avg_mcc = total_mcc / len(loader)
    return avg_loss, avg_mcc


# =====================
# Main
# =====================
def main():
    dataset = GlacierDataset(base_path=BASE_PATH, patch_size=128)
    model = UNetPP(in_channels=5, out_channels=4)
    model_path = "../weights/model_UNPP.pth"

    if os.path.exists(model_path):
        print(f"✅ Loaded pre-trained weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model = model.to(DEVICE)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = FocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )

    best_mcc = -1
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_mcc = train_one_loop(epoch, model, loader, optimizer, criterion, DEVICE)
        scheduler.step(train_mcc)

        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | MCC: {train_mcc:.4f}")

        # Save best model
        if train_mcc > best_mcc:
            best_mcc = train_mcc
            torch.save(model.state_dict(), "../solution/weights/model_finetuned_back.pth")
            print(f"✅ New best model saved (MCC={best_mcc:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print("⏹️ Early stopping triggered!")
            break


if __name__ == "__main__":
    main()
