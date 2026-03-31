"""
ps5_train.py
------------
Trains the CT stroke classifier.
Run: py -3.11 ps5_train.py
"""

import os, time, torch, torch.nn as nn, numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ps5_dataset import get_dataloaders, LABEL_TO_CLASS
from ps5_model   import get_model

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 25
BATCH_SIZE = 32
LR         = 2e-4
PATIENCE   = 7
MODEL_SAVE = r"C:\Users\Yash Nagrale\Desktop\neuroguard\models\best_ps5_classifier.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        preds   = outputs.argmax(dim=1)
        probs   = torch.softmax(outputs, dim=1)[:, 1]

        total_loss += loss.item()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    acc   = correct / total * 100
    auroc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), acc, auroc, all_preds, all_labels, all_probs


def main():
    os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)

    print("\n" + "="*60)
    print("STEP 1: Loading dataset")
    print("="*60)
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    print("\n" + "="*60)
    print("STEP 2: Building model")
    print("="*60)
    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "="*60)
    print("STEP 3: Training")
    print("="*60)
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Acc':>7} | {'AUROC':>6} | {'Note':>12}")
    print("-" * 60)

    best_auroc, patience_cnt = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, auroc, preds, lbls, probs = validate(model, val_loader, criterion)
        scheduler.step()

        improved = auroc > best_auroc
        if improved:
            best_auroc   = auroc
            patience_cnt = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc, "auroc": auroc}, MODEL_SAVE)

        note = f"* BEST ({time.time()-t0:.0f}s)" if improved else f"({time.time()-t0:.0f}s)"
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2f}% | "
              f"{val_acc:>6.2f}% | {auroc:>5.3f} | {note}")

        if not improved:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
                break

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Final evaluation")
    print("="*60)

    checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _, _, _, final_preds, final_labels, final_probs = validate(model, val_loader, criterion)

    auroc = roc_auc_score(final_labels, final_probs)
    auprc = average_precision_score(final_labels, final_probs)

    print(f"\nBest AUROC          : {best_auroc:.4f}")
    print(f"Final AUROC         : {auroc:.4f}")
    print(f"Average Precision   : {auprc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(final_labels, final_preds,
                                target_names=["Normal", "Stroke"]))
    print("Confusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))
    print(f"\nModel saved to: {MODEL_SAVE}")
    print("\nTRAINING COMPLETE!")


if __name__ == "__main__":
    main()
