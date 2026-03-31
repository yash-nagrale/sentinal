"""
ps1_train.py  (improved version — GPU + weighted loss + better split)
----------------------------------------------------------------------
Three improvements over v1:
1. Uses GPU automatically (RTX 3050 = 6-8x faster than CPU)
2. Weighted CrossEntropyLoss — penalises Grade 3 mistakes more heavily
3. Better val split — 1,490 val images instead of 282
Run: py -3.11 ps1_train.py
"""

import os, time, torch, torch.nn as nn, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ps1_dataset import get_dataloaders, LABEL_TO_GRADE
from ps1_model   import get_model

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 64        # larger batch = better GPU utilisation
LR         = 2e-4
PATIENCE   = 8
MODEL_SAVE = r"C:\Users\Yash Nagrale\Desktop\neuroguard\models\best_ps1.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  No GPU found — running on CPU (slower)")


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

        if (batch_idx + 1) % 30 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs        = model(images)
        loss           = criterion(outputs, labels)
        preds          = outputs.argmax(dim=1)

        total_loss += loss.item()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total * 100, all_preds, all_labels


def main():
    os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)

    print("\n" + "="*60)
    print("STEP 1: Loading dataset")
    print("="*60)
    # get_dataloaders now returns class_weights too
    train_loader, val_loader, class_weights = get_dataloaders(batch_size=BATCH_SIZE)

    print("\n" + "="*60)
    print("STEP 2: Building model")
    print("="*60)
    model = get_model().to(DEVICE)

    # Weighted loss — Grade 3 gets higher penalty for mistakes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "="*60)
    print("STEP 3: Training")
    print("="*60)
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Note':>10}")
    print("-" * 62)

    best_val_acc, patience_cnt = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc             = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc, preds, lbls  = validate(model, val_loader, criterion)
        scheduler.step()

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc}, MODEL_SAVE)

        secs = time.time() - t0
        note = f"* BEST ({secs:.0f}s)" if improved else f"({secs:.0f}s)"
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2f}% | "
              f"{val_loss:>8.4f} | {val_acc:>6.2f}% | {note}")

        if not improved:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
                break

    print("\n" + "="*60)
    print("STEP 4: Final evaluation")
    print("="*60)
    checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    _, _, final_preds, final_labels = validate(model, val_loader, criterion)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    grade_names = [LABEL_TO_GRADE[i] for i in range(4)]
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=grade_names))
    print("Confusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))
    print(f"\nModel saved to: {MODEL_SAVE}")


if __name__ == "__main__":
    main()
