"""
train.py
--------
Trains the deterioration prediction model.

Run:
  py -3.11 train.py --model transformer    (recommended)
  py -3.11 train.py --model lstm           (faster, slightly lower accuracy)
"""

import os, argparse, numpy as np, torch, torch.nn as nn, pickle
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from model import BidirectionalLSTM, TemporalTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = r"C:\Users\Yash Nagrale\Desktop\neuroguard\data\processed"
MODEL_DIR = r"C:\Users\Yash Nagrale\Desktop\neuroguard\models"

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 60
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
PATIENCE     = 8
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Focal Loss — handles 94:6 class imbalance ─────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        bce          = nn.functional.binary_cross_entropy(probs, targets, reduction="none")
        p_t          = probs * targets + (1 - probs) * (1 - targets)
        alpha_t      = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def load_data():
    X_seq    = np.load(os.path.join(DATA_DIR, "X_seq_train.npy"))
    X_static = np.load(os.path.join(DATA_DIR, "X_static_train.npy"))
    y        = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    print(f"Loaded: X_seq={X_seq.shape} | Positive rate: {y.mean()*100:.2f}%")

    idx = np.arange(len(y))
    tr_idx, vl_idx = train_test_split(idx, test_size=VAL_SPLIT, stratify=y, random_state=SEED)

    return (X_seq[tr_idx], X_static[tr_idx], y[tr_idx],
            X_seq[vl_idx], X_static[vl_idx], y[vl_idx])


def make_loader(X_seq, X_static, y, use_sampler=False, shuffle=True):
    ds = TensorDataset(
        torch.FloatTensor(X_seq),
        torch.FloatTensor(X_static),
        torch.FloatTensor(y),
    )
    sampler = None
    if use_sampler:
        counts  = np.bincount(y.astype(int))
        weights = 1.0 / counts[y.astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      sampler=sampler, num_workers=0)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_seq, X_stat, y_batch in loader:
        X_seq, X_stat, y_batch = X_seq.to(DEVICE), X_stat.to(DEVICE), y_batch.to(DEVICE)
        preds = model(X_seq, X_stat)
        loss  = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for X_seq, X_stat, y_batch in loader:
        probs = model(X_seq.to(DEVICE), X_stat.to(DEVICE)).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.numpy())
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    return roc_auc_score(labels, probs), average_precision_score(labels, probs), probs, labels


def train_model(model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "="*60)
    print(f"Training: {model_name.upper()}")
    print("="*60)

    X_seq_tr, X_stat_tr, y_tr, X_seq_vl, X_stat_vl, y_vl = load_data()

    temporal_dim = X_seq_tr.shape[2]
    static_dim   = X_stat_tr.shape[1]
    print(f"Temporal features: {temporal_dim} | Static features: {static_dim}")

    if model_name == "lstm":
        model = BidirectionalLSTM(temporal_dim, static_dim)
    else:
        model = TemporalTransformer(temporal_dim, static_dim)

    model = model.to(DEVICE)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    tr_loader = make_loader(X_seq_tr, X_stat_tr, y_tr, use_sampler=True)
    vl_loader = make_loader(X_seq_vl, X_stat_vl, y_vl, shuffle=False)

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    save_path    = os.path.join(MODEL_DIR, f"best_{model_name}.pt")
    best_auroc   = 0.0
    patience_cnt = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val AUROC':>9} | {'Val AUPRC':>9} | {'Best':>6}")
    print("-" * 48)

    for epoch in range(1, EPOCHS + 1):
        train_loss             = train_epoch(model, tr_loader, optimizer, criterion)
        val_auroc, val_auprc, _, _ = evaluate(model, vl_loader)
        scheduler.step()

        improved = val_auroc > best_auroc
        if improved:
            best_auroc   = val_auroc
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_auroc:>9.4f} | "
              f"{val_auprc:>9.4f} | {'* BEST' if improved else ''}")

        if not improved:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best AUROC: {best_auroc:.4f}")
                break

    # Save threshold
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    _, _, probs, labels = evaluate(model, vl_loader)
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.05, 0.95, 200):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    with open(os.path.join(MODEL_DIR, f"threshold_{model_name}.txt"), "w") as f:
        f.write(str(best_t))

    print(f"\nBest AUROC : {best_auroc:.4f}")
    print(f"Best threshold (F1): {best_t:.3f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm","transformer"], default="transformer")
    args = parser.parse_args()
    train_model(args.model)
