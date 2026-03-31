"""
evaluate.py
-----------
Loads the best saved model and runs full evaluation.
Run: py -3.11 evaluate.py --model transformer
"""

import os, argparse, numpy as np, torch, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                              precision_recall_curve, confusion_matrix,
                              classification_report, f1_score)
from sklearn.model_selection import train_test_split
from model import BidirectionalLSTM, TemporalTransformer

DATA_DIR  = r"C:\Users\Yash Nagrale\Desktop\neuroguard\data\processed"
MODEL_DIR = r"C:\Users\Yash Nagrale\Desktop\neuroguard\models"
OUT_DIR   = r"C:\Users\Yash Nagrale\Desktop\neuroguard\outputs"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_val_data():
    X_seq    = np.load(os.path.join(DATA_DIR, "X_seq_train.npy"))
    X_static = np.load(os.path.join(DATA_DIR, "X_static_train.npy"))
    y        = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    idx      = np.arange(len(y))
    _, vl_idx = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
    return X_seq[vl_idx], X_static[vl_idx], y[vl_idx]


@torch.no_grad()
def get_probs(model, X_seq, X_static, batch_size=512):
    model.eval()
    probs = []
    for i in range(0, len(X_seq), batch_size):
        Xs  = torch.FloatTensor(X_seq[i:i+batch_size]).to(DEVICE)
        Xst = torch.FloatTensor(X_static[i:i+batch_size]).to(DEVICE)
        probs.extend(model(Xs, Xst).cpu().numpy())
    return np.array(probs)


def evaluate(model_name):
    os.makedirs(OUT_DIR, exist_ok=True)

    X_seq, X_static, y = load_val_data()
    temporal_dim        = X_seq.shape[2]
    static_dim          = X_static.shape[1]

    if model_name == "lstm":
        model = BidirectionalLSTM(temporal_dim, static_dim)
    else:
        model = TemporalTransformer(temporal_dim, static_dim)

    save_path = os.path.join(MODEL_DIR, f"best_{model_name}.pt")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)

    y_prob = get_probs(model, X_seq, X_static)

    auroc = roc_auc_score(y, y_prob)
    auprc = average_precision_score(y, y_prob)

    with open(os.path.join(MODEL_DIR, f"threshold_{model_name}.txt")) as f:
        threshold = float(f.read().strip())

    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"Model          : {model_name.upper()}")
    print(f"AUROC          : {auroc:.4f}")
    print(f"AUPRC          : {auprc:.4f}")
    print(f"Threshold      : {threshold:.3f}")
    print(f"\n{classification_report(y, y_pred, target_names=['Stable','Deteriorating'])}")
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Sensitivity    : {tp/(tp+fn):.4f}")
    print(f"Specificity    : {tn/(tn+fp):.4f}")
    print(f"{'='*50}")

    # Plot ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y, y_prob)
    axes[0].plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUROC={auroc:.3f}")
    axes[0].plot([0,1],[0,1],"k--",lw=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y, y_prob)
    axes[1].plot(rec, prec, color="#1D9E75", lw=2, label=f"AUPRC={auprc:.3f}")
    axes[1].axhline(y.mean(), color="gray", linestyle="--", lw=0.8, label="Baseline")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"evaluation_{model_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm","transformer"], default="transformer")
    args = parser.parse_args()
    evaluate(args.model)
