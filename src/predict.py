"""
predict.py
----------
Generates predictions on the validation set (no labels).
Run: py -3.11 predict.py
"""

import os, numpy as np, pandas as pd, torch, pickle
from model import BidirectionalLSTM, TemporalTransformer

DATA_DIR  = r"C:\Users\Yash Nagrale\Desktop\neuroguard\data\processed"
MODEL_DIR = r"C:\Users\Yash Nagrale\Desktop\neuroguard\models"
OUT_DIR   = r"C:\Users\Yash Nagrale\Desktop\neuroguard\outputs"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_batch(model, X_seq, X_static, batch_size=512):
    model.eval()
    probs = []
    for i in range(0, len(X_seq), batch_size):
        Xs  = torch.FloatTensor(X_seq[i:i+batch_size]).to(DEVICE)
        Xst = torch.FloatTensor(X_static[i:i+batch_size]).to(DEVICE)
        probs.extend(model(Xs, Xst).cpu().numpy())
    return np.array(probs)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading validation data...")
    X_seq    = np.load(os.path.join(DATA_DIR, "X_seq_val.npy"))
    X_static = np.load(os.path.join(DATA_DIR, "X_static_val.npy"))

    with open(os.path.join(DATA_DIR, "meta_val.pkl"), "rb") as f:
        meta_val = pickle.load(f)

    temporal_dim = X_seq.shape[2]
    static_dim   = X_static.shape[1]

    # Ensemble: average LSTM + Transformer if both exist
    all_probs = []
    for name, ModelClass, kwargs in [
        ("lstm",        BidirectionalLSTM,   {"temporal_input_size": temporal_dim, "static_input_size": static_dim}),
        ("transformer", TemporalTransformer, {"temporal_input_size": temporal_dim, "static_input_size": static_dim}),
    ]:
        path = os.path.join(MODEL_DIR, f"best_{name}.pt")
        if not os.path.exists(path):
            print(f"  Skipping {name} — not found")
            continue
        model = ModelClass(**kwargs)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        probs = predict_batch(model, X_seq, X_static)
        all_probs.append(probs)
        print(f"  {name}: mean risk = {probs.mean():.4f}")

    ensemble_probs = np.mean(all_probs, axis=0)

    # Load threshold (average across models)
    thresholds = []
    for name in ["lstm", "transformer"]:
        p = os.path.join(MODEL_DIR, f"threshold_{name}.txt")
        if os.path.exists(p):
            with open(p) as f:
                thresholds.append(float(f.read().strip()))
    threshold = np.mean(thresholds) if thresholds else 0.5
    print(f"\nUsing threshold: {threshold:.3f}")

    df = pd.DataFrame(meta_val)
    df["risk_score"]      = ensemble_probs.round(4)
    df["predicted_label"] = (ensemble_probs >= threshold).astype(int)
    df["risk_level"]      = pd.cut(ensemble_probs,
                                   bins=[0, 0.3, 0.6, 1.0],
                                   labels=["LOW","MODERATE","HIGH"])

    # Per-patient summary
    patient_summary = (df.groupby("patient_id")
                         .agg(max_risk=("risk_score","max"),
                              mean_risk=("risk_score","mean"),
                              predicted_deterioration=("predicted_label","max"))
                         .reset_index())

    df.to_csv(os.path.join(OUT_DIR, "predictions_all_windows.csv"), index=False)
    patient_summary.to_csv(os.path.join(OUT_DIR, "predictions_per_patient.csv"), index=False)

    print(f"\nPredictions saved to: {OUT_DIR}")
    print(f"  Total windows : {len(df)}")
    print(f"  Patients predicted to deteriorate: "
          f"{patient_summary['predicted_deterioration'].sum()} / {len(patient_summary)}")


if __name__ == "__main__":
    run()
