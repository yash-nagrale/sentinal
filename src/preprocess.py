"""
preprocess.py
-------------
Loads train.csv and val_no_labels.csv, assigns patient IDs,
engineers features, creates sliding-window sequences, and saves
ready-to-train numpy arrays.

Run: py -3.11 preprocess.py
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_CSV = r"C:\Users\Yash Nagrale\Desktop\neuroguard\data\train.csv"
VAL_CSV   = r"C:\Users\Yash Nagrale\Desktop\neuroguard\data\val_no_labels.csv"
OUT_DIR   = r"C:\Users\Yash Nagrale\Desktop\neuroguard\data\processed"

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW = 12    # hours of history the model sees at once
STEP   = 1     # slide window by 1 hour each time

VITAL_COLS = [
    "heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
    "systolic_bp", "diastolic_bp", "oxygen_flow", "mobility_score",
    "nurse_alert", "wbc_count", "lactate", "creatinine",
    "crp_level", "hemoglobin", "sepsis_risk_score"
]
STATIC_COLS = ["age", "comorbidity_index"]
TARGET_COL  = "deterioration_next_12h"


def assign_patient_ids(df):
    """Detect patient boundaries by finding where hour_from_admission resets."""
    df = df.copy()
    df["patient_id"] = (df["hour_from_admission"].diff() < 0).cumsum()
    return df


def engineer_features(df):
    """Add clinically meaningful derived features."""
    df = df.copy()
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"]            = df["diastolic_bp"] + df["pulse_pressure"] / 3
    df["shock_index"]    = df["heart_rate"] / (df["systolic_bp"] + 1e-6)
    df["spo2_below_94"]  = (df["spo2_pct"] < 94).astype(int)
    df["tachycardia"]    = (df["heart_rate"] > 100).astype(int)
    df["tachypnea"]      = (df["respiratory_rate"] > 20).astype(int)
    df["high_lactate"]   = (df["lactate"] > 2.0).astype(int)
    df["crp_high"]       = (df["crp_level"] > 50).astype(int)
    df["qsofa"] = (
        (df["respiratory_rate"] >= 22).astype(int) +
        (df["systolic_bp"] <= 100).astype(int) +
        df["nurse_alert"]
    )
    for col in ["heart_rate", "spo2_pct", "respiratory_rate", "systolic_bp"]:
        df[f"{col}_trend4"] = (
            df.groupby("patient_id")[col].transform(lambda x: x.diff(4))
        )
    return df


def encode_categoricals(df, encoders=None):
    df = df.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    oxygen_order = {"none": 0, "nasal": 1, "mask": 2, "hfnc": 3, "niv": 4}
    df["oxygen_device_enc"] = df["oxygen_device"].map(oxygen_order).fillna(0)

    for col in ["gender", "admission_type"]:
        if fit_mode:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col]    = le
        else:
            df[f"{col}_enc"] = encoders[col].transform(df[col].astype(str))

    return df, encoders


def build_windows(df, feature_cols, window, step, has_labels):
    X_seq, X_static, y_out, meta = [], [], [], []
    static_feats = STATIC_COLS + ["gender_enc", "admission_type_enc"]

    for pid, grp in df.groupby("patient_id", sort=False):
        grp = grp.reset_index(drop=True)
        n   = len(grp)
        if n < window:
            continue

        static_vec = grp[static_feats].iloc[0].values.astype(np.float32)

        for i in range(0, n - window + 1, step):
            seq = grp[feature_cols].iloc[i:i+window].values.astype(np.float32)
            X_seq.append(seq)
            X_static.append(static_vec)
            meta.append({
                "patient_id":      pid,
                "window_end_hour": grp["hour_from_admission"].iloc[i+window-1]
            })
            if has_labels:
                y_out.append(int(grp[TARGET_COL].iloc[i+window-1]))

    X_seq    = np.array(X_seq,    dtype=np.float32)
    X_static = np.array(X_static, dtype=np.float32)
    y_out    = np.array(y_out,    dtype=np.float32) if has_labels else None
    return X_seq, X_static, y_out, meta


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)

    print("Assigning patient IDs...")
    train = assign_patient_ids(train)
    val   = assign_patient_ids(val)
    print(f"  Train patients: {train['patient_id'].nunique()}")
    print(f"  Val   patients: {val['patient_id'].nunique()}")

    print("Engineering features...")
    train = engineer_features(train)
    val   = engineer_features(val)

    print("Encoding categoricals...")
    train, encoders = encode_categoricals(train)
    val,   _        = encode_categoricals(val, encoders=encoders)

    trend_cols     = [f"{c}_trend4" for c in ["heart_rate","spo2_pct","respiratory_rate","systolic_bp"]]
    engineered     = ["pulse_pressure","map","shock_index","spo2_below_94",
                      "tachycardia","tachypnea","high_lactate","crp_high","qsofa"]
    feature_cols   = VITAL_COLS + engineered + trend_cols + ["oxygen_device_enc","hour_from_admission"]

    for col in trend_cols:
        train[col] = train.groupby("patient_id")[col].transform(lambda x: x.fillna(0))
        val[col]   = val.groupby("patient_id")[col].transform(lambda x: x.fillna(0))

    print("Fitting scaler...")
    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols]   = scaler.transform(val[feature_cols])

    print("Building sliding windows...")
    X_seq_tr, X_stat_tr, y_tr, _        = build_windows(train, feature_cols, WINDOW, STEP, True)
    X_seq_vl, X_stat_vl, _,   meta_val  = build_windows(val,   feature_cols, WINDOW, STEP, False)

    print(f"  Train windows : {X_seq_tr.shape} | Positive rate: {y_tr.mean()*100:.2f}%")
    print(f"  Val   windows : {X_seq_vl.shape}")

    print("Saving processed data...")
    np.save(os.path.join(OUT_DIR, "X_seq_train.npy"),    X_seq_tr)
    np.save(os.path.join(OUT_DIR, "X_static_train.npy"), X_stat_tr)
    np.save(os.path.join(OUT_DIR, "y_train.npy"),        y_tr)
    np.save(os.path.join(OUT_DIR, "X_seq_val.npy"),      X_seq_vl)
    np.save(os.path.join(OUT_DIR, "X_static_val.npy"),   X_stat_vl)

    with open(os.path.join(OUT_DIR, "scaler.pkl"),       "wb") as f: pickle.dump(scaler,       f)
    with open(os.path.join(OUT_DIR, "encoders.pkl"),     "wb") as f: pickle.dump(encoders,     f)
    with open(os.path.join(OUT_DIR, "meta_val.pkl"),     "wb") as f: pickle.dump(meta_val,     f)
    with open(os.path.join(OUT_DIR, "feature_cols.pkl"), "wb") as f: pickle.dump(feature_cols, f)

    print(f"\nAll processed files saved to: {OUT_DIR}")
    print("Done! Now run: py -3.11 train.py --model transformer")


if __name__ == "__main__":
    run()
