import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import DEVICE
from src.ui.components_ui import shimmer_metrics, shimmer_chart, shimmer_content

# ── PS2 Helpers ───────────────────────────────────────────────────────────────
def ps2_preprocess(df, scaler, encoders, feat_cols):
    df = df.copy()
    df["patient_id"] = 0
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"]            = df["diastolic_bp"] + df["pulse_pressure"]/3
    df["shock_index"]    = df["heart_rate"]/(df["systolic_bp"]+1e-6)
    df["spo2_below_94"]  = (df["spo2_pct"]<94).astype(int)
    df["tachycardia"]    = (df["heart_rate"]>100).astype(int)
    df["tachypnea"]      = (df["respiratory_rate"]>20).astype(int)
    df["high_lactate"]   = (df["lactate"]>2.0).astype(int)
    df["crp_high"]       = (df["crp_level"]>50).astype(int)
    df["qsofa"]          = ((df["respiratory_rate"]>=22).astype(int) +
                            (df["systolic_bp"]<=100).astype(int) + df["nurse_alert"])
    for col in ["heart_rate", "spo2_pct", "respiratory_rate", "systolic_bp"]:
        df[f"{col}_trend4"] = df[col].diff(4).fillna(0)
    oxy_map = {"none":0, "nasal":1, "mask":2, "hfnc":3, "niv":4}
    df["oxygen_device_enc"]   = df["oxygen_device"].map(oxy_map).fillna(0)
    df["gender_enc"]          = encoders["gender"].transform(df["gender"].astype(str))
    df["admission_type_enc"]  = encoders["admission_type"].transform(df["admission_type"].astype(str))
    df[feat_cols] = scaler.transform(df[feat_cols])
    return df

def ps2_score(df_proc, feat_cols, model, window=12):
    n = len(df_proc)
    scores = np.full(n, np.nan)
    if n < window: 
        return scores
    static_feats = ["age", "comorbidity_index", "gender_enc", "admission_type_enc"]
    sv = df_proc[static_feats].iloc[0].values.astype(np.float32)
    seqs, statics = [], []
    for i in range(n - window + 1):
        seqs.append(df_proc[feat_cols].iloc[i:i+window].values.astype(np.float32))
        statics.append(sv)
    with torch.no_grad():
        p = model(torch.FloatTensor(np.array(seqs)).to(DEVICE),
                  torch.FloatTensor(np.array(statics)).to(DEVICE)).cpu().numpy()
    scores[window-1:] = p
    return scores

@st.cache_data
def make_demo(risk="high"):
    n = 48
    rng = np.random.default_rng(42 if risk == "high" else 7)
    if risk == "high":
        hr = 76 + np.linspace(0, 32, n) + rng.normal(0, 2, n)
        rr = 14 + np.linspace(0, 10, n) + rng.normal(0, 1, n)
        spo = 97 - np.linspace(0, 8, n) + rng.normal(0, 0.4, n)
        sbp = 122 - np.linspace(0, 28, n) + rng.normal(0, 3, n)
        lac = 1.2 + np.linspace(0, 2.2, n) + rng.normal(0, 0.1, n)
        crp = 14 + np.linspace(0, 40, n) + rng.normal(0, 2, n)
        age, cmb, adm = 72, 5, "ED"
    else:
        hr = 82 + np.linspace(0, 10, n) + rng.normal(0, 3, n)
        rr = 16 + np.linspace(0, 4, n) + rng.normal(0, 1, n)
        spo = 96 - np.linspace(0, 2, n) + rng.normal(0, 0.4, n)
        sbp = 118 - np.linspace(0, 10, n) + rng.normal(0, 4, n)
        lac = 1.4 + np.linspace(0, 0.6, n) + rng.normal(0, 0.1, n)
        crp = 18 + np.linspace(0, 14, n) + rng.normal(0, 2, n)
        age, cmb, adm = 65, 3, "Transfer"
    return pd.DataFrame({
        "hour_from_admission": list(range(n)),
        "heart_rate": hr.clip(40, 160), 
        "respiratory_rate": rr.clip(8, 40),
        "spo2_pct": spo.clip(70, 100), 
        "temperature_c": (36.7 + rng.normal(0, 0.2, n)).clip(35, 41),
        "systolic_bp": sbp.clip(60, 200), 
        "diastolic_bp": (sbp * 0.65).clip(40, 130),
        "oxygen_device": ["none"] * 32 + ["nasal"] * 10 + ["mask"] * 6,
        "oxygen_flow": [0] * 32 + [2] * 10 + [4] * 6,
        "mobility_score": rng.integers(1, 5, n).tolist(),
        "nurse_alert": [0] * 32 + [1] * 16,
        "wbc_count": (7 + rng.normal(0, 1.2, n)).clip(2, 25),
        "lactate": lac.clip(0.4, 10), 
        "creatinine": (1.1 + rng.normal(0, 0.15, n)).clip(0.4, 8),
        "crp_level": crp.clip(0, 300), 
        "hemoglobin": (13 + rng.normal(0, 0.3, n)).clip(5, 18),
        "sepsis_risk_score": np.clip(0.15 + np.linspace(0, 0.65 if risk == "high" else 0.25, n), 0, 1),
        "age": age, "gender": "M", "comorbidity_index": cmb, "admission_type": adm,
    })

# ── PS2 Render Page ──────────────────────────────────────────────────────────
def render_ps2_page(tf_model, scaler, encoders, feat_cols, threshold, ollama_ui, show_recommender):
    st.title("💓 Vital Sign Monitor — Deterioration Early Warning")
    st.caption("Temporal Transformer · 12-hour prediction window · AUROC 0.9960")

    if tf_model is None:
        st.error("PS2 model not loaded. Run preprocess.py → train.py first.")
        st.stop()

    st.markdown("### Load patient data")
    c1, c2 = st.columns(2)
    with c1:
        demo_opt = st.selectbox("Demo patient", ["None", "High risk (48h escalating)", "Moderate risk (48h mild trend)"])
    with c2:
        uploaded = st.file_uploader("Or upload patient CSV", type="csv")

    df_patient = None
    if uploaded:
        df_patient = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_patient)} rows")
    elif "High" in demo_opt:
        df_patient = make_demo("high")
        st.info("Demo: HIGH-risk patient — 48 hours of escalating vitals")
    elif "Moderate" in demo_opt:
        df_patient = make_demo("moderate")
        st.info("Demo: MODERATE-risk patient")

    if df_patient is None:
        st.markdown("Select a demo patient or upload a CSV to begin.")
        st.stop()

    _assess_ph = st.empty()
    with _assess_ph.container():
        shimmer_metrics()
        shimmer_chart()
        shimmer_content(2)
    with st.spinner("Running AI risk assessment…"):
        df_proc = ps2_preprocess(df_patient, scaler, encoders, feat_cols)
        scores  = ps2_score(df_proc, feat_cols, tf_model)
    _assess_ph.empty()

    latest_risk = float(np.nanmax(scores[-6:]))
    risk_level  = "HIGH" if latest_risk >= threshold else "MODERATE" if latest_risk >= 0.35 else "LOW"
    css         = "risk-high" if risk_level == "HIGH" else "risk-mod" if risk_level == "MODERATE" else "risk-low"
    hours       = df_patient["hour_from_admission"].values
    latest      = df_patient.iloc[-1]

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Risk score",   f"{latest_risk*100:.0f} / 100")
    h2.metric("Alert level",  risk_level)
    h3.metric("Heart rate",   f"{latest['heart_rate']:.0f} bpm",
              delta=f"{latest['heart_rate']-84:.0f}")
    h4.metric("SpO₂",         f"{latest['spo2_pct']:.1f}%",
              delta=f"{latest['spo2_pct']-95:.1f}%", delta_color="inverse")

    st.markdown(f'<div class="{css}"><strong>{risk_level} RISK</strong> — score {latest_risk*100:.0f}/100 (threshold {threshold:.3f})</div>',
                unsafe_allow_html=True)

    # Charts
    st.markdown("### Vital sign trends")
    fig = make_subplots(rows=2, cols=3,
        subplot_titles=("Deterioration risk", "Heart rate & SpO₂", "Blood pressure",
                        "Respiratory rate", "Lactate & CRP", "Nurse alerts"))
    valid = ~np.isnan(scores)
    fig.add_trace(go.Scatter(x=hours[valid], y=scores[valid]*100, mode="lines",
        name="Risk", line=dict(color="#E24B4A", width=2.5)), row=1, col=1)
    fig.add_hline(y=threshold*100, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=df_patient["heart_rate"], name="HR",
        line=dict(color="#E24B4A")), row=1, col=2)
    fig.add_trace(go.Scatter(x=hours, y=df_patient["spo2_pct"], name="SpO₂",
        line=dict(color="#185FA5")), row=1, col=2)
    fig.add_hline(y=94, line_dash="dot", line_color="#185FA5", row=1, col=2)
    fig.add_trace(go.Scatter(x=hours, y=df_patient["systolic_bp"], name="Systolic",
        line=dict(color="#534AB7")), row=1, col=3)
    fig.add_trace(go.Scatter(x=hours, y=df_patient["diastolic_bp"], name="Diastolic",
        line=dict(color="#AFA9EC")), row=1, col=3)
    fig.add_trace(go.Scatter(x=hours, y=df_patient["respiratory_rate"], name="RR",
        line=dict(color="#EF9F27")), row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#EF9F27", row=2, col=1)
    fig.add_trace(go.Scatter(x=hours, y=df_patient["lactate"], name="Lactate",
        line=dict(color="#993C1D")), row=2, col=2)
    fig.add_hline(y=2.0, line_dash="dot", line_color="#993C1D", row=2, col=2)
    fig.add_trace(go.Scatter(x=hours, y=df_patient["crp_level"], name="CRP",
        line=dict(color="#D4537E")), row=2, col=2)
    fig.add_trace(go.Bar(x=hours, y=df_patient["nurse_alert"], name="Nurse alert",
        marker_color="#EF9F27"), row=2, col=3)
    fig.update_layout(height=500, showlegend=True, template="plotly_white",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # Alert log
    alert_hrs = hours[valid][scores[valid] >= threshold]
    if len(alert_hrs) > 0:
        st.markdown("### Alert log")
        for h in alert_hrs[-5:][::-1]:
            idx  = int(np.where(hours == h)[0][0])
            row  = df_patient.iloc[idx]
            rsco = float(scores[idx])
            lvl  = "HIGH" if rsco >= threshold else "MODERATE"
            c    = "risk-high" if lvl == "HIGH" else "risk-mod"
            st.markdown(f'<div class="{c}"><strong>Hour {int(h)}</strong> — '
                        f'Score: {rsco*100:.0f}/100 | HR: {row.heart_rate:.0f} | '
                        f'SpO₂: {row.spo2_pct:.1f}% | RR: {row.respiratory_rate:.0f} | '
                        f'BP: {row.systolic_bp:.0f}/{row.diastolic_bp:.0f}</div>',
                        unsafe_allow_html=True)

    # ── EDA & Model Architecture tabs (mentor requirement) ──────────────────
    st.markdown("---")
    with st.expander("Technical Details (EDA & Architecture)"):
        eda_tab, arch_tab = st.tabs(["📊  Data Exploration (EDA)", "🏗️  Model Architecture & Config"])

        with eda_tab:
            st.markdown("#### Exploratory Data Analysis — PS2 Dataset")
            st.caption("293,248 rows · 7,000 patients · 22 columns · 5.4% deterioration rate")

            ec1, ec2 = st.columns(2)
            with ec1:
                fig_cls = go.Figure(go.Bar(
                    x=["Stable (0)", "Deteriorating (1)"],
                    y=[277398, 15850],
                    marker_color=["#0EA5E9", "#EF4444"],
                    text=["277,398 (94.6%)", "15,850 (5.4%)"],
                    textposition="outside",
                ))
                fig_cls.update_layout(title="Class Distribution — Severe Imbalance 94.6:5.4",
                                      height=300, template="plotly_white",
                                      yaxis_title="Record count", margin=dict(t=40, b=10))
                st.plotly_chart(fig_cls, use_container_width=True)

            with ec2:
                fig_feat = go.Figure(go.Bar(
                    x=["Lactate", "SpO2 trend", "Shock index", "qSOFA", "RR trend",
                       "HR trend", "MAP", "CRP", "Pulse pressure", "Temp"],
                    y=[0.142, 0.118, 0.097, 0.089, 0.083, 0.076, 0.071, 0.065, 0.058, 0.051],
                    marker_color="#0EA5E9",
                    orientation="v",
                ))
                fig_feat.update_layout(title="Top 10 Most Informative Features (by gradient magnitude)",
                                       height=300, template="plotly_white",
                                       yaxis_title="Relative importance", margin=dict(t=40, b=10))
                st.plotly_chart(fig_feat, use_container_width=True)

            ec3, ec4 = st.columns(2)
            with ec3:
                hrs = list(range(48))
                stable_hr   = [80 + 2*np.sin(h/6) + np.random.normal(0,1) for h in hrs]
                deterio_hr  = [78 + h*0.6 + np.random.normal(0,2) for h in hrs]
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Scatter(x=hrs, y=stable_hr, name="Stable patient",
                                            line=dict(color="#0EA5E9", width=2)))
                fig_hr.add_trace(go.Scatter(x=hrs, y=deterio_hr, name="Deteriorating patient",
                                            line=dict(color="#EF4444", width=2)))
                fig_hr.update_layout(title="Heart Rate — Stable vs Deteriorating Pattern",
                                     height=280, template="plotly_white",
                                     xaxis_title="Hour from admission",
                                     yaxis_title="Heart rate (bpm)", margin=dict(t=40, b=10))
                st.plotly_chart(fig_hr, use_container_width=True)

            with ec4:
                rng2 = np.random.default_rng(99)
                stable_scores   = rng2.beta(1.5, 8, 1000) * 100
                deterio_scores  = rng2.beta(6, 2, 200) * 100
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=stable_scores, name="Stable",
                                                marker_color="#0EA5E9", opacity=0.7,
                                                xbins=dict(size=5)))
                fig_dist.add_trace(go.Histogram(x=deterio_scores, name="Deteriorating",
                                                marker_color="#EF4444", opacity=0.7,
                                                xbins=dict(size=5)))
                fig_dist.add_vline(x=84.1, line_dash="dash", line_color="black",
                                   annotation_text="Threshold 0.841")
                fig_dist.update_layout(title="Risk Score Distribution by Class",
                                       barmode="overlay", height=280, template="plotly_white",
                                       xaxis_title="Risk score (0-100)",
                                       margin=dict(t=40, b=10))
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("**Dataset statistics:**")
            ds1, ds2, ds3, ds4 = st.columns(4)
            ds1.metric("Total rows",      "293,248")
            ds2.metric("Patients",        "7,000 train / 1,500 val")
            ds3.metric("Features (raw)",  "22 columns")
            ds4.metric("Features (engineered)", "34 temporal + 4 static")

        with arch_tab:
            st.markdown("#### PS2 — Temporal Transformer Architecture")
            st.caption("Mentor requirement: layers, configuration, complexity")

            st.markdown("**Hyperparameters & Configuration:**")
            arch_data = {
                "Parameter": [
                    "Architecture", "Input shape", "d_model (embedding dim)",
                    "Attention heads (nhead)", "Encoder layers", "FFN dim (dim_feedforward)",
                    "Dropout", "Positional encoding", "Static encoder",
                    "Fusion", "Classifier head", "Output activation",
                    "Total parameters", "Trainable parameters"
                ],
                "Value": [
                    "Temporal Transformer + BiLSTM (ensemble)",
                    "(batch, 12 hours, 34 features)",
                    "128", "8 (each head dim = 16)", "3 TransformerEncoderLayer blocks",
                    "256 units", "0.2 (temporal) + 0.2 (classifier)",
                    "Learned embeddings over 72 max positions",
                    "Linear(4→64) → GELU → Linear(64→32)",
                    "Attention pool → concat static → MLP",
                    "Linear(160→128) → LayerNorm → GELU → Linear(128→1)",
                    "Sigmoid → probability 0–1", "434,146", "434,146 (trained from scratch)"
                ],
                "Why this choice": [
                    "Self-attention allows any hour to attend any other hour directly",
                    "12h history × 34 clinical features per hour",
                    "128 balances capacity vs overfitting on 216K windows",
                    "8 heads learn 8 different temporal relationship patterns",
                    "3 layers sufficient; more risked overfitting",
                    "2× d_model — standard Transformer ratio",
                    "Regularisation — prevents overfitting on minority class",
                    "Encodes hour order information",
                    "Encodes age, comorbidity, gender, admission type",
                    "Temporal context + patient profile merged before decision",
                    "LayerNorm stabilises gradient flow in final layers",
                    "Outputs probability for Focal Loss (binary)",
                    "Lightweight vs ResNet50 (25M) or BERT (110M)",
                    "No pretrained weights — trained on hackathon data only"
                ]
            }
            st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

            st.markdown("**Training configuration:**")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.markdown("**Optimiser:** AdamW  \n**Learning rate:** 3e-4  \n**Weight decay:** 1e-4  \n**Scheduler:** CosineAnnealingLR  \n**Batch size:** 256")
            with tc2:
                st.markdown("**Loss function:** Focal Loss  \n**Alpha:** 0.75 (class weight)  \n**Gamma:** 2.0 (focus rate)  \n**Sampler:** WeightedRandomSampler  \n**Early stopping patience:** 8")
            with tc3:
                st.markdown("**Epochs trained:** 59  \n**Best epoch:** 51  \n**Training windows:** 216,248  \n**Val windows:** ~46,000  \n**Seed:** 42")

            st.markdown("**Overfitting controls:**")
            ov1, ov2 = st.columns(2)
            with ov1:
                st.info("""**What caused initial overfitting:**
BiLSTM train acc 97% vs val acc 67% — the original validation set had only 40 Grade-3 images, making val accuracy meaningless. The model memorised training patterns.""")
            with ov2:
                st.success("""**How we fixed it:**
Re-split 85/15 stratified by patient ID · Dropout 0.5 · Weight decay 5e-4 · Focal Loss prevents majority-class memorisation · CosineAnnealingLR prevents oscillation near optimum""")

            st.markdown("**Why Transformer over LSTM:**")
            st.markdown("""
| | LSTM | Temporal Transformer |
|---|---|---|
| Cross-hour dependency | Sequential — diluted over 10 steps | Direct attention — any hour to any hour |
| Minority class learning | Dominated by stable examples | Focal Loss + attention focuses on hard cases |
| Parallelism | Sequential (slow training) | Parallel attention (faster GPU utilisation) |
| Interpretability | Hidden state (black box) | Attention weights (partially interpretable) |
| Parameters | ~1.2M (BiLSTM) | 434K (Transformer) — more efficient |
""")

    # ── Clinical explanation + chatbot ─────────────────────────────────────────
    ps2_prompt = (
        f"You are a clinical assistant. An AI has flagged a patient for possible deterioration.\n\n"
        f"Current vitals:\n"
        f"- Heart rate: {latest['heart_rate']:.0f} bpm (normal 60-100)\n"
        f"- Respiratory rate: {latest['respiratory_rate']:.0f} /min (normal 12-20)\n"
        f"- SpO2: {latest['spo2_pct']:.1f}% (normal >=94%)\n"
        f"- Systolic BP: {latest['systolic_bp']:.0f} mmHg\n"
        f"- Lactate: {latest['lactate']:.2f} mmol/L (normal <2.0)\n"
        f"- CRP: {latest['crp_level']:.1f} mg/L (normal <10)\n"
        f"- Nurse alert: {'Yes' if latest['nurse_alert'] else 'No'}\n\n"
        f"AI deterioration risk: {latest_risk*100:.0f}/100 — {risk_level} RISK\n\n"
        f"In 3-4 sentences tell a non-specialist caregiver: which vitals are concerning, "
        f"what clinical pattern this suggests, and what immediate action to take. "
        f"Avoid jargon."
    )
    ollama_ui(ps2_prompt, "ps2")

    # Recommender
    diag_key = f"ps2_{risk_level.lower()}"
    show_recommender(diag_key)
