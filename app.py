
import streamlit as st
import os, sys
import subprocess

BASE   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(BASE, "src")
DATA   = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
sys.path.insert(0, SRC)
sys.path.insert(0, BASE)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SentinAl",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def start_ollama_serve():
    """Starts Ollama serve in the background if it's not already running."""
    import requests
    try:
        # Check if Ollama is already running
        requests.get("http://localhost:11434/", timeout=1)
        return "Already running"
    except Exception:
        pass
        
    try:
        # If not running, start it
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        return process
    except Exception:
        return None

# Start Ollama serve silently when the app loads
start_ollama_serve()

# ── Branded loading screen (visible while heavy imports load) ─────────────────
_boot = st.empty()
_boot.markdown('''
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            padding:140px 0;text-align:center;font-family:Inter,system-ui,sans-serif">
    <div style="margin-bottom:12px">
        <svg width="52" height="52" viewBox="0 0 24 24" fill="#0EA5E9" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/>
        </svg>
    </div>
    <h2 style="margin:0;color:#0F172A;font-weight:800;letter-spacing:3px;font-size:26px">
        SENTINAL</h2>
    <p style="color:#64748B;margin:10px 0 0;font-size:14px">
        Loading clinical AI models &amp; dependencies…</p>
    <div style="width:180px;height:4px;background:#E2E8F0;border-radius:4px;
                margin-top:24px;overflow:hidden">
        <div style="width:100%;height:100%;border-radius:4px;
                    background:linear-gradient(90deg,#0EA5E9,#06B6D4,#0EA5E9);
                    background-size:200% 100%;
                    animation:_boot_shimmer 1.2s ease-in-out infinite"></div>
    </div>
</div>
<style>@keyframes _boot_shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}</style>
''', unsafe_allow_html=True)

# ── Heavy imports (deferred until after the loading screen is shown) ──────────
import pandas as pd
import numpy as np
import torch, pickle, requests
# plotly imported lazily inside PS2 section (saves ~1s on non-PS2 pages)
from PIL import Image
import torchvision.transforms as T

# ── Clear loading screen ──────────────────────────────────────────────────────
_boot.empty()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"]{font-family:'Inter',sans-serif}

/* ── Sidebar ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1628 0%,#0F1D32 100%);border-right:1px solid #1E293B}
section[data-testid="stSidebar"] *{color:#CBD5E1!important}
section[data-testid="stSidebar"] .stRadio>div{gap:2px}
section[data-testid="stSidebar"] .stRadio label{border-radius:10px;padding:10px 14px!important;transition:all .2s ease}
section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(255,255,255,0.06)}
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked){background:rgba(14,165,233,0.12)}

/* ── Risk alerts ──────────────────────────────────────────────── */
.risk-high{background:linear-gradient(135deg,#FEF2F2,#FECACA);border-left:4px solid #EF4444;padding:14px 18px;border-radius:12px;margin:8px 0;color:#7F1D1D}
.risk-mod {background:linear-gradient(135deg,#FFFBEB,#FDE68A);border-left:4px solid #F59E0B;padding:14px 18px;border-radius:12px;margin:8px 0;color:#78350F}
.risk-low {background:linear-gradient(135deg,#F0FDF4,#BBF7D0);border-left:4px solid #22C55E;padding:14px 18px;border-radius:12px;margin:8px 0;color:#14532D}

/* ── Overview metric cards ────────────────────────────────────── */
.metric-card{border-radius:16px;padding:26px 24px;position:relative;overflow:hidden}
.metric-card .label{font-size:10px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;opacity:.65;margin-bottom:10px}
.metric-card .value{font-size:38px;font-weight:800;margin:0;line-height:1.15}
.metric-card .badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;padding:5px 12px;border-radius:20px;margin-top:14px}
.mc-ps2{background:linear-gradient(135deg,#E0F2FE 0%,#BAE6FD 100%);color:#0C4A6E}
.mc-ps2 .badge{background:#DCFCE7;color:#166534}
.mc-ps1{background:linear-gradient(135deg,#FCE7F3 0%,#FBCFE8 100%);color:#831843}
.mc-ps1 .badge{background:#FFF1F2;color:#9F1239}
.mc-ps5{background:linear-gradient(135deg,#E0E7FF 0%,#C7D2FE 100%);color:#312E81}
.mc-ps5 .badge{background:#DBEAFE;color:#1E40AF}
.mc-modules{background:linear-gradient(135deg,#0F172A 0%,#1E293B 100%);color:#F1F5F9}
.mc-modules .badge{background:rgba(250,204,21,.15);color:#FDE68A}

/* ── Overview module cards ────────────────────────────────────── */
.module-card{border:1px solid #E2E8F0;border-radius:18px;padding:30px 26px;background:#fff;transition:all .25s ease;position:relative;overflow:hidden;min-height:300px;display:flex;flex-direction:column;cursor:pointer}
.module-card:hover{box-shadow:0 10px 40px rgba(0,0,0,.07);transform:translateY(-3px)}
.module-card .tag{position:absolute;top:18px;right:18px;font-size:10px;font-weight:700;letter-spacing:1.2px;color:#94A3B8;background:#F1F5F9;padding:5px 12px;border-radius:8px}
.module-card .icon{width:50px;height:50px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:18px}
.module-card h3{font-size:20px;font-weight:700;color:#0F172A;margin:0 0 10px}
.module-card .desc{font-size:13px;color:#64748B;line-height:1.7;flex:1}
.module-card .metrics{display:flex;gap:28px;margin-top:20px;padding-top:18px;border-top:1px solid #F1F5F9}
.module-card .metric-item .mlabel{font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin-bottom:2px}
.module-card .metric-item .mval{font-size:24px;font-weight:800}
.mc-vital{border-top:3px solid #0EA5E9}.mc-vital .icon{background:#E0F2FE;color:#0284C7}.mc-vital .mval{color:#0284C7}
.mc-stroke{border-top:3px solid #10B981}.mc-stroke .icon{background:#D1FAE5;color:#059669}.mc-stroke .mval{color:#059669}
.mc-wound{border-top:3px solid #F43F5E}.mc-wound .icon{background:#FFE4E6;color:#E11D48}.mc-wound .mval{color:#E11D48}

/* ── Recommender card ─────────────────────────────────────────── */
.recommender-card{background:linear-gradient(135deg,#ECFDF5 0%,#D1FAE5 100%);border:1px solid #A7F3D0;border-radius:18px;padding:34px 32px;margin:20px 0;display:flex;align-items:center;gap:28px;flex-wrap:wrap}
.rec-icon{width:58px;height:58px;background:#10B981;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:28px;color:#fff;flex-shrink:0}
.rec-info{flex:1;min-width:220px}
.rec-info h3{font-size:22px;font-weight:700;color:#064E3B;margin:0 0 6px}
.rec-desc{font-size:13px;color:#047857;line-height:1.65}
.rec-right{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.rec-badge-box{background:#fff;border:1px solid #A7F3D0;border-radius:14px;padding:16px 22px;text-align:center}
.rec-badge-box .rb-label{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#059669;margin-bottom:6px}
.rec-badge-box .rb-value{font-size:16px;font-weight:700;color:#064E3B}
.rec-launch{background:#10B981;color:#fff!important;font-weight:700;font-size:15px;border:none;border-radius:12px;padding:14px 32px;cursor:pointer;transition:all .2s;text-decoration:none;display:inline-block}
.rec-launch:hover{background:#059669;transform:translateY(-1px);box-shadow:0 4px 14px rgba(16,185,129,.35)}

/* ── Performance table ────────────────────────────────────────── */
.perf-header{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:16px;flex-wrap:wrap;gap:12px}
.perf-header h2{font-size:22px;font-weight:700;color:#0F172A;margin:0}
.perf-header .ph-sub{font-size:13px;color:#64748B;margin:4px 0 0}
.perf-table{width:100%;border-collapse:separate;border-spacing:0;border:1px solid #E2E8F0;border-radius:14px;overflow:hidden}
.perf-table th{background:#F8FAFC;color:#64748B;font-size:10px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;padding:15px 20px;text-align:left;border-bottom:1px solid #E2E8F0}
.perf-table td{padding:18px 20px;font-size:14px;color:#334155;border-bottom:1px solid #F1F5F9}
.perf-table tr:last-child td{border-bottom:none}
.perf-table tr:hover td{background:#F8FAFC}
.model-badge{padding:5px 14px;border-radius:8px;font-size:12px;font-weight:700;display:inline-block}
.badge-ps2{background:#DBEAFE;color:#1D4ED8}
.badge-ps1{background:#FCE7F3;color:#BE185D}
.badge-ps5{background:#D1FAE5;color:#059669}
.perf-status{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:700;color:#059669}
.perf-status-dot{width:8px;height:8px;background:#10B981;border-radius:50%;display:inline-block}

/* ── Ollama / doc / chat ──────────────────────────────────────── */
.ollama-box{background:linear-gradient(135deg,#F0FDF4,#DCFCE7);border-left:3px solid #10B981;padding:14px 18px;border-radius:12px;font-size:14px;line-height:1.75;margin-top:10px;color:#166534}
.doc-card{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:18px 22px;margin:10px 0;box-shadow:0 1px 4px rgba(0,0,0,.04);transition:box-shadow .2s}
.doc-card:hover{box-shadow:0 4px 16px rgba(0,0,0,.08)}
.doc-card h4{margin:0 0 6px;font-size:15px;color:#0F172A;font-weight:600}
.doc-card p{margin:3px 0;font-size:13px;color:#64748B}
.star{color:#F59E0B;font-size:13px}
.open-badge{background:#DCFCE7;color:#166534;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:600}
.closed-badge{background:#FEE2E2;color:#991B1B;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:600}

/* ── Sidebar custom elements ──────────────────────────────────── */
.sidebar-brand{display:flex;align-items:center;gap:12px;padding:4px 0;margin-bottom:2px}
.sidebar-brand .sb-icon{width:38px;height:38px;background:linear-gradient(135deg,#0EA5E9,#06B6D4);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px}
.sidebar-brand .sb-name{font-size:17px;font-weight:800;letter-spacing:2px;color:#F1F5F9!important}
.sidebar-sub{font-size:11px;color:#64748B!important;margin:-2px 0 0 50px}
.status-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;font-size:13px}
.sdot-green{width:8px;height:8px;background:#10B981;border-radius:50%;display:inline-block}
.sdot-red{width:8px;height:8px;background:#EF4444;border-radius:50%;display:inline-block}

/* ── Shimmer ──────────────────────────────────────────────────── */
@keyframes shimmer{0%{background-position:-468px 0}100%{background-position:468px 0}}
.shimmer-line{background:linear-gradient(90deg,#F1F5F9 25%,#E2E8F0 37%,#F1F5F9 63%);background-size:936px 100%;animation:shimmer 1.4s ease-in-out infinite;border-radius:8px;margin:10px 0}
.shimmer-metric{height:72px;border-radius:16px}
.shimmer-chart{height:320px;border-radius:16px}
.shimmer-text{height:16px;width:80%}
.shimmer-text-short{height:16px;width:50%}
.shimmer-card-block{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:18px;margin:10px 0}

/* ── Footer ───────────────────────────────────────────────────── */
.app-footer{text-align:center;color:#94A3B8;font-size:12px;padding:28px 0;border-top:1px solid #E2E8F0;margin-top:48px}
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Limit CPU threads to avoid over-subscription (especially helps Windows performance)
if not torch.cuda.is_available():
    torch.set_num_threads(min(4, os.cpu_count() or 4))

# ── Shimmer / skeleton helpers ───────────────────────────────────────────────
def _shimmer(css_class="shimmer-text", extra_style=""):
    st.markdown(f'<div class="shimmer-line {css_class}" style="{extra_style}"></div>',
                unsafe_allow_html=True)

def shimmer_metrics(n=4):
    cols = st.columns(n)
    for c in cols:
        with c:
            _shimmer("shimmer-metric")

def shimmer_chart():
    _shimmer("shimmer-chart")

def shimmer_content(lines=5):
    widths = ["95%","80%","90%","65%","75%"]
    for i in range(lines):
        _shimmer("shimmer-text", f"width:{widths[i % len(widths)]}")

def shimmer_cards(n=3):
    for _ in range(n):
        st.markdown(
            '<div class="shimmer-card-block">'
            '<div class="shimmer-line shimmer-text" style="width:60%"></div>'
            '<div class="shimmer-line shimmer-text-short"></div>'
            '</div>', unsafe_allow_html=True)

# ── Cached Ollama status (avoids 2s blocking call on every re-render) ────────
@st.cache_data(ttl=30, show_spinner=False)
def _check_ollama():
    try:
        requests.get("http://localhost:11434/", timeout=2)
        return True
    except Exception:
        return False

# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ps2():
    try:
        from model import TemporalTransformer
        with open(os.path.join(DATA,"feature_cols.pkl"),"rb") as f: feat_cols=pickle.load(f)
        with open(os.path.join(DATA,"scaler.pkl"),"rb") as f: scaler=pickle.load(f)
        with open(os.path.join(DATA,"encoders.pkl"),"rb") as f: encoders=pickle.load(f)

        tf = TemporalTransformer(len(feat_cols), 4)
        tf.load_state_dict(torch.load(
            os.path.join(MODELS,"best_transformer.pt"),
            map_location=DEVICE,
            weights_only=True
        ))
        tf = tf.to(DEVICE)   # <-- ADD THIS
        tf.eval()

        thresh_path = os.path.join(MODELS,"threshold_transformer.txt")
        threshold = float(open(thresh_path).read()) if os.path.exists(thresh_path) else 0.841
        return tf, scaler, encoders, feat_cols, threshold
    except Exception as e:
        st.error(f"PS2 load error: {e}")
        return None, None, None, None, 0.841

@st.cache_resource
def load_ps1():
    try:
        from ps1_model import FootWoundClassifier
        m = FootWoundClassifier(num_classes=4, dropout=0.5)
        m.load_state_dict(torch.load(
            os.path.join(MODELS, "best_ps1.pt"),
            map_location=DEVICE,
            weights_only=False
        )["model_state"])
        m = m.to(DEVICE)   # <-- ADD THIS
        m.eval()
        return m
    except Exception as e:
        st.error(f"PS1 load error: {e}")
        return None

@st.cache_resource
def load_ps5():
    try:
        from ps5_model import StrokeClassifier
        m = StrokeClassifier()
        m.load_state_dict(torch.load(
            os.path.join(MODELS, "best_ps5_classifier.pt"),
            map_location=DEVICE,
            weights_only=True
        )["model_state"])
        m = m.to(DEVICE)   # <-- ADD THIS
        m.eval()
        return m
    except Exception as e:
        st.error(f"PS5 load error: {e}")
        return None

# ── PS2 helpers ───────────────────────────────────────────────────────────────
def ps2_preprocess(df, scaler, encoders, feat_cols):
    df = df.copy(); df["patient_id"] = 0
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"]            = df["diastolic_bp"] + df["pulse_pressure"]/3
    df["shock_index"]    = df["heart_rate"]/(df["systolic_bp"]+1e-6)
    df["spo2_below_94"]  = (df["spo2_pct"]<94).astype(int)
    df["tachycardia"]    = (df["heart_rate"]>100).astype(int)
    df["tachypnea"]      = (df["respiratory_rate"]>20).astype(int)
    df["high_lactate"]   = (df["lactate"]>2.0).astype(int)
    df["crp_high"]       = (df["crp_level"]>50).astype(int)
    df["qsofa"]          = ((df["respiratory_rate"]>=22).astype(int)+
                            (df["systolic_bp"]<=100).astype(int)+df["nurse_alert"])
    for col in ["heart_rate","spo2_pct","respiratory_rate","systolic_bp"]:
        df[f"{col}_trend4"] = df[col].diff(4).fillna(0)
    oxy_map = {"none":0,"nasal":1,"mask":2,"hfnc":3,"niv":4}
    df["oxygen_device_enc"]   = df["oxygen_device"].map(oxy_map).fillna(0)
    df["gender_enc"]          = encoders["gender"].transform(df["gender"].astype(str))
    df["admission_type_enc"]  = encoders["admission_type"].transform(df["admission_type"].astype(str))
    df[feat_cols] = scaler.transform(df[feat_cols])
    return df

def ps2_score(df_proc, feat_cols, model, window=12):
    n = len(df_proc); scores = np.full(n, np.nan)
    if n < window: return scores
    static_feats = ["age","comorbidity_index","gender_enc","admission_type_enc"]
    sv = df_proc[static_feats].iloc[0].values.astype(np.float32)
    seqs,statics = [],[]
    for i in range(n-window+1):
        seqs.append(df_proc[feat_cols].iloc[i:i+window].values.astype(np.float32))
        statics.append(sv)
    with torch.no_grad():
        p = model(torch.FloatTensor(np.array(seqs)).to(DEVICE),
                  torch.FloatTensor(np.array(statics)).to(DEVICE)).cpu().numpy()
    scores[window-1:] = p; return scores

@st.cache_data
def make_demo(risk="high"):
    n=48; rng=np.random.default_rng(42 if risk=="high" else 7)
    if risk=="high":
        hr=76+np.linspace(0,32,n)+rng.normal(0,2,n)
        rr=14+np.linspace(0,10,n)+rng.normal(0,1,n)
        spo=97-np.linspace(0,8,n)+rng.normal(0,0.4,n)
        sbp=122-np.linspace(0,28,n)+rng.normal(0,3,n)
        lac=1.2+np.linspace(0,2.2,n)+rng.normal(0,0.1,n)
        crp=14+np.linspace(0,40,n)+rng.normal(0,2,n)
        age,cmb,adm=72,5,"ED"
    else:
        hr=82+np.linspace(0,10,n)+rng.normal(0,3,n)
        rr=16+np.linspace(0,4,n)+rng.normal(0,1,n)
        spo=96-np.linspace(0,2,n)+rng.normal(0,0.4,n)
        sbp=118-np.linspace(0,10,n)+rng.normal(0,4,n)
        lac=1.4+np.linspace(0,0.6,n)+rng.normal(0,0.1,n)
        crp=18+np.linspace(0,14,n)+rng.normal(0,2,n)
        age,cmb,adm=65,3,"Transfer"
    return pd.DataFrame({
        "hour_from_admission": list(range(n)),
        "heart_rate":hr.clip(40,160),"respiratory_rate":rr.clip(8,40),
        "spo2_pct":spo.clip(70,100),"temperature_c":(36.7+rng.normal(0,0.2,n)).clip(35,41),
        "systolic_bp":sbp.clip(60,200),"diastolic_bp":(sbp*0.65).clip(40,130),
        "oxygen_device":["none"]*32+["nasal"]*10+["mask"]*6,
        "oxygen_flow":[0]*32+[2]*10+[4]*6,
        "mobility_score":rng.integers(1,5,n).tolist(),
        "nurse_alert":[0]*32+[1]*16,
        "wbc_count":(7+rng.normal(0,1.2,n)).clip(2,25),
        "lactate":lac.clip(0.4,10),"creatinine":(1.1+rng.normal(0,0.15,n)).clip(0.4,8),
        "crp_level":crp.clip(0,300),"hemoglobin":(13+rng.normal(0,0.3,n)).clip(5,18),
        "sepsis_risk_score":np.clip(0.15+np.linspace(0,0.65 if risk=="high" else 0.25,n),0,1),
        "age":age,"gender":"M","comorbidity_index":cmb,"admission_type":adm,
    })

# Best model for this project: qwen2.5:3b (1.9 GB, fits in RAM, medical-aware)
OLLAMA_MODELS = ["qwen2.5:3b", "codellama:latest"]

# Labels for CLIP zero-shot classification
_FOOT_WOUND_LABELS = [
    "a medical photograph of a foot wound or diabetic foot ulcer",
    "a photograph of a healthy foot with no wound",
    "a photograph of an animal or pet",
    "a photograph of food or a meal",
    "a photograph of a landscape, building, or scenery",
    "a photograph of a person's face or portrait",
    "a screenshot, diagram, or document",
    "a random photograph not related to foot wounds",
]
_FOOT_WOUND_THRESHOLD = 0.25  # minimum probability for the foot-wound label


@st.cache_resource
def _load_clip():
    """Load CLIP model + processor on the active device. Cached across reruns."""
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    return clip_model, processor


def validate_foot_wound_image(img) -> tuple:
    """
    Use CLIP zero-shot classification to check if the image is a foot wound.
    Returns (is_valid: bool, message: str).
    """
    try:
        clip_model, processor = _load_clip()
    except Exception:
        return True, "clip_unavailable"

    # Single forward pass — returns logits_per_image directly (version-safe)
    inputs = processor(text=_FOOT_WOUND_LABELS, images=img,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()

    foot_wound_prob = float(probs[0])
    best_idx = int(probs.argmax())
    is_valid = best_idx == 0 or foot_wound_prob >= _FOOT_WOUND_THRESHOLD

    if is_valid:
        return True, f"Foot wound confidence: {foot_wound_prob*100:.1f}%"
    else:
        best_label = _FOOT_WOUND_LABELS[best_idx].replace("a photograph of ", "")
        return False, (f"Image looks like {best_label} "
                       f"(foot wound confidence: {foot_wound_prob*100:.1f}%)")


def ask_ollama(prompt: str, model: str = "qwen2.5:3b") -> str:
    """
    Streaming Ollama call — works on ALL Ollama versions.
    stream=False causes HTTP 500 on Ollama 0.3+, streaming fixes this.
    """
    import json as _json
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        )
        if r.status_code != 200:
            return f"❌ Ollama HTTP {r.status_code} — try: ollama pull {model}"
        full = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = _json.loads(line.decode("utf-8"))
                full += chunk.get("response", "")
                if chunk.get("done", False):
                    break
            except _json.JSONDecodeError:
                continue
        return full.strip() or "Model returned empty response."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Run: ollama serve"
    except requests.exceptions.Timeout:
        return "⏱ Timed out after 120s — model loading. Try again in 30s."
    except Exception as e:
        return f"❌ Error: {e}"


def ollama_ui(context_prompt: str, module_key: str):
    """Reusable clinical explanation + chatbot. Works in PS2, PS1, PS5."""
    st.markdown("---")
    st.markdown("### 🤖 Clinical explanation")
    col_btn, col_model = st.columns([2, 1])
    with col_model:
        sel_model = st.selectbox("Model", OLLAMA_MODELS, index=0,
                                 key=f"olm_{module_key}",
                                 label_visibility="collapsed",
                                 help="qwen2.5:3b recommended — fast, medical-aware, 1.9 GB")
    with col_btn:
        if st.button("Generate clinical summary", key=f"olmbtn_{module_key}"):
            with st.spinner(f"Asking {sel_model}... (10–30s)"):
                result = ask_ollama(context_prompt, sel_model)
            st.session_state[f"olmres_{module_key}"] = result
    if f"olmres_{module_key}" in st.session_state:
        st.markdown(
            f'''<div class="ollama-box">{st.session_state[f"olmres_{module_key}"]}</div>''',
            unsafe_allow_html=True)

    st.markdown("### 💬 Ask the clinical assistant")
    chat_key = f"chat_{module_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    q = st.text_input("Ask a question...",
                      placeholder="e.g. What does this result mean for daily life?",
                      key=f"chatinput_{module_key}", label_visibility="collapsed")
    if q:
        full_prompt = (f"{context_prompt}\n\n"
                       f"Caregiver/patient question: {q}\n"
                       f"Answer in 2-3 sentences, simple language, no medical jargon.")
        with st.spinner("Thinking..."):
            ans = ask_ollama(full_prompt, sel_model)
        st.session_state[chat_key].append(("You", q))
        st.session_state[chat_key].append(("Assistant", ans))
    for role, msg in st.session_state[chat_key][-10:]:
        align = "right" if role == "You" else "left"
        bg = "#E6F1FB" if role == "You" else "#F1EFE8"
        st.markdown(
            f'''<div style="text-align:{align};margin:4px 0">'''
            f'''<span style="background:{bg};padding:8px 14px;border-radius:16px;'''
            f'''display:inline-block;max-width:85%;font-size:13px;white-space:pre-wrap">'''
            f'''{msg}</span></div>''', unsafe_allow_html=True)

IMG_TFM = T.Compose([T.Resize((224,224)),T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ── Recommender widget ────────────────────────────────────────────────────────
def show_recommender(diagnosis_key: str):
    """Renders the full doctor recommender UI block (100% free, no API keys)."""
    import recommender as rec

    info    = rec.get_specialists_for_diagnosis(diagnosis_key)
    urgency = info["urgency_level"]
    urg_msg = info["urgency_message"]
    css     = "risk-high" if "🔴" in urgency else "risk-mod" if "🟠" in urgency else "risk-low"

    st.markdown("---")
    st.subheader("📍 Nearest Specialist Recommender")
    st.markdown(f'<div class="{css}"><strong>{urgency}</strong> — {urg_msg}</div>',
                unsafe_allow_html=True)

    # Specialist types for this diagnosis
    st.markdown("**Recommended specialist types for this diagnosis:**")
    spec_cols = st.columns(len(info["specialists"]))
    for col, name in zip(spec_cols, info["specialists"]):
        col.info(f"🩺 {name}")

    # ── Detect user location via IP (once per session) ──────────────────────
    if "_user_location" not in st.session_state:
        st.session_state["_user_location"] = rec.detect_location()
    detected = st.session_state["_user_location"]

    # Build city list for the selectbox; detect default
    city_names = list(rec.MAJOR_CITIES.keys())
    default_city = "Pune, Maharashtra"
    if detected and detected["city"]:
        # Try to match detected city to a list entry
        for c in city_names:
            if detected["city"].lower() in c.lower():
                default_city = c
                break

    # Pre-set selectbox default on first render
    sel_key = f"city_sel_{diagnosis_key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = default_city

    # ── Location input ────────────────────────────────────────────────────────
    st.markdown("**Find nearby hospitals & clinics:**")
    selected_city = st.selectbox(
        "Select a city (type to search)",
        city_names,
        key=sel_key,
        label_visibility="collapsed",
    )
    custom_loc = st.text_input(
        "Or type any location and press Enter:",
        placeholder="e.g. Wardha, 411005, or any address",
        key=f"custom_{diagnosis_key}",
    )

    # Resolve: custom text overrides selectbox when filled
    location_label = custom_loc.strip() if custom_loc and custom_loc.strip() else selected_city

    # Auto-search when location changes or on first load
    cached_loc   = st.session_state.get(f"loc_used_{diagnosis_key}", "")
    first_load   = f"results_{diagnosis_key}" not in st.session_state
    loc_changed  = location_label != cached_loc

    def _run_search():
        coords = rec.geocode_location(location_label)
        if coords is None:
            st.error(f"Could not find **{location_label}**. "
                     "Check the spelling and try again.")
            return False
        facilities = rec.search_nearby_facilities(
            coords[0], coords[1], radius_m=15000, max_results=10)
        st.session_state[f"results_{diagnosis_key}"]   = facilities
        st.session_state[f"coords_{diagnosis_key}"]    = coords
        st.session_state[f"loc_used_{diagnosis_key}"]   = location_label
        return True

    if first_load or loc_changed:
        _doc_ph = st.empty()
        with _doc_ph.container():
            shimmer_cards(3)
        with st.spinner(f"Searching near {location_label}…"):
            ok = _run_search()
        _doc_ph.empty()
        if ok is False:
            return

    # ── Display results ───────────────────────────────────────────────────────
    facilities = st.session_state.get(f"results_{diagnosis_key}", [])
    loc_used   = st.session_state.get(f"loc_used_{diagnosis_key}", location_label)

    if not facilities:
        if not first_load:
            st.info(f"No hospitals or clinics found within 15 km of **{loc_used}**. "
                    "Try a larger city nearby.")
        return

    if loc_used != location_label:
        st.caption(f"Showing results for **{loc_used}**. "
                   f"Click **Search** to update for **{location_label}**.")

    for i, fac in enumerate(facilities):
        type_badge = {
            "Hospital": '<span class="open-badge">Hospital</span>',
            "Clinic":   '<span style="background:#DBEAFE;color:#1E40AF;padding:3px 10px;'
                        'border-radius:10px;font-size:11px;font-weight:600">Clinic</span>',
            "Doctor":   '<span style="background:#F3E8FF;color:#6B21A8;padding:3px 10px;'
                        'border-radius:10px;font-size:11px;font-weight:600">Doctor</span>',
        }.get(fac["facility_type"], "")

        phone_str = f"📞 {fac['phone']}" if fac.get("phone") else ""
        web_str = ""
        if fac.get("website"):
            web_str = (f' · <a href="{fac["website"]}" target="_blank">'
                       f'🌐 Website</a>')
        hours_str = f"🕐 {fac['opening_hours']}" if fac.get("opening_hours") else ""
        maps_link = (f'<a href="{fac["maps_url"]}" target="_blank">'
                     f'📍 Open in Maps</a>')
        dist_str = f"🚶 {fac['distance_km']} km away"

        st.markdown(f"""
<div class="doc-card">
  <h4>#{i+1} &nbsp; {fac['name']} &nbsp; {type_badge}</h4>
  <p>📌 {fac['address']}</p>
  <p>{dist_str} &nbsp;&nbsp; {phone_str} &nbsp;&nbsp; {hours_str}</p>
  <p>{maps_link}{web_str}</p>
</div>""", unsafe_allow_html=True)

    # Map view
    try:
        map_data = [{"lat": f["lat"], "lon": f["lon"]}
                    for f in facilities if f.get("lat") and f.get("lon")]
        if map_data:
            st.markdown("**Map view:**")
            st.map(pd.DataFrame(map_data), zoom=12)
    except Exception:
        pass


# ── Navigation helper (card clicks set ?nav= query param) ────────────────────
_nav_param = st.query_params.get("nav")
if _nav_param:
    _NAV_MAP = {"ps2": "📈  Vital Signs (PS2)", "ps5": "🧠  Stroke Detector (PS5)", "ps1": "🦶  Foot Wound (PS1)"}
    if _nav_param in _NAV_MAP:
        st.session_state["nav_module"] = _NAV_MAP[_nav_param]
    del st.query_params["nav"]
    st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
    <div class="sidebar-brand">
        <div class="sb-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="#fff"><path d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/></svg></div>
        <span class="sb-name">SENTINAL</span>
    </div>
    <div class="sidebar-sub">Predict Early. Protect Lives.</div>
    ''', unsafe_allow_html=True)
    st.markdown("---")
    module = st.radio("Module", [
        "🏠  Overview",
        "📈  Vital Signs (PS2)",
        "🦶  Foot Wound (PS1)",
        "🧠  Stroke Detector (PS5)",
    ], label_visibility="collapsed", key="nav_module")
    st.markdown("---")
    # System status
    st.markdown("##### SYSTEM STATUS")
    _ollama_ok = _check_ollama()
    _gpu_ok = torch.cuda.is_available()
    _gpu_name = torch.cuda.get_device_name(0) if _gpu_ok else "CPU only"
    st.markdown(f'''
    <div class="status-row"><span>Compute</span><span style="font-size:11px;color:{'#10B981' if _gpu_ok else '#94A3B8'}">{_gpu_name}</span></div>
    <div class="status-row"><span>Ollama LLM</span><span class="{"sdot-green" if _ollama_ok else "sdot-red"}"></span></div>
    <div class="status-row"><span>System Health</span><span class="sdot-green"></span></div>
    ''', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("All AI runs locally · No patient data leaves this device")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
from src.ui.home_ui import render_home_page
from src.ui.ps2_ui import render_ps2_page
from src.ui.ps1_ui import render_ps1_page
from src.ui.ps5_ui import render_ps5_page
from src.ui.components_ui import shimmer_metrics, shimmer_content, shimmer_chart

if module.startswith("🏠"):
    render_home_page()

elif module.startswith("📈"):
    # Show shimmer while model loads (first time only — cached after)
    _ps2_placeholder = st.empty()
    with _ps2_placeholder.container():
        shimmer_metrics()
        shimmer_chart()
    with st.spinner("⚙️ Loading PS2 Temporal Transformer…"):
        tf_model, scaler, encoders, feat_cols, threshold = load_ps2()
    _ps2_placeholder.empty()
    
    render_ps2_page(tf_model, scaler, encoders, feat_cols, threshold, ollama_ui, show_recommender)

elif module.startswith("🦶"):
    _ps1_placeholder = st.empty()
    with _ps1_placeholder.container():
        shimmer_metrics(2)
        shimmer_content(3)
    with st.spinner("⚙️ Loading PS1 wound classifier…"):
        model = load_ps1()
    _ps1_placeholder.empty()
    
    render_ps1_page(model, validate_foot_wound_image, ollama_ui, show_recommender, shimmer_metrics, shimmer_content)

elif module.startswith("🧠"):
    _ps5_placeholder = st.empty()
    with _ps5_placeholder.container():
        shimmer_metrics(2)
        shimmer_content(3)
    with st.spinner("⚙️ Loading PS5 stroke detector…"):
        model = load_ps5()
    _ps5_placeholder.empty()
    
    render_ps5_page(model, ollama_ui, show_recommender, shimmer_metrics, shimmer_content)

st.markdown('''
<div class="app-footer">
    Sentin<strong>Al</strong> 
    &middot; Not for clinical use &middot; All AI runs locally
</div>''', unsafe_allow_html=True)
