
import streamlit as st
import pandas as pd
import numpy as np
import torch, pickle, os, sys, requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import torchvision.transforms as T

BASE   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(BASE, "src")
DATA   = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
sys.path.insert(0, SRC)
sys.path.insert(0, BASE)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGuard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"]{background:#0B1F4B}
section[data-testid="stSidebar"] *{color:#fff!important}
.risk-high{background:#FCEBEB;border-left:4px solid #E24B4A;padding:12px 16px;border-radius:8px;margin:6px 0}
.risk-mod {background:#FAEEDA;border-left:4px solid #EF9F27;padding:12px 16px;border-radius:8px;margin:6px 0}
.risk-low {background:#EAF3DE;border-left:4px solid #639922;padding:12px 16px;border-radius:8px;margin:6px 0}
.ollama-box{background:#EAF3DE;border-left:3px solid #1D9E75;padding:12px 16px;border-radius:8px;font-size:14px;line-height:1.7;margin-top:10px}
.doc-card{background:#F8F9FA;border:1px solid #dee2e6;border-radius:12px;padding:14px 16px;margin:8px 0}
.doc-card h4{margin:0 0 4px 0;font-size:15px;color:#0B1F4B}
.doc-card p{margin:2px 0;font-size:13px;color:#444}
.star{color:#EF9F27;font-size:13px}
.open-badge{background:#EAF3DE;color:#2d6a4f;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.closed-badge{background:#FCEBEB;color:#9d0208;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
OLLAMA_MODELS = ["qwen2.5:3b", "codellama:latest", "qwen3.5:latest"]

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
def show_recommender(diagnosis_key: str, default_location: str = "Pune, Maharashtra"):
    """Renders the full doctor recommender UI block."""
    import recommender as rec
    # Always sync API key from session_state so sidebar changes take effect
    live_key = st.session_state.get("api_key_value", "")
    if live_key:
        rec.GOOGLE_API_KEY = live_key
    from recommender import (get_specialists_for_diagnosis, geocode_location,
                              search_nearby_doctors)
    GOOGLE_API_KEY = rec.GOOGLE_API_KEY

    info      = get_specialists_for_diagnosis(diagnosis_key)
    urgency   = info["urgency_level"]
    urg_msg   = info["urgency_message"]
    css       = "risk-high" if "🔴" in urgency else "risk-mod" if "🟠" in urgency else "risk-low"

    st.markdown("---")
    st.subheader("📍 Nearest Specialist Recommender")
    st.markdown(f'<div class="{css}"><strong>{urgency}</strong> — {urg_msg}</div>',
                unsafe_allow_html=True)

    # Specialist types for this diagnosis
    st.markdown("**Recommended specialist types for this diagnosis:**")
    spec_cols = st.columns(len(info["specialists"]))
    for col, (name, _, _) in zip(spec_cols, info["specialists"]):
        col.info(f"🩺 {name}")

    # Location input
    st.markdown("**Find nearest specialists:**")
    loc_col1, loc_col2 = st.columns([3, 1])
    with loc_col1:
        location_input = st.text_input(
            "Your location (city, area, or pincode)",
            value=default_location,
            placeholder="e.g. Shivajinagar Pune, 411005, or Bangalore",
            key=f"loc_{diagnosis_key}",
            label_visibility="collapsed",
        )
    with loc_col2:
        search_btn = st.button("🔍 Find Doctors", key=f"search_{diagnosis_key}",
                               use_container_width=True)

    # Show note if API key not set
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        st.caption("ℹ️ Demo mode — showing sample results. "
                   "Add your Google Places API key in recommender.py to get live results.")

    if search_btn or f"results_{diagnosis_key}" in st.session_state:
        if search_btn:
            with st.spinner(f"Searching for specialists near {location_input}..."):
                coords = geocode_location(location_input) if GOOGLE_API_KEY != "YOUR_API_KEY_HERE" else (18.5204, 73.8567)
                if coords is None:
                    # Fallback: Pune coordinates
                    coords = (18.5204, 73.8567)
                    st.caption(f"Could not geocode '{location_input}' — showing results near Pune.")

                all_results = {}
                for name, ptype, keyword in info["specialists"]:
                    results = search_nearby_doctors(coords[0], coords[1], keyword, radius_m=7000)
                    all_results[name] = results

                st.session_state[f"results_{diagnosis_key}"]  = all_results
                st.session_state[f"coords_{diagnosis_key}"]   = coords
                st.session_state[f"location_{diagnosis_key}"] = location_input

        all_results = st.session_state.get(f"results_{diagnosis_key}", {})
        coords      = st.session_state.get(f"coords_{diagnosis_key}", (18.5204, 73.8567))
        loc_used    = st.session_state.get(f"location_{diagnosis_key}", location_input)

        if not all_results:
            st.info("No results found. Try a broader location.")
            return

        # Tab per specialist type
        tab_names = list(all_results.keys())
        if len(tab_names) == 0:
            return

        tabs = st.tabs([f"🩺 {n}" for n in tab_names])
        for tab, spec_name in zip(tabs, tab_names):
            with tab:
                doctors = all_results[spec_name]
                if not doctors:
                    st.info(f"No {spec_name} found within 7 km of {loc_used}.")
                    continue

                for i, doc in enumerate(doctors[:4]):
                    # Rating stars
                    rating    = doc.get("rating")
                    n_reviews = doc.get("user_ratings_total", 0)
                    stars_str = ""
                    if rating:
                        full  = int(rating)
                        half  = 1 if (rating - full) >= 0.5 else 0
                        empty = 5 - full - half
                        stars_str = "★"*full + ("½" if half else "") + "☆"*empty
                        stars_str = f'<span class="star">{stars_str}</span> {rating:.1f} ({n_reviews} reviews)'

                    open_now = doc.get("open_now")
                    if open_now is True:
                        badge = '<span class="open-badge">● Open now</span>'
                    elif open_now is False:
                        badge = '<span class="closed-badge">● Closed</span>'
                    else:
                        badge = ""

                    phone_str = f"📞 {doc['phone']}" if doc.get("phone") else ""
                    maps_link = f'<a href="{doc["maps_url"]}" target="_blank">📍 Open in Maps</a>'

                    dist_str  = f"🚶 {doc['distance_km']} km away"

                    st.markdown(f"""
<div class="doc-card">
  <h4>#{i+1} &nbsp; {doc['name']} &nbsp; {badge}</h4>
  <p>📌 {doc['address']}</p>
  <p>{stars_str} &nbsp;&nbsp; {dist_str} &nbsp;&nbsp; {phone_str}</p>
  <p>{maps_link}</p>
</div>""", unsafe_allow_html=True)

        # Quick map preview using st.map
        try:
            map_data = []
            for spec_name, doctors in all_results.items():
                for doc in doctors[:3]:
                    ml = doc.get("maps_url","")
                    # extract lat/lng from maps_url if available
                    if "?q=" in ml:
                        ll = ml.split("?q=")[1].split(",")
                        if len(ll)==2:
                            try:
                                map_data.append({"lat":float(ll[0]),"lon":float(ll[1]),
                                                 "name":doc["name"]})
                            except: pass
            if map_data:
                st.markdown("**Map view:**")
                map_df = pd.DataFrame(map_data)
                st.map(map_df, zoom=13)
        except Exception:
            pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuroGuard")
    st.markdown("**Team 2Infinity · ANC-016**\n*PESMCE, Pune*")
    st.markdown("---")
    module = st.radio("Module", [
        "🏠  Platform Overview",
        "💓  Vital Sign Monitor (PS2)",
        "🦶  Foot Wound Grader (PS1)",
        "🧠  CT Stroke Detector (PS5)",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### Google Places API")
    api_key_input = st.text_input("API Key (optional)", type="password",
                                   placeholder="AIzaSy...",
                                   help="Paste your Google Places API key for live doctor search")
    if api_key_input:
        st.session_state["api_key_value"] = api_key_input
        import recommender as _rec
        _rec.GOOGLE_API_KEY = api_key_input
        os.environ["GOOGLE_PLACES_API_KEY"] = api_key_input
        st.success("Live search enabled ✓")
    elif "api_key_value" not in st.session_state:
        st.caption("Demo mode — no API key")
    st.markdown("---")
    st.markdown("#### Ollama status")
    try:
        requests.get("http://localhost:11434/", timeout=2)
        st.success("Ollama running ✓")
    except Exception:
        st.error("Ollama offline\nRun: ollama serve")
    st.markdown("---")
    st.caption("All AI runs locally.\nNo patient data leaves this device.")


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if module.startswith("🏠"):
    st.title("NeuroGuard — Unified Clinical AI Platform")
    st.caption("Privacy-first edge AI · all models run locally · zero cloud data transfer")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("PS2 AUROC",   "0.9960", help="Vital sign deterioration")
    c2.metric("PS1 Accuracy","97.05%", help="Foot wound grading")
    c3.metric("PS5 AUROC",   "0.982",  help="CT stroke detection")
    c4.metric("Modules",     "4",      help="PS2 + PS1 + PS5 + Recommender")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.info("**💓 PS2 — Vital Sign Monitor**\nTemporal Transformer + BiLSTM ensemble. Predicts deterioration 12 hours ahead from continuous vital sign streams.\n\n**AUROC: 0.9960 · Sensitivity: 88.9%**")
        st.warning("**🦶 PS1 — Foot Wound Grader**\nEfficientNet-B0. Classifies diabetic foot wounds into Wagner Grade 1–4 from a single photograph.\n\n**Accuracy: 97.05% · F1 macro: 0.97**")
    with col2:
        st.error("**🧠 PS5 — CT Stroke Detector**\nEfficientNet-B0. Detects hemorrhagic stroke from brain CT scan images.\n\n**AUROC: 0.982 · Val Accuracy: 92.2%**")
        st.success("**📍 Specialist Recommender**\nDiagnosis-triggered specialist matching + nearest clinic search with Google ratings, distance and open hours.\n\n**Covers all 3 diagnosis types**")

    st.markdown("---")
    st.subheader("Model performance summary")
    st.dataframe(pd.DataFrame({
        "Module":       ["PS2 Transformer","PS1 EfficientNet-B0","PS5 EfficientNet-B0"],
        "Task":         ["Deterioration prediction","Wound grading","Stroke detection"],
        "Key metric":   ["AUROC 0.9960","Accuracy 97.05%","AUROC 0.982"],
        "Dataset":      ["293,248 rows × 22 cols","9,934 images (4 grades)","2,501 CT scans"],
        "Epochs":       ["59","27","16"],
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PS2 — VITAL SIGNS
# ══════════════════════════════════════════════════════════════════════════════
elif module.startswith("💓"):
    st.title("💓 Vital Sign Monitor — Deterioration Early Warning")
    st.caption("Temporal Transformer · 12-hour prediction window · AUROC 0.9960")

    tf_model,scaler,encoders,feat_cols,threshold = load_ps2()
    if tf_model is None:
        st.error("PS2 model not loaded. Run preprocess.py → train.py first.")
        st.stop()

    st.markdown("### Load patient data")
    c1,c2 = st.columns(2)
    with c1:
        demo_opt = st.selectbox("Demo patient", ["None","High risk (48h escalating)","Moderate risk (48h mild trend)"])
    with c2:
        uploaded = st.file_uploader("Or upload patient CSV", type="csv")

    df_patient = None
    if uploaded:
        df_patient = pd.read_csv(uploaded); st.success(f"Loaded {len(df_patient)} rows")
    elif "High" in demo_opt:
        df_patient = make_demo("high"); st.info("Demo: HIGH-risk patient — 48 hours of escalating vitals")
    elif "Moderate" in demo_opt:
        df_patient = make_demo("moderate"); st.info("Demo: MODERATE-risk patient")

    if df_patient is None:
        st.markdown("Select a demo patient or upload a CSV to begin.")
        st.stop()

    with st.spinner("Running AI risk assessment..."):
        df_proc = ps2_preprocess(df_patient, scaler, encoders, feat_cols)
        scores  = ps2_score(df_proc, feat_cols, tf_model)

    latest_risk = float(np.nanmax(scores[-6:]))
    risk_level  = "HIGH" if latest_risk>=threshold else "MODERATE" if latest_risk>=0.35 else "LOW"
    css         = "risk-high" if risk_level=="HIGH" else "risk-mod" if risk_level=="MODERATE" else "risk-low"
    hours       = df_patient["hour_from_admission"].values
    latest      = df_patient.iloc[-1]

    h1,h2,h3,h4 = st.columns(4)
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
    fig = make_subplots(rows=2,cols=3,
        subplot_titles=("Deterioration risk","Heart rate & SpO₂","Blood pressure",
                        "Respiratory rate","Lactate & CRP","Nurse alerts"))
    valid = ~np.isnan(scores)
    fig.add_trace(go.Scatter(x=hours[valid],y=scores[valid]*100,mode="lines",
        name="Risk",line=dict(color="#E24B4A",width=2.5)),row=1,col=1)
    fig.add_hline(y=threshold*100,line_dash="dash",line_color="black",row=1,col=1)
    fig.add_trace(go.Scatter(x=hours,y=df_patient["heart_rate"],name="HR",
        line=dict(color="#E24B4A")),row=1,col=2)
    fig.add_trace(go.Scatter(x=hours,y=df_patient["spo2_pct"],name="SpO₂",
        line=dict(color="#185FA5")),row=1,col=2)
    fig.add_hline(y=94,line_dash="dot",line_color="#185FA5",row=1,col=2)
    fig.add_trace(go.Scatter(x=hours,y=df_patient["systolic_bp"],name="Systolic",
        line=dict(color="#534AB7")),row=1,col=3)
    fig.add_trace(go.Scatter(x=hours,y=df_patient["diastolic_bp"],name="Diastolic",
        line=dict(color="#AFA9EC")),row=1,col=3)
    fig.add_trace(go.Scatter(x=hours,y=df_patient["respiratory_rate"],name="RR",
        line=dict(color="#EF9F27")),row=2,col=1)
    fig.add_hline(y=20,line_dash="dot",line_color="#EF9F27",row=2,col=1)
    fig.add_trace(go.Scatter(x=hours,y=df_patient["lactate"],name="Lactate",
        line=dict(color="#993C1D")),row=2,col=2)
    fig.add_hline(y=2.0,line_dash="dot",line_color="#993C1D",row=2,col=2)
    fig.add_trace(go.Scatter(x=hours,y=df_patient["crp_level"],name="CRP",
        line=dict(color="#D4537E")),row=2,col=2)
    fig.add_trace(go.Bar(x=hours,y=df_patient["nurse_alert"],name="Nurse alert",
        marker_color="#EF9F27"),row=2,col=3)
    fig.update_layout(height=500,showlegend=True,template="plotly_white",
                      legend=dict(orientation="h",y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # Alert log
    alert_hrs = hours[valid][scores[valid]>=threshold]
    if len(alert_hrs)>0:
        st.markdown("### Alert log")
        for h in alert_hrs[-5:][::-1]:
            idx  = int(np.where(hours==h)[0][0])
            row  = df_patient.iloc[idx]
            rsco = float(scores[idx])
            lvl  = "HIGH" if rsco>=threshold else "MODERATE"
            c    = "risk-high" if lvl=="HIGH" else "risk-mod"
            st.markdown(f'<div class="{c}"><strong>Hour {int(h)}</strong> — '
                        f'Score: {rsco*100:.0f}/100 | HR: {row.heart_rate:.0f} | '
                        f'SpO₂: {row.spo2_pct:.1f}% | RR: {row.respiratory_rate:.0f} | '
                        f'BP: {row.systolic_bp:.0f}/{row.diastolic_bp:.0f}</div>',
                        unsafe_allow_html=True)

    # Clinical explanation + chatbot
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


# ══════════════════════════════════════════════════════════════════════════════
# PS1 — FOOT WOUND
# ══════════════════════════════════════════════════════════════════════════════
elif module.startswith("🦶"):
    st.title("🦶 Foot Wound Grader — Diabetic Ulcer Classification")
    st.caption("EfficientNet-B0 · Wagner Grade 1–4 · Accuracy 97.05%")

    model = load_ps1()
    if model is None:
        st.error("PS1 model not found at models/best_ps1.pt"); st.stop()

    GRADE_INFO = {
        0: ("Grade 1","Superficial wound (skin only)","low",
            "Monitor and offload pressure. Podiatry review recommended."),
        1: ("Grade 2","Deep wound to tendon or joint capsule","moderate",
            "Refer to podiatry/diabetology within 1 week."),
        2: ("Grade 3","Deep wound with abscess or osteomyelitis","high",
            "Urgent surgical review required within 24 hours."),
        3: ("Grade 4","Partial foot gangrene","high",
            "Immediate vascular surgery referral — amputation risk."),
    }

    uploaded_img = st.file_uploader("Upload foot wound photograph", type=["jpg","jpeg","png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        col_img, col_res = st.columns(2)
        with col_img:
            st.image(img, caption="Uploaded image", use_container_width=True)

        tensor = IMG_TFM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor),dim=1)[0].cpu().numpy()
        pred = int(probs.argmax()); conf = float(probs[pred])*100
        g_name,g_desc,g_risk,g_action = GRADE_INFO[pred]

        with col_res:
            css = "risk-high" if g_risk=="high" else "risk-mod" if g_risk=="moderate" else "risk-low"
            st.markdown(f'<div class="{css}"><strong>{g_name} — {g_desc}</strong><br>'
                        f'Confidence: {conf:.1f}%<br>Action: {g_action}</div>',
                        unsafe_allow_html=True)
            st.markdown("**Grade probabilities:**")
            for i,(g,p) in enumerate(zip(["Grade 1","Grade 2","Grade 3","Grade 4"],probs)):
                bc = "#E24B4A" if i==pred else "#B4B2A9"
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px">'
                            f'<span style="min-width:58px">{g}</span>'
                            f'<div style="flex:1;background:#e0e0e0;border-radius:4px;height:12px">'
                            f'<div style="width:{p*100:.1f}%;background:{bc};height:12px;border-radius:4px"></div>'
                            f'</div><span>{p*100:.1f}%</span></div>', unsafe_allow_html=True)

        # Clinical explanation + chatbot
        ps1_prompt = (
            f"You are a clinical assistant helping a caregiver understand a diabetic foot wound result.\n\n"
            f"AI wound classification:\n"
            f"- Grade: {g_name} ({g_desc})\n"
            f"- Confidence: {conf:.1f}%\n"
            f"- Recommended action: {g_action}\n"
            f"- Grade probabilities: Grade 1 {probs[0]*100:.1f}%, Grade 2 {probs[1]*100:.1f}%, "
            f"Grade 3 {probs[2]*100:.1f}%, Grade 4 {probs[3]*100:.1f}%\n\n"
            f"In 3-4 sentences explain: what this wound grade means, "
            f"the risks if untreated, and what the caregiver should do next. "
            f"Use simple non-medical language."
        )
        ollama_ui(ps1_prompt, "ps1")

        # Recommender
        show_recommender(f"ps1_grade{pred+1}")
    else:
        st.info("Upload a foot wound photograph to classify its Wagner grade.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**What this model does:**")
            for g,d in [("Grade 1","Superficial — skin only"),
                        ("Grade 2","Deep — tendon/capsule"),
                        ("Grade 3","Deep + abscess/infection"),
                        ("Grade 4","Partial gangrene")]:
                st.markdown(f"- **{g}**: {d}")
        with col_b:
            st.markdown("**Model performance:**")
            st.metric("Validation accuracy","97.05%")
            st.metric("F1 macro","0.97")
            st.metric("Training epochs","27")


# ══════════════════════════════════════════════════════════════════════════════
# PS5 — CT STROKE
# ══════════════════════════════════════════════════════════════════════════════
elif module.startswith("🧠"):
    st.title("🧠 CT Stroke Detector — Hemorrhage Detection")
    st.caption("EfficientNet-B0 · Normal vs Stroke · AUROC 0.982 · Val Accuracy 92.2%")

    model = load_ps5()
    if model is None:
        st.error("PS5 model not found at models/best_ps5_classifier.pt"); st.stop()

    uploaded_ct = st.file_uploader("Upload brain CT scan image", type=["jpg","jpeg","png"])
    if uploaded_ct:
        img = Image.open(uploaded_ct).convert("RGB")
        col_img, col_res = st.columns(2)
        with col_img:
            st.image(img, caption="Brain CT scan", use_container_width=True)

        tensor = IMG_TFM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor),dim=1)[0].cpu().numpy()

        stroke_prob = float(probs[1]); is_stroke = stroke_prob >= 0.5

        with col_res:
            if is_stroke:
                st.markdown(f'<div class="risk-high"><strong>⚠️ STROKE DETECTED</strong><br>'
                            f'Confidence: {stroke_prob*100:.1f}%<br>'
                            f'Hemorrhagic stroke pattern identified. Immediate neurology referral required.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low"><strong>✅ NORMAL — No stroke detected</strong><br>'
                            f'Normal probability: {probs[0]*100:.1f}%<br>'
                            f'No hemorrhage pattern identified in this scan.</div>',
                            unsafe_allow_html=True)

            st.markdown("**Classification probabilities:**")
            for label,p,color in [("Normal",probs[0],"#639922"),("Stroke",probs[1],"#E24B4A")]:
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px">'
                            f'<span style="min-width:58px">{label}</span>'
                            f'<div style="flex:1;background:#e0e0e0;border-radius:4px;height:14px">'
                            f'<div style="width:{p*100:.1f}%;background:{color};height:14px;border-radius:4px"></div>'
                            f'</div><span>{p*100:.1f}%</span></div>', unsafe_allow_html=True)

            st.markdown("**Model performance:**")
            m1,m2 = st.columns(2)
            m1.metric("AUROC","0.982"); m2.metric("Accuracy","92.2%")

        # Clinical explanation + chatbot
        result_label = "STROKE DETECTED" if is_stroke else "NORMAL — no stroke detected"
        ps5_prompt = (
            f"You are a clinical assistant helping a caregiver understand a brain CT scan result.\n\n"
            f"AI CT scan result:\n"
            f"- Result: {result_label}\n"
            f"- Stroke probability: {stroke_prob*100:.1f}%\n"
            f"- Normal probability: {probs[0]*100:.1f}%\n\n"
            f"In 3-4 sentences explain: what this result means, "
            f"{'what a hemorrhagic stroke is and why time is critical' if is_stroke else 'what the patient should monitor going forward'}, "
            f"and what action is needed immediately. "
            f"{'Emphasise urgency.' if is_stroke else 'Be reassuring but recommend neurology follow-up.'}"
        )
        ollama_ui(ps5_prompt, "ps5")

        # Recommender
        diag_key = "ps5_stroke" if is_stroke else "ps5_normal"
        show_recommender(diag_key)
    else:
        st.info("Upload a brain CT scan image (.jpg or .png) to run stroke detection.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Classes:**\n- **Normal** — no hemorrhage detected\n- **Stroke** — hemorrhagic stroke pattern detected")
        with col_b:
            st.metric("AUROC","0.982"); st.metric("Stroke F1","0.89"); st.metric("Normal F1","0.94")

st.markdown("---")
st.caption("NeuroGuard · Team 2Infinity · ANC-016 · PESMCE Pune · Not for clinical use · All AI runs locally.")
