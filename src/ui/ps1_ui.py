import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
from src.config import DEVICE

IMG_TFM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def render_ps1_page(model, validate_foot_wound_image, ollama_ui, show_recommender, shimmer_metrics, shimmer_content):
    st.title("🦶 Foot Wound Grader — Diabetic Ulcer Classification")
    st.caption("EfficientNet-B0 · Wagner Grade 1–4 · Accuracy 97.05%")

    if model is None:
        st.error("PS1 model not found at models/best_ps1.pt")
        st.stop()

    GRADE_INFO = {
        0: ("Grade 1", "Superficial wound (skin only)", "low",
            "Monitor and offload pressure. Podiatry review recommended."),
        1: ("Grade 2", "Deep wound to tendon or joint capsule", "moderate",
            "Refer to podiatry/diabetology within 1 week."),
        2: ("Grade 3", "Deep wound with abscess or osteomyelitis", "high",
            "Urgent surgical review required within 24 hours."),
        3: ("Grade 4", "Partial foot gangrene", "high",
            "Immediate vascular surgery referral — amputation risk."),
    }

    uploaded_img = st.file_uploader("Upload foot wound photograph", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        col_img, col_res = st.columns(2)
        with col_img:
            st.image(img, caption="Uploaded image", use_container_width=True)

        # ── Validate: is this actually a foot wound image? ──────────────
        file_id = f"{uploaded_img.name}_{uploaded_img.size}"
        if st.session_state.get("ps1_validated_file") != file_id:
            with st.spinner("🔍 Validating image — checking if this is a foot wound…"):
                is_valid, val_msg = validate_foot_wound_image(img)
            st.session_state["ps1_validated_file"] = file_id
            st.session_state["ps1_valid"] = is_valid
            st.session_state["ps1_val_msg"] = val_msg
        
        is_valid = st.session_state["ps1_valid"]
        val_msg  = st.session_state["ps1_val_msg"]

        if val_msg == "clip_unavailable":
            st.warning(
                "⚠️ CLIP model could not be loaded for image validation. "
                "Run: `pip install transformers` to enable it."
            )
        if not is_valid:
            st.error(
                "**Invalid image uploaded.** This does not appear to be a foot wound photograph.\n\n"
                f"**Reason:** {val_msg}\n\n"
                "Please upload a clear photograph of a diabetic foot wound for grading."
            )
            st.stop()
        # ────────────────────────────────────────────────────────────────

        tensor = IMG_TFM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred]) * 100
        g_name, g_desc, g_risk, g_action = GRADE_INFO[pred]

        with col_res:
            css = "risk-high" if g_risk == "high" else "risk-mod" if g_risk == "moderate" else "risk-low"
            st.markdown(f'<div class="{css}"><strong>{g_name} — {g_desc}</strong><br>'
                        f'Confidence: {conf:.1f}%<br>Action: {g_action}</div>',
                        unsafe_allow_html=True)
            st.markdown("**Grade probabilities:**")
            for i, (g, p) in enumerate(zip(["Grade 1", "Grade 2", "Grade 3", "Grade 4"], probs)):
                bc = "#E24B4A" if i == pred else "#B4B2A9"
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
            for g, d in [("Grade 1", "Superficial — skin only"),
                         ("Grade 2", "Deep — tendon/capsule"),
                         ("Grade 3", "Deep + abscess/infection"),
                         ("Grade 4", "Partial gangrene")]:
                st.markdown(f"- **{g}**: {d}")
        with col_b:
            st.markdown("**Model performance:**")
            st.metric("Validation accuracy", "97.05%")
            st.metric("F1 macro", "0.97")
            st.metric("Training epochs", "27")
