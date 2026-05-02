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

def render_ps5_page(model, ollama_ui, show_recommender, shimmer_metrics, shimmer_content):
    st.title("🧠 CT Stroke Detector — Hemorrhage Detection")
    st.caption("EfficientNet-B0 · Normal vs Stroke · AUROC 0.982 · Val Accuracy 92.2%")

    if model is None:
        st.error("PS5 model not found at models/best_ps5_classifier.pt")
        st.stop()

    uploaded_ct = st.file_uploader("Upload brain CT scan image", type=["jpg", "jpeg", "png"])
    if uploaded_ct:
        img = Image.open(uploaded_ct).convert("RGB")
        col_img, col_res = st.columns(2)
        with col_img:
            st.image(img, caption="Brain CT scan", use_container_width=True)

        tensor = IMG_TFM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

        stroke_prob = float(probs[1])
        is_stroke = stroke_prob >= 0.5

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
            for label, p, color in [("Normal", probs[0], "#639922"), ("Stroke", probs[1], "#E24B4A")]:
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px">'
                            f'<span style="min-width:58px">{label}</span>'
                            f'<div style="flex:1;background:#e0e0e0;border-radius:4px;height:14px">'
                            f'<div style="width:{p*100:.1f}%;background:{color};height:14px;border-radius:4px"></div>'
                            f'</div><span>{p*100:.1f}%</span></div>', unsafe_allow_html=True)

            st.markdown("**Model performance:**")
            m1, m2 = st.columns(2)
            m1.metric("AUROC", "0.982")
            m2.metric("Accuracy", "92.2%")

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
            st.metric("AUROC", "0.982")
            st.metric("Stroke F1", "0.89")
            st.metric("Normal F1", "0.94")
