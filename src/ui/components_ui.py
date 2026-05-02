import streamlit as st

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
