import streamlit as st

def render_home_page():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown('''
    <div style="margin-bottom:8px">
        <h1 style="font-size:34px;font-weight:800;color:#0F172A;margin:0">
            Sentin<span style="color:#0EA5E9">Al</span>
        </h1>
        <p style="font-size:14px;color:#64748B;margin:6px 0 0">
            Your Personal Healthcare Companion &mdash; Privacy-first edge AI &middot;
            instant diagnosis &middot; zero cloud data transfer
        </p>
    </div>''', unsafe_allow_html=True)

    # ── Patient-centric Feature cards ──────────────────────────────────────────
    st.markdown('''
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0 32px">
      <div class="metric-card mc-ps2">
        <div class="label">Continuous Monitoring</div>
        <div class="value">24/7</div>
        <div class="badge">&#8599; Stay Safe</div>
      </div>
      <div class="metric-card mc-ps1">
        <div class="label">Data Privacy</div>
        <div class="value">100%</div>
        <div class="badge">&#9673; Local Processing</div>
      </div>
      <div class="metric-card mc-ps5">
        <div class="label">Diagnostic Speed</div>
        <div class="value">&lt; 1s</div>
        <div class="badge">&#10022; Instant Results</div>
      </div>
      <div class="metric-card mc-modules">
        <div class="label">Healthcare Solutions</div>
        <div class="value">3</div>
        <div class="badge">&#10022; And growing...</div>
      </div>
    </div>''', unsafe_allow_html=True)

    # ── Module cards (clickable — each card is an <a> link) ─────────────────
    st.markdown('''
    <style>.card-link{text-decoration:none!important;color:inherit!important;display:block}</style>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:28px">
      <a class="card-link" href="?nav=ps2">
        <div class="module-card mc-vital">
          <span class="tag">Solution</span>
          <div class="icon">&#128200;</div>
          <h3>Vital Sign Monitor</h3>
          <div class="desc">Continuous clinical monitoring with early warning alerts. Protect yourself or your loved ones with early detection.</div>
          <div style="margin-top:20px;padding-top:18px;border-top:1px solid #F1F5F9;font-weight:bold;color:#0284C7">Launch Monitor &rarr;</div>
        </div>
      </a>
      <a class="card-link" href="?nav=ps5">
        <div class="module-card mc-stroke">
          <span class="tag">Solution</span>
          <div class="icon">&#129504;</div>
          <h3>CT Stroke Detector</h3>
          <div class="desc">Automated identification of strokes from CT scans. Get immediate insights when every second counts.</div>
          <div style="margin-top:20px;padding-top:18px;border-top:1px solid #F1F5F9;font-weight:bold;color:#059669">Scan Now &rarr;</div>
        </div>
      </a>
      <a class="card-link" href="?nav=ps1">
        <div class="module-card mc-wound">
          <span class="tag">Solution</span>
          <div class="icon">&#129470;</div>
          <h3>Foot Wound Grader</h3>
          <div class="desc">Upload a photo of a foot wound for instant assessment and next-step recommendations.</div>
          <div style="margin-top:20px;padding-top:18px;border-top:1px solid #F1F5F9;font-weight:bold;color:#E11D48">Assess Wound &rarr;</div>
        </div>
      </a>
    </div>''', unsafe_allow_html=True)

    # ── Specialist recommender banner ─────────────────────────────────────────
    st.markdown('''
    <div class="recommender-card">
      <div class="rec-icon">&#128154;</div>
      <div class="rec-info">
        <h3>Need a Doctor?</h3>
        <div class="rec-desc">
          Our integrated recommender suggests optimal clinical intervention paths and finds the best specialist referrals near you.
        </div>
      </div>
      <div class="rec-right">
        <div class="rec-badge-box">
          <div class="rb-label">Availability</div>
          <div class="rb-value">Available in all solutions</div>
        </div>
      </div>
    </div>''', unsafe_allow_html=True)
