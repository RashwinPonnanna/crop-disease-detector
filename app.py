"""
app.py  (CNN Version — Multiple Image Support)
───────────────────────────────────────────────
Image-Based Crop Disease Detection System
BCA Final Year Project
"""

import streamlit as st
import json
import time
from PIL import Image

from preprocessing import preprocess_image, extract_features
from model_handler import predict_disease
from utils import validate_image

st.set_page_config(
    page_title="🌿 Crop Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .hero-banner {
        background: linear-gradient(135deg, #1a3a2a 0%, #2d6a4f 50%, #52b788 100%);
        padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
        text-align: center; box-shadow: 0 8px 32px rgba(26,58,42,0.3);
    }
    .hero-banner h1 { font-family: 'Playfair Display', serif; color: #d8f3dc; font-size: 2.4rem; margin: 0; }
    .hero-banner p { color: rgba(216,243,220,0.75); margin-top:0.5rem; font-size:1rem; }
    .section-card {
        background:#fff; border:1px solid #e0ede5; border-radius:12px;
        padding:1.4rem 1.6rem; margin-bottom:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.05);
    }
    .section-card h3 { font-family:'Playfair Display',serif; color:#1a3a2a; margin-top:0; }
    .disease-badge {
        display:inline-block; background:linear-gradient(90deg,#e76f51,#f4a261);
        color:white; padding:0.4rem 1.1rem; border-radius:99px; font-weight:500; font-size:0.95rem; margin-bottom:0.8rem;
    }
    .healthy-badge { background:linear-gradient(90deg,#2d6a4f,#52b788); }
    .confidence-bar-wrap { background:#e0ede5; border-radius:8px; height:12px; width:100%; margin-top:0.5rem; }
    .confidence-bar-fill { height:100%; border-radius:8px; background:linear-gradient(90deg,#52b788,#2d6a4f); }
    .pill-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.6rem; }
    .pill { background:#d8f3dc; color:#1a3a2a; border-radius:99px; padding:0.25rem 0.75rem; font-size:0.82rem; font-weight:500; }
    .metric-box { text-align:center; background:#f6fbf7; border:1px solid #c7e0ce; border-radius:10px; padding:1rem; }
    .metric-box .val { font-size:1.8rem; font-weight:700; color:#2d6a4f; }
    .metric-box .lbl { font-size:0.78rem; color:#555; margin-top:0.2rem; }
    .model-badge { display:inline-block; background:#e8f4fd; color:#1a5276; border:1px solid #aed6f1; border-radius:8px; padding:0.3rem 0.8rem; font-size:0.8rem; font-weight:500; }
    .image-divider { border: 2px dashed #c7e0ce; border-radius: 12px; padding: 1rem; margin: 1.5rem 0; background: #f6fbf7; }
    [data-testid="stSidebar"] { background:#f4faf6; border-right:1px solid #d8f3dc; }
    [data-testid="stFileUploader"] { border:2px dashed #52b788 !important; border-radius:12px !important; background:#f6fbf7 !important; }
    .stButton > button { background:linear-gradient(90deg,#2d6a4f,#52b788); color:white; border:none; border-radius:8px; padding:0.6rem 1.6rem; font-weight:500; font-size:0.95rem; }
    .stButton > button:hover { opacity:0.88; }
    .footer { text-align:center; color:#888; font-size:0.78rem; margin-top:3rem; padding-top:1rem; border-top:1px solid #e0ede5; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Crop Disease Detector")
    st.markdown("**BCA Final Year Project**")
    st.markdown('<div class="model-badge">🧠 MobileNetV2 CNN Model</div>', unsafe_allow_html=True)
    st.caption("Pre-trained on PlantVillage — CPU only, no GPU needed")
    st.divider()
    confidence_threshold = st.slider("Confidence Threshold (%)", min_value=40, max_value=100, value=60)
    st.divider()
    st.markdown("### 📋 Detectable Diseases")
    for d in ["🍅 Bacterial Spot","🍅 Early Blight","🍅 Late Blight","🍅 Leaf Mold",
              "🍅 Septoria Leaf Spot","🍅 Spider Mites","🍅 Target Spot",
              "🍅 Mosaic Virus","🍅 Yellow Leaf Curl","✅ Healthy Leaf"]:
        st.markdown(f"<small>{d}</small>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### ⚙️ Pipeline Steps")
    for i, s in enumerate(["Upload Image(s)","Validate","Preprocess","Extract Features","CNN Predict","Show Results"],1):
        st.markdown(
            f'<span style="display:inline-block;width:22px;height:22px;border-radius:50%;background:#2d6a4f;color:white;font-size:0.75rem;text-align:center;line-height:22px;margin-right:6px;">{i}</span><small>{s}</small>',
            unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🌿 Crop Disease Detection System</h1>
    <p>Upload one or more tomato leaf images · CNN classifies each · Get instant remedies</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────
st.markdown("### 📤 Upload Leaf Image(s)")
st.caption("Hold Ctrl while clicking to select multiple files at once")

uploaded_files = st.file_uploader(
    "Drag & drop or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="JPG, PNG, WEBP — max 10MB each"
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} image(s) uploaded:**")
    thumb_cols = st.columns(min(len(uploaded_files), 5))
    for i, (col, f) in enumerate(zip(thumb_cols, uploaded_files)):
        with col:
            img = Image.open(f)
            st.image(img, caption=f"Image {i+1}", use_container_width=True)
            f.seek(0)
    st.markdown("<br>", unsafe_allow_html=True)
    analyse_btn = st.button(f"🔬 Analyse {len(uploaded_files)} Image(s)", use_container_width=True)
else:
    st.info("👆 Upload one or more tomato leaf images above, then click Analyse.")
    analyse_btn = False

# ── Analysis ──────────────────────────────────────────────────
if uploaded_files and analyse_btn:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        st.markdown(
            f'<div class="image-divider"><strong>🖼️ Image {idx+1} of {len(uploaded_files)}: {uploaded_file.name}</strong></div>',
            unsafe_allow_html=True)

        uploaded_file.seek(0)
        image = Image.open(uploaded_file)

        with st.status(f"🔄 Analysing {uploaded_file.name}...", expanded=True) as status:
            st.write("**Step 1** — Validating image...")
            uploaded_file.seek(0)
            is_valid, msg = validate_image(uploaded_file)
            if not is_valid:
                st.error(f"❌ {msg}")
                status.update(label="❌ Validation failed", state="error")
                continue
            st.write("✅ Validated")

            st.write("**Step 2** — Preprocessing...")
            processed_img, prep_info = preprocess_image(image)
            st.write(f"✅ {prep_info['size']} | {prep_info['mode']}")

            st.write("**Step 3** — Extracting features...")
            features = extract_features(processed_img)
            st.write(f"✅ Brightness:{features['brightness']:.0f} | Green:{features['green_ratio']:.1f}% | Brown:{features['brown_ratio']:.1f}%")

            st.write("**Step 4** — Running CNN prediction...")
            result = predict_disease(processed_img, features)
            st.write(f"✅ **{result['disease_name']}** ({result['confidence']:.1f}%)")

            st.write("**Step 5** — Formatting results...")
            time.sleep(0.2)
            st.write("✅ Done")

            status.update(label=f"✅ {uploaded_file.name} → {result['disease_name']}", state="complete", expanded=False)

        # Results layout
        col_img, col_res = st.columns([1, 1.5], gap="large")

        with col_img:
            st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-box"><div class="val">{image.width}</div><div class="lbl">Width px</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-box"><div class="val">{image.height}</div><div class="lbl">Height px</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-box"><div class="val">{image.mode}</div><div class="lbl">Mode</div></div>', unsafe_allow_html=True)

        with col_res:
            st.markdown(f'<div class="model-badge" style="margin-bottom:1rem;">🧠 {result["method"]}</div>', unsafe_allow_html=True)

            is_healthy  = "healthy" in result["disease_name"].lower()
            badge_class = "healthy-badge" if is_healthy else "disease-badge"
            badge_icon  = "✅" if is_healthy else "⚠️"
            confidence  = result["confidence"]
            sev_colors  = {"Low":"#52b788","Medium":"#f4a261","High":"#e76f51","None":"#52b788"}
            sev_color   = sev_colors.get(result["severity"], "#888")

            st.markdown(f"""
            <div class="section-card">
                <h3>🎯 Detection Result</h3>
                <span class="disease-badge {badge_class}">{badge_icon} {result['disease_name']}</span>
                <p style="margin:0.3rem 0 0;font-size:0.85rem;color:#666;"><em>{result.get('scientific_name','N/A')}</em></p>
                <p style="margin:0.3rem 0;font-size:0.88rem;color:#555;">
                    Confidence: <strong>{confidence}%</strong> &nbsp;|&nbsp;
                    Severity: <strong style="color:{sev_color};">{result['severity']}</strong>
                </p>
                <div class="confidence-bar-wrap">
                    <div class="confidence-bar-fill" style="width:{min(confidence,100)}%;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            if confidence < confidence_threshold:
                st.warning(f"⚠️ Confidence ({confidence}%) below threshold ({confidence_threshold}%). Try a clearer image.")

            st.markdown(f"""
            <div class="section-card">
                <h3>🩺 Symptoms</h3>
                <p style="color:#333;line-height:1.75;font-size:0.9rem;">{result['symptoms']}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="section-card">
                <h3>🔬 Causes</h3>
                <p style="color:#333;line-height:1.75;font-size:0.9rem;">{result['causes']}</p>
            </div>""", unsafe_allow_html=True)

            remedy_pills = "".join(f'<span class="pill">💊 {r}</span>' for r in result["remedies"])
            st.markdown(f"""
            <div class="section-card">
                <h3>💊 Remedies</h3>
                <div class="pill-row">{remedy_pills}</div>
            </div>""", unsafe_allow_html=True)

            prev_pills = "".join(f'<span class="pill">🛡️ {p}</span>' for p in result["prevention"])
            st.markdown(f"""
            <div class="section-card">
                <h3>🛡️ Preventive Measures</h3>
                <div class="pill-row">{prev_pills}</div>
            </div>""", unsafe_allow_html=True)

            with st.expander("🔬 Preprocessed Image & Features"):
                st.image(processed_img, caption="After preprocessing (224×224)", use_container_width=True)
                st.json(features)

            report = {
                "image_filename":  uploaded_file.name,
                "disease_name":    result["disease_name"],
                "scientific_name": result.get("scientific_name","N/A"),
                "confidence_pct":  confidence,
                "severity":        result["severity"],
                "symptoms":        result["symptoms"],
                "causes":          result["causes"],
                "remedies":        result["remedies"],
                "prevention":      result["prevention"],
                "method":          result["method"],
                "image_features":  features,
            }
            st.download_button(
                label=f"📥 Download Report — {uploaded_file.name}",
                data=json.dumps(report, indent=2),
                file_name=f"report_{uploaded_file.name}.json",
                mime="application/json",
                use_container_width=True,
                key=f"download_btn_{idx}"
            )

elif not uploaded_files:
    st.markdown("---")
    st.markdown("""
    <div class="section-card" style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:3rem;">🍃</div>
        <h3 style="color:#2d6a4f;">Ready to Detect</h3>
        <p style="color:#888;">Upload one or more tomato leaf images above<br>then click <strong>Analyse Image(s)</strong>.</p>
    </div>""", unsafe_allow_html=True)

# ── Pipeline Explainer ────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔄 How the CNN Pipeline Works")
cols = st.columns(5)
steps = [
    ("📤","1. Upload","Upload one or multiple leaf photos"),
    ("🖼️","2. Preprocess","Resize, denoise & enhance"),
    ("🔍","3. Extract","Colour & texture features"),
    ("🧠","4. CNN Model","MobileNetV2 classifies disease"),
    ("📋","5. Results","Full info shown per image"),
]
for col, (icon, title, desc) in zip(cols, steps):
    col.markdown(f"""
    <div class="section-card" style="text-align:center;">
        <div style="font-size:1.8rem;">{icon}</div>
        <strong style="color:#1a3a2a;">{title}</strong>
        <p style="font-size:0.78rem;color:#666;margin-top:0.4rem;">{desc}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    🌿 Crop Disease Detection System &nbsp;|&nbsp; BCA Final Year Project &nbsp;|&nbsp;
    MobileNetV2 CNN · PlantVillage Dataset &nbsp;|&nbsp; Streamlit · Python
</div>
""", unsafe_allow_html=True)