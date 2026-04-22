import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
import io
import time
import os

# ─────────────────────────────────────────────
#  MODEL PATH
#  Place liver_best_model.pth inside a "models"
#  folder next to app.py
# ─────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "models", "liver_best_model.pth")


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Liver Tumor Segmentation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp {
    background: #0f1117;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #141824 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
}
div[data-testid="metric-container"] {
    background: #1a2035;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px !important;
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
.card {
    background: #141824;
    border: 1px solid #1e2535;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 4px solid #3b82f6;
    padding-left: 10px;
    margin-bottom: 16px;
}
.badge-green  { background:#16a34a22; color:#4ade80; border:1px solid #16a34a; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.badge-red    { background:#dc262622; color:#f87171; border:1px solid #dc2626; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.badge-blue   { background:#2563eb22; color:#60a5fa; border:1px solid #2563eb; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.summary-row {
    display:flex; justify-content:space-between; align-items:center;
    padding: 8px 0;
    border-bottom: 1px solid #1e2535;
    color: #cbd5e1;
    font-size: 0.9rem;
}
.dot-green  { width:10px; height:10px; border-radius:50%; background:#4ade80; display:inline-block; margin-right:8px; }
.dot-red    { width:10px; height:10px; border-radius:50%; background:#f87171; display:inline-block; margin-right:8px; }
.dot-blue   { width:10px; height:10px; border-radius:50%; background:#60a5fa; display:inline-block; margin-right:8px; }
[data-testid="stFileUploaderDropzone"] {
    background: #141824 !important;
    border: 2px dashed #2563eb !important;
    border-radius: 12px !important;
}
.stDownloadButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    margin-bottom: 8px;
}
.top-header h1 {
    color: #f1f5f9;
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 2px;
}
.top-header p {
    color: #94a3b8;
    font-size: 0.9rem;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0f1117; }
::-webkit-scrollbar-thumb { background:#2563eb55; border-radius:10px; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_from_bytes(model_bytes: bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=1, classes=3)
    buffer = io.BytesIO(model_bytes)
    model.load_state_dict(torch.load(buffer, map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=1, classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# ─────────────────────────────────────────────
#  AUTO-LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────
if "model" not in st.session_state:
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            m, d = load_model_from_path(DEFAULT_MODEL_PATH)
            st.session_state.model  = m
            st.session_state.device = d
        except Exception as e:
            st.session_state.model_load_error = str(e)


# ─────────────────────────────────────────────
#  PREDICTION HELPERS
# ─────────────────────────────────────────────
def predict(model, device, img_gray_256: np.ndarray):
    x = img_gray_256 / 255.0
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    mask  = np.argmax(probs, axis=0)
    return mask, probs

def make_colored_masks(img_gray, mask):
    h, w = mask.shape
    black = np.zeros((h, w, 3), dtype=np.uint8)
    liver_mask = black.copy()
    liver_mask[mask == 1] = [0, 200, 80]
    tumor_mask = black.copy()
    tumor_mask[mask == 2] = [220, 40, 40]
    base_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay  = base_bgr.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.4 + np.array([0, 180, 70])  * 0.6).astype(np.uint8)
    overlay[mask == 2] = (overlay[mask == 2] * 0.3 + np.array([220, 40, 40]) * 0.7).astype(np.uint8)
    return liver_mask, tumor_mask, overlay

def gray_to_pil(arr):
    return Image.fromarray(arr)

def bgr_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def compute_confidence(probs, mask):
    liver_conf = float(probs[1][mask == 1].mean()) if (mask == 1).any() else 0.0
    tumor_conf = float(probs[2][mask == 2].mean()) if (mask == 2).any() else 0.0
    bg_conf    = float(probs[0][mask == 0].mean()) if (mask == 0).any() else 0.0
    overall    = float(np.max(probs, axis=0).mean())
    return liver_conf, tumor_conf, bg_conf, overall

def img_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def make_donut_chart(bg_pct, liver_pct, tumor_pct):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="none")
    sizes  = [bg_pct, liver_pct, tumor_pct]
    colors = ["#4b5563", "#4ade80", "#f87171"]
    wedge_props = {"width": 0.45, "edgecolor": "#0f1117", "linewidth": 2}
    ax.pie(sizes, colors=colors, wedgeprops=wedge_props, startangle=90, counterclock=False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    for i, (val, color) in enumerate(zip(sizes, colors)):
        if val > 1:
            angle = 90 - sum(sizes[:i]) - val / 2
            angle_rad = np.radians(angle)
            r = 0.78
            x = r * np.cos(angle_rad)
            y = r * np.sin(angle_rad)
            ax.text(x, y, f"{val:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:24px;padding:10px 0">
        <div style="background:#1d4ed8;border-radius:10px;width:40px;height:40px;
                    display:flex;align-items:center;justify-content:center;font-size:20px;">
            🛡️
        </div>
        <div>
            <div style="font-weight:800;font-size:1rem;color:#f1f5f9">Liver Tumor</div>
            <div style="font-size:0.75rem;color:#3b82f6;font-weight:600">Segmentation System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="color:#3b82f6;font-weight:700;font-size:0.82rem;margin-bottom:10px">
        Model Information
    </div>
    """, unsafe_allow_html=True)

    model_loaded = "model" in st.session_state

    st.markdown(f"""
    <div style="font-size:0.8rem;line-height:2">
        {'✅' if model_loaded else '⬜'} <span style="color:#94a3b8">Model: DeepLabV3+</span><br>
        {'✅' if model_loaded else '⬜'} <span style="color:#94a3b8">Backbone: ResNet34</span><br>
        {'✅' if model_loaded else '⬜'} <span style="color:#94a3b8">Classes: 3 (Background, Liver, Tumor)</span><br>
        {'✅' if model_loaded else '⏳'} <span style="color:{'#4ade80' if model_loaded else '#f59e0b'}">
            Status: {'Model Loaded ✅' if model_loaded else 'Not Loaded'}</span>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.divider()
        st.markdown("""
        <div style="background:#2d1a1a;border:1px solid #7f1d1d;border-radius:10px;padding:12px;margin-bottom:12px">
            <div style="color:#f87171;font-size:0.82rem;font-weight:600;margin-bottom:4px">⚠️ Model not found</div>
            <div style="color:#94a3b8;font-size:0.76rem">
                Auto-load failed. Upload your <code>.pth</code> file below,
                or place it at <code>models/liver_best_model.pth</code> next to app.py.
            </div>
        </div>
        """, unsafe_allow_html=True)

        pth_file = st.file_uploader("Upload model .pth", type=["pth", "pt"],
                                    label_visibility="collapsed")
        if pth_file is not None:
            if st.button("🔄 Load Model", use_container_width=True):
                with st.spinner("Loading model..."):
                    try:
                        m, d = load_model_from_bytes(pth_file.read())
                        st.session_state.model  = m
                        st.session_state.device = d
                        st.success("✅ Model loaded successfully!")
                        time.sleep(0.6)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")

    if "model_load_error" in st.session_state and not model_loaded:
        st.error(f"Auto-load error: {st.session_state.model_load_error}")


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <h1>Liver Tumor Segmentation System</h1>
    <p>Upload a CT scan image — DeepLabV3+ (ResNet34) model will detect Liver and Tumor regions.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  UPLOAD SECTION
# ─────────────────────────────────────────────
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload CT Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload grayscale CT scan (PNG / JPG / BMP)",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  RUN PREDICTION
# ─────────────────────────────────────────────
results = None

if uploaded_file is not None:
    if "model" not in st.session_state:
        st.warning("⚠️ Model is not loaded. Please upload the model file from the sidebar.")
    else:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img_256 = cv2.resize(raw, (256, 256))

        with st.spinner("Running segmentation..."):
            mask, probs = predict(st.session_state.model, st.session_state.device, img_256)
            liver_conf, tumor_conf, bg_conf, overall_conf = compute_confidence(probs, mask)

        liver_detected   = bool((mask == 1).any())
        tumor_detected   = bool((mask == 2).any())
        total_pixels     = mask.size
        tumor_pixels     = int((mask == 2).sum())
        liver_pixels     = int((mask == 1).sum())
        bg_pixels        = int((mask == 0).sum())
        tumor_percentage = tumor_pixels / total_pixels * 100
        liver_pct        = liver_pixels  / total_pixels * 100
        bg_pct           = bg_pixels     / total_pixels * 100

        liver_mask_img, tumor_mask_img, overlay_img = make_colored_masks(img_256, mask)

        results = {
            "img_256"       : img_256,
            "mask"          : mask,
            "liver_mask"    : liver_mask_img,
            "tumor_mask"    : tumor_mask_img,
            "overlay"       : overlay_img,
            "liver_detected": liver_detected,
            "tumor_detected": tumor_detected,
            "liver_conf"    : liver_conf,
            "tumor_conf"    : tumor_conf,
            "bg_conf"       : bg_conf,
            "overall_conf"  : overall_conf,
            "tumor_pixels"  : tumor_pixels,
            "tumor_pct"     : tumor_percentage,
            "liver_pct"     : liver_pct,
            "bg_pct"        : bg_pct,
        }
        st.session_state.results = results

if results is None and "results" in st.session_state:
    results = st.session_state.results


# ─────────────────────────────────────────────
#  STAT CARDS — 3 only (Overall Confidence removed)
# ─────────────────────────────────────────────
if results:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card" style="text-align:center">
            <div style="font-size:2rem">🖼️</div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px">Image Size</div>
            <div style="font-size:1.6rem;font-weight:800;color:#f1f5f9">256 × 256</div>
            <div style="color:#64748b;font-size:0.75rem">Pixels</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        ld_color = "#4ade80" if results["liver_detected"] else "#f87171"
        ld_text  = "Yes"     if results["liver_detected"] else "No"
        ld_conf  = results["liver_conf"] * 100
        st.markdown(f"""
        <div class="card" style="text-align:center;border-color:#16a34a44">
            <div style="font-size:2rem">🫀</div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px">Liver Detected</div>
            <div style="font-size:1.6rem;font-weight:800;color:{ld_color}">{ld_text}</div>
            <span class="badge-green">Confidence: {ld_conf:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        td_color = "#f87171" if results["tumor_detected"] else "#4ade80"
        td_text  = "Yes"     if results["tumor_detected"] else "No"
        td_conf  = results["tumor_conf"] * 100
        st.markdown(f"""
        <div class="card" style="text-align:center;border-color:#dc262644">
            <div style="font-size:2rem">🎯</div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:4px">Tumor Detected</div>
            <div style="font-size:1.6rem;font-weight:800;color:{td_color}">{td_text}</div>
            <span class="badge-red">Confidence: {td_conf:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Segmentation Results ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 Segmentation Results</div>', unsafe_allow_html=True)

    ic1, ic2, ic3, ic4 = st.columns(4)
    imgs = [
        ("Original Image",          gray_to_pil(results["img_256"])),
        ("Liver Mask",              bgr_to_pil(results["liver_mask"])),
        ("Tumor Mask",              bgr_to_pil(results["tumor_mask"])),
        ("Overlay (Liver + Tumor)", bgr_to_pil(results["overlay"])),
    ]
    for col, (title, pil_img) in zip([ic1, ic2, ic3, ic4], imgs):
        with col:
            st.markdown(f"<p style='color:#94a3b8;font-size:0.82rem;text-align:center;margin-bottom:6px'>{title}</p>",
                        unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Bottom Row ──
    bot1, bot2, bot3 = st.columns([1.4, 1.4, 1.2])

    with bot1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Prediction Summary</div>', unsafe_allow_html=True)

        liver_badge = '<span class="badge-green">Detected</span>'  if results["liver_detected"] else '<span class="badge-red">Not Found</span>'
        tumor_badge = '<span class="badge-red">Detected</span>'    if results["tumor_detected"] else '<span class="badge-green">Not Found</span>'

        st.markdown(f"""
        <div class="summary-row">
            <span><span class="dot-green"></span>Liver Region</span>
            {liver_badge}
        </div>
        <div class="summary-row">
            <span><span class="dot-red"></span>Tumor Region</span>
            {tumor_badge}
        </div>
        <div class="summary-row">
            <span><span class="dot-blue"></span>Background</span>
            <span class="badge-blue">Detected</span>
        </div>
        <div class="summary-row" style="border:none;margin-top:10px">
            <span style="color:#94a3b8">Total Tumor Area</span>
            <span style="color:#f1f5f9;font-weight:600">{results['tumor_pixels']:,} pixels</span>
        </div>
        <div class="summary-row" style="border:none">
            <span style="color:#94a3b8">Tumor Percentage</span>
            <span style="color:#f87171;font-weight:600">{results['tumor_pct']:.2f} %</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bot2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🥧 Class Distribution</div>', unsafe_allow_html=True)

        fig = make_donut_chart(results["bg_pct"], results["liver_pct"], results["tumor_pct"])
        col_d, col_legend = st.columns([1.2, 1])
        with col_d:
            st.pyplot(fig, use_container_width=True)
        with col_legend:
            st.markdown("""
            <div style="margin-top:30px;font-size:0.8rem;line-height:2.2">
                <span class="dot-green"></span>
                <span style="color:#94a3b8">Liver (1)</span><br>
                <span class="dot-red"></span>
                <span style="color:#94a3b8">Tumor (2)</span><br>
                <span style="width:10px;height:10px;border-radius:2px;
                       background:#4b5563;display:inline-block;margin-right:8px"></span>
                <span style="color:#94a3b8">Background (0)</span>
            </div>
            """, unsafe_allow_html=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with bot3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚡ Actions</div>', unsafe_allow_html=True)

        mask_pil    = bgr_to_pil(results["tumor_mask"])
        overlay_pil = bgr_to_pil(results["overlay"])

        st.download_button(
            "⬇️  Download Mask",
            data=img_to_bytes(mask_pil),
            file_name="tumor_mask.png",
            mime="image/png",
            use_container_width=True,
        )
        st.download_button(
            "🖼️  Download Overlay",
            data=img_to_bytes(overlay_pil),
            file_name="overlay.png",
            mime="image/png",
            use_container_width=True,
        )

        report_text = f"""
LIVER TUMOR SEGMENTATION REPORT
================================
Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}
Model       : DeepLabV3+  (ResNet34)
Image Size  : 256 x 256 px

DETECTION RESULTS
-----------------
Liver Detected   : {'Yes' if results['liver_detected'] else 'No'}
Liver Confidence : {results['liver_conf']*100:.2f}%

Tumor Detected   : {'Yes' if results['tumor_detected'] else 'No'}
Tumor Confidence : {results['tumor_conf']*100:.2f}%
Overall Conf     : {results['overall_conf']*100:.2f}%

PIXEL STATISTICS
----------------
Total Tumor Area : {results['tumor_pixels']:,} pixels
Tumor %          : {results['tumor_pct']:.2f}%
Liver %          : {results['liver_pct']:.2f}%
Background %     : {results['bg_pct']:.2f}%
""".strip()

        st.download_button(
            "📄  Download Report (TXT)",
            data=report_text.encode(),
            file_name="segmentation_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if st.button("🗑️  Clear Results", use_container_width=True):
            if "results" in st.session_state:
                del st.session_state.results
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:60px 20px">
        <div style="font-size:4rem;margin-bottom:16px">🫀</div>
        <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-bottom:8px">
            No Scan Uploaded Yet
        </div>
        <div style="color:#64748b;max-width:400px;margin:auto">
            Upload a CT scan image above to run liver and tumor segmentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.75rem;padding:20px 0 8px">
    © 2026 Liver Tumor Segmentation System &nbsp;|&nbsp; All Rights Reserved
    &nbsp;&nbsp;
    <span style="color:#ef4444">❤️</span> Developed for better healthcare
</div>
""", unsafe_allow_html=True)