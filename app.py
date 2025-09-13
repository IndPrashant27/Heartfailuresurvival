# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw
from pathlib import Path

# Prepare circular favicon
favicon_path = "assets/heart_favicon.png"
if not Path(favicon_path).exists():  # generate once
    img = Image.open("assets/heart.jpg").convert("RGBA")
    size = (256, 256)
    img = img.resize(size, Image.Resampling.LANCZOS)

    # Create circle mask
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    img.putalpha(mask)

    img.save(favicon_path)

# Now use this as page icon
st.set_page_config(
    page_title="Heart Failure Survival Risk – Pro",
    page_icon=favicon_path,
    layout="wide"
)


# --- Simple theme toggle ---
if "dark" not in st.session_state:
    st.session_state.dark = False
with st.sidebar:
    st.checkbox("🌙 Dark mode", key="dark")

# --- CSS (dark/light, card styles, hover, zoom) ---
PRIMARY = "#cc0000"
CSS = f"""
<style>
:root {{
  --bg: {"#0f1218" if st.session_state.dark else "#ffffff"};
  --text: {"#e6eef8" if st.session_state.dark else "#111827"};
  --muted: {"#aab4c0" if st.session_state.dark else "#4b5563"};
  --card: {"#161b22" if st.session_state.dark else "#f9fafb"};
  --border: {"#2a313b" if st.session_state.dark else "#e5e7eb"};
}}
html, body, [data-testid="stAppViewContainer"] {{ background: var(--bg); color: var(--text); }}
h1,h2,h3,h4,h5,h6, p, span, div {{ color: var(--text); }}
.stMarkdown a {{ color: {PRIMARY}; text-decoration: none; }}

.hero {{
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(204,0,0,.08), transparent);
  border-radius: 18px; padding: 22px; margin-bottom: 10px;
  box-shadow: 0 6px 24px rgba(0,0,0,.08);
}}
.card {{
  border: 1px solid var(--border); background: var(--card); border-radius: 16px;
  padding: 16px; transition: transform .12s ease, box-shadow .12s ease; height:100%;
}}
.card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,0,0,.18); }}
.small {{ color: var(--muted); font-size: 0.92rem; }}
hr {{ border-color: var(--border); }}
.img-zoom img {{ border-radius:14px; width:100%; height:210px; object-fit:cover; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Paths ----------
ART_DIR = Path("models/artifacts")
MODEL_PATH = ART_DIR / "model_best.pkl"
METRICS_PATH = ART_DIR / "metrics.json"
EXPL_PATH = ART_DIR / "shap_explainer.pkl"

# ---------- Hero ----------
st.markdown('<div class="hero">', unsafe_allow_html=True)

col1, col2 = st.columns([0.15, 0.85])

with col1:
    st.image("assets/heart.jpg", width=120)   # <-- bigger, clearer heart image

with col2:
    st.markdown(
        "## Heart Failure Survival Risk – Pro\n"
        "<span class='small'>Predict survival probability with explainability and benchmarks. "
        "This demo is for education; not for clinical use.</span>",
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)


# ---------- Sidebar inputs ----------
st.sidebar.header("Patient Inputs")
def number(label, minv, maxv, val, step=1, key=None, fmt=None):
    return st.sidebar.number_input(label, min_value=minv, max_value=maxv, value=val, step=step, key=key, format=fmt)

age          = int(number("Age", 18, 120, 60, step=1, key="age"))
anaemia      = int(number("Anaemia (0/1)", 0, 1, 0, step=1, key="anaemia"))
cpk          = int(number("Creatinine Phosphokinase", 0, 8000, 600, step=10, key="cpk"))
diabetes     = int(number("Diabetes (0/1)", 0, 1, 0, step=1, key="diabetes"))
ef           = int(number("Ejection Fraction", 5, 80, 35, step=1, key="ef"))
hbp          = int(number("High Blood Pressure (0/1)", 0, 1, 0, step=1, key="hbp"))
platelets    = int(number("Platelets", 25_000, 900_000, 260_000, step=1_000, key="platelets", fmt="%d"))
serum_creat  = float(number("Serum Creatinine", 0.0, 10.0, 1.0, step=0.1, key="serum_creat"))
serum_na     = int(number("Serum Sodium", 100, 160, 137, step=1, key="serum_na"))
sex          = int(number("Sex (0=female, 1=male)", 0, 1, 1, step=1, key="sex"))
smoking      = int(number("Smoking (0/1)", 0, 1, 0, step=1, key="smoking"))
time_days    = int(number("Follow-up time (days)", 1, 300, 80, step=1, key="time_days"))

features = ["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction",
            "high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking","time"]
X_input = pd.DataFrame(
    [[age, anaemia, cpk, diabetes, ef, hbp, platelets, serum_creat, serum_na, sex, smoking, time_days]],
    columns=features
)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load(MODEL_PATH)
    metrics = json.loads(Path(METRICS_PATH).read_text()) if METRICS_PATH.exists() else {}
    shap_bundle = joblib.load(EXPL_PATH) if EXPL_PATH.exists() else None
    return pipe, metrics, shap_bundle
pipe, metrics, shap_bundle = load_artifacts()

# ---------- Prediction + Benchmarks ----------
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")
    prob = float(pipe.predict_proba(X_input)[:, 1][0])
    st.metric("Survival Risk (probability of death event)", f"{prob:.2%}")
    st.progress(min(1.0, prob))
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Benchmarks (CV)")
    if metrics:
        try:
            rows = []
            by_model = metrics.get("by_model", {})
            for name, m in by_model.items():
                rows.append({
                    "model": name,
                    "accuracy": round(m.get("accuracy", 0.0), 4),
                    "precision": round(m.get("precision", 0.0), 4),
                    "recall": round(m.get("recall", 0.0), 4),
                    "f1": round(m.get("f1", 0.0), 4),
                    "roc_auc": round(m.get("roc_auc", 0.0), 4),
                })
            dfm = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
            st.dataframe(dfm, use_container_width=True)
            st.caption(f"Best model: **{metrics.get('best_model','?')}**")
        except Exception:
            st.json(metrics)
    else:
        st.info("Run train.py first to generate metrics and artifacts.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------- Explainability ----------
st.subheader("Explainability (SHAP)")
st.markdown('<div class="card">', unsafe_allow_html=True)

def _pick_base_value(explainer):
    bv = getattr(explainer, "expected_value", None)
    if bv is None:
        bv = getattr(explainer, "expected_values", None)
    if isinstance(bv, (list, tuple, np.ndarray)):
        arr = np.asarray(bv).flatten()
        return float(arr[-1]) if arr.size > 1 else float(arr[0])
    return float(bv) if bv is not None else 0.0

def _to_feature_vector(sv, n_features):
    """Coerce SHAP outputs to a 1-D vector (positive class if available)."""
    a = np.asarray(sv)

    # Common cases
    if a.ndim == 1 and a.size == n_features:
        return a  # (n_features,)
    if a.ndim == 2:
        # (1, n_features)
        if a.shape == (1, n_features):
            return a[0]
        # (n_features, 1)  or (n_features, 2) -> pick last column (positive class)
        if a.shape[0] == n_features and a.shape[1] in (1, 2):
            return a[:, -1]
        # Sometimes libraries use (2, n_features) -> pick last row
        if a.shape[1] == n_features and a.shape[0] in (1, 2):
            return a[-1]
    if a.ndim == 3:
        # (1, n_features, 1/2) -> take the last class
        if a.shape[0] == 1 and a.shape[1] == n_features:
            return a[0, :, -1]

    raise ValueError(f"Unexpected SHAP shape {a.shape}; cannot plot.")

try:
    import shap
    if shap_bundle:
        explainer = shap_bundle["explainer"]
        X_trans = pipe.named_steps["prep"].transform(X_input)

        # Get SHAP values from old/new APIs
        if hasattr(explainer, "shap_values"):
            sv = explainer.shap_values(X_trans)    # may return list or array
            # If list of classes, take positive class if present
            if isinstance(sv, list):
                sv = sv[-1] if len(sv) > 1 else sv[0]
        else:
            sv = explainer(X_trans).values

        shap_vec = _to_feature_vector(sv, n_features=len(features))
        base_val = _pick_base_value(explainer)

        explanation = shap.Explanation(
            values=shap_vec,
            base_values=base_val,
            data=X_input.values[0],
            feature_names=features
        )

        fig, _ = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)
    else:
        st.info("SHAP explainer not found. Run train.py to generate it.")
except Exception as e:
    st.warning(f"SHAP visualization unavailable: {e}")

st.caption("Disclaimer: Educational demo. Not for clinical use.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Healthy Habits ----------
st.markdown("<hr/>", unsafe_allow_html=True)
st.header("Live healthier with your heart ❤️‍🩹")
st.write("Below are practical wellness pointers. **Not medical advice** — always consult a clinician.")

def card(title, img_path, bullets):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"#### {title}")
    st.image(img_path, use_container_width=True)
    st.markdown(bullets)
    st.markdown("</div>", unsafe_allow_html=True)

IMG_FOODS  = "assets/foods.jpg"
IMG_FRUITS = "assets/berries.jpg"
IMG_MOVE   = "assets/move.jpg"
IMG_SODIUM = "assets/salt.jpg"
IMG_LIFEST = "assets/yoga.jpg"

colA, colB, colC = st.columns(3)
with colA:
    card("Heart-Healthy Foods", IMG_FOODS,
         "- Oats, whole grains, legumes, nuts, seeds\n- Healthy fats: olive oil, fatty fish\n- Leafy greens & veggies\n- Limit: trans fats & processed snacks")
with colB:
    card("Smart Fruits", IMG_FRUITS,
         "- Berries: fiber & polyphenols\n- Citrus: vitamin C & potassium\n- Apples & pears: soluble fiber\n- Prefer whole fruit over juice")
with colC:
    card("Move More", IMG_MOVE,
         "- ~150 min/week moderate cardio\n- 2×/week strength training\n- Brisk walks, cycling, swimming\n- Break up long sitting")

colD, colE = st.columns([1,1])
with colD:
    card("Sodium & BP Tips", IMG_SODIUM,
         "- Cook at home; check sodium labels\n- Use herbs & spices over salt\n- Discuss BP targets with doctor")
with colE:
    card("Lifestyle Anchors", IMG_LIFEST,
         "- Sleep 7–9 hrs\n- Don’t smoke; limit alcohol\n- Stress mgmt: mindfulness, breathing")

st.markdown("<p class='small'>Informational only, not a substitute for medical advice.</p>", unsafe_allow_html=True)
