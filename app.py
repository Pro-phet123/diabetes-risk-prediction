# ============================================================
# 🩺 DIABETES RISK INTELLIGENCE
# Premium Responsive Streamlit Clinical Screening Dashboard
#
# Author:
# Olalemi Olaoluwakintan Emmanuel
#
# Project:
# Diabetes Risk Prediction Using Machine Learning
#
# NOTE:
# This application is an educational/research prototype.
# It is NOT a medical diagnostic system.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "diabetes_risk_prediction_model.pkl"
)

DATA_PATH = (
    BASE_DIR /
    "diabetes_nan.csv"
)


# ============================================================
# EXPECTED MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


# ============================================================
# PROJECT METRICS
# ============================================================

BASELINE_METRICS = {
    "Accuracy": 0.7013,
    "Precision": 0.5870,
    "Recall": 0.5000,
    "F1 Score": 0.5400
}


OPTIMIZED_METRICS = {
    "Accuracy": 0.6883,
    "Precision": 0.5395,
    "Recall": 0.7593,
    "F1 Score": 0.6308,
    "ROC-AUC": 0.7869
}


CONFUSION_MATRIX = {
    "TN": 65,
    "FP": 35,
    "FN": 13,
    "TP": 41
}


# ============================================================
# FEATURE COEFFICIENTS
# ============================================================

FEATURE_COEFFICIENTS = {
    "Glucose": 0.156972,
    "BMI": 0.102375,
    "SkinThickness": 0.062338,
    "Insulin": 0.058959,
    "Age": 0.058537,
    "Pregnancies": 0.053139,
    "BloodPressure": 0.045498,
    "DiabetesPedigreeFunction": 0.043508
}


# ============================================================
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:

    st.session_state.theme = "Dark"


if "assessment_result" not in st.session_state:

    st.session_state.assessment_result = None


# ============================================================
# THEME DEFINITIONS
# ============================================================

THEMES = {

    "Dark": {
        "background": "#050505",
        "surface": "#0b0b0d",
        "surface2": "#111114",
        "card": "rgba(17, 17, 20, 0.88)",
        "border": "rgba(255,255,255,0.09)",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "primary": "#60a5fa",
        "secondary": "#a78bfa",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#fb7185",
        "hero1": "#050505",
        "hero2": "#111827",
        "hero3": "#172554"
    },

    "Midnight Blue": {
        "background": "#020617",
        "surface": "#0f172a",
        "surface2": "#172033",
        "card": "rgba(15,23,42,0.88)",
        "border": "rgba(148,163,184,0.12)",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "primary": "#38bdf8",
        "secondary": "#818cf8",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#fb7185",
        "hero1": "#020617",
        "hero2": "#0f172a",
        "hero3": "#1e3a8a"
    },

    "Emerald": {
        "background": "#020807",
        "surface": "#071311",
        "surface2": "#0c1d19",
        "card": "rgba(7,19,17,0.90)",
        "border": "rgba(52,211,153,0.12)",
        "text": "#ecfdf5",
        "muted": "#94a3b8",
        "primary": "#34d399",
        "secondary": "#2dd4bf",
        "success": "#4ade80",
        "warning": "#fbbf24",
        "danger": "#fb7185",
        "hero1": "#020807",
        "hero2": "#071311",
        "hero3": "#064e3b"
    },

    "Light": {
        "background": "#f5f7fb",
        "surface": "#ffffff",
        "surface2": "#f8fafc",
        "card": "rgba(255,255,255,0.92)",
        "border": "rgba(15,23,42,0.08)",
        "text": "#0f172a",
        "muted": "#64748b",
        "primary": "#2563eb",
        "secondary": "#7c3aed",
        "success": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626",
        "hero1": "#ffffff",
        "hero2": "#eff6ff",
        "hero3": "#dbeafe"
    }
}


# ============================================================
# CURRENT THEME
# ============================================================

theme = THEMES[
    st.session_state.theme
]


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       ROOT
       ======================================================== */

    :root {{
        --background: {theme["background"]};
        --surface: {theme["surface"]};
        --surface2: {theme["surface2"]};
        --card: {theme["card"]};
        --border: {theme["border"]};
        --text: {theme["text"]};
        --muted: {theme["muted"]};
        --primary: {theme["primary"]};
        --secondary: {theme["secondary"]};
        --success: {theme["success"]};
        --warning: {theme["warning"]};
        --danger: {theme["danger"]};
    }}


    /* ========================================================
       GLOBAL
       ======================================================== */

    html,
    body,
    [class*="css"] {{
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}


    .stApp {{
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(37,99,235,0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(124,58,237,0.08),
                transparent 25%
            ),
            var(--background);

        color: var(--text);
    }}


    .main {{
        padding-top: 1rem;
        padding-bottom: 4rem;
    }}


    /* ========================================================
       REMOVE DEFAULT PADDING ON SMALLER SCREENS
       ======================================================== */

    .block-container {{
        max-width: 1450px;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-top: 1.5rem;
    }}


    /* ========================================================
       TEXT
       ======================================================== */

    h1,
    h2,
    h3,
    h4,
    p,
    label {{
        color: var(--text);
    }}


    .muted {{
        color: var(--muted);
    }}


    /* ========================================================
       HERO
       ======================================================== */

    .hero {{
        position: relative;
        overflow: hidden;

        padding: 2.5rem;

        border-radius: 28px;

        background:
            radial-gradient(
                circle at 80% 20%,
                rgba(96,165,250,0.18),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                {theme["hero1"]},
                {theme["hero2"]},
                {theme["hero3"]}
            );

        border: 1px solid var(--border);

        box-shadow:
            0 25px 70px
            rgba(0,0,0,0.25);

        margin-bottom: 1.5rem;
    }}


    .hero-badge {{
        display: inline-flex;

        align-items: center;
        gap: 0.4rem;

        padding: 0.4rem 0.8rem;

        border-radius: 999px;

        background:
            rgba(255,255,255,0.07);

        border:
            1px solid
            rgba(255,255,255,0.10);

        font-size: 0.78rem;

        color:
            var(--muted);

        margin-bottom: 1rem;
    }}


    .hero-title {{
        font-size: clamp(
            2rem,
            5vw,
            3.8rem
        );

        line-height: 1.05;

        font-weight: 850;

        letter-spacing: -0.05em;

        margin: 0;

        color: var(--text);
    }}


    .hero-subtitle {{
        max-width: 780px;

        font-size: clamp(
            0.95rem,
            2vw,
            1.15rem
        );

        line-height: 1.7;

        color: var(--muted);

        margin-top: 1rem;
    }}


    /* ========================================================
       GLASS CARDS
       ======================================================== */

    .glass-card {{
        background: var(--card);

        border:
            1px solid
            var(--border);

        border-radius: 22px;

        padding: 1.4rem;

        box-shadow:
            0 10px 40px
            rgba(0,0,0,0.12);

        backdrop-filter:
            blur(14px);

        -webkit-backdrop-filter:
            blur(14px);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }}


    .glass-card:hover {{
        transform:
            translateY(-2px);

        border-color:
            rgba(96,165,250,0.25);
    }}


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {{
        background: var(--card);

        border:
            1px solid
            var(--border);

        border-radius: 20px;

        padding: 1.25rem;

        min-height: 130px;

        box-shadow:
            0 10px 35px
            rgba(0,0,0,0.12);
    }}


    .metric-label {{
        color: var(--muted);

        font-size: 0.82rem;

        font-weight: 600;

        text-transform:
            uppercase;

        letter-spacing:
            0.06em;
    }}


    .metric-value {{
        color: var(--text);

        font-size: 2rem;

        font-weight: 850;

        letter-spacing: -0.04em;

        margin-top: 0.3rem;
    }}


    .metric-delta {{
        color: var(--success);

        font-size: 0.78rem;

        margin-top: 0.35rem;
    }}


    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-title {{
        font-size: 1.4rem;

        font-weight: 800;

        letter-spacing: -0.025em;

        margin-top: 1.5rem;

        margin-bottom: 0.25rem;
    }}


    .section-description {{
        color: var(--muted);

        font-size: 0.92rem;

        margin-bottom: 1rem;
    }}


    /* ========================================================
       STATUS PILLS
       ======================================================== */

    .status {{
        display: inline-flex;

        align-items: center;

        gap: 0.45rem;

        padding:
            0.35rem
            0.7rem;

        border-radius: 999px;

        font-size: 0.76rem;

        font-weight: 700;
    }}


    .status-green {{
        color: var(--success);

        background:
            rgba(52,211,153,0.10);

        border:
            1px solid
            rgba(52,211,153,0.18);
    }}


    .status-red {{
        color: var(--danger);

        background:
            rgba(251,113,133,0.10);

        border:
            1px solid
            rgba(251,113,133,0.18);
    }}


    /* ========================================================
       RISK CARDS
       ======================================================== */

    .risk-high {{
        padding: 1.5rem;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(220,38,38,0.12),
                rgba(127,29,29,0.08)
            );

        border:
            1px solid
            rgba(248,113,113,0.20);

        border-left:
            5px solid
            var(--danger);
    }}


    .risk-low {{
        padding: 1.5rem;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(22,163,74,0.12),
                rgba(6,78,59,0.08)
            );

        border:
            1px solid
            rgba(74,222,128,0.20);

        border-left:
            5px solid
            var(--success);
    }}


    .risk-medium {{
        padding: 1.5rem;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                rgba(217,119,6,0.12),
                rgba(120,53,15,0.08)
            );

        border:
            1px solid
            rgba(251,191,36,0.20);

        border-left:
            5px solid
            var(--warning);
    }}


    /* ========================================================
       SAFETY CARD
       ======================================================== */

    .safety-card {{
        padding: 1rem 1.2rem;

        border-radius: 18px;

        background:
            rgba(245,158,11,0.08);

        border:
            1px solid
            rgba(245,158,11,0.18);

        color: var(--text);

        margin-bottom: 1.5rem;
    }}


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 14px;

        min-height: 46px;

        font-weight: 700;

        border:
            1px solid
            var(--border);

        transition:
            all 0.2s ease;
    }}


    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        transform:
            translateY(-1px);

        border-color:
            var(--primary);
    }}


    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-baseweb="input"] {{
        border-radius: 12px;
    }}


    div[data-baseweb="select"] {{
        border-radius: 12px;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {{
        background:
            var(--surface);

        border-right:
            1px solid
            var(--border);
    }}


    /* ========================================================
       DATAFRAMES
       ======================================================== */

    div[data-testid="stDataFrame"] {{
        border-radius: 16px;
        overflow: hidden;
    }}


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {{
        text-align: center;

        color: var(--muted);

        padding:
            3rem 0
            1rem;

        font-size: 0.8rem;
    }}


    /* ========================================================
       MOBILE RESPONSIVENESS
       ======================================================== */

    @media (max-width: 768px) {{

        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }}

        .hero {{
            padding: 1.5rem;

            border-radius: 22px;
        }}

        .hero-title {{
            font-size: 2rem;
        }}

        .hero-subtitle {{
            font-size: 0.9rem;
        }}

        .metric-card {{
            min-height: 110px;
        }}

        .metric-value {{
            font-size: 1.55rem;
        }}

        .glass-card {{
            padding: 1rem;
            border-radius: 18px;
        }}

        .section-title {{
            font-size: 1.2rem;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            min-height: 50px;
        }}

        div[data-testid="stMetricValue"] {{
            font-size: 1.5rem;
        }}

        div[data-testid="stHorizontalBlock"] {{
            gap: 0.65rem;
        }}
    }}


    @media (max-width: 480px) {{

        .block-container {{
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }}

        .hero {{
            padding: 1.2rem;
        }}

        .hero-title {{
            font-size: 1.75rem;
        }}

        .hero-subtitle {{
            font-size: 0.85rem;
        }}

        .metric-card {{
            padding: 0.9rem;
        }}

        .metric-label {{
            font-size: 0.7rem;
        }}

        .metric-value {{
            font-size: 1.3rem;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None, (
            f"Model file not found: "
            f"{MODEL_PATH.name}"
        )

    try:

        loaded_model = joblib.load(
            MODEL_PATH
        )

        return loaded_model, None

    except Exception as error:

        return None, str(error)


model, model_error = load_model()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():

        return None, (
            f"Dataset file not found: "
            f"{DATA_PATH.name}"
        )

    try:

        loaded_data = pd.read_csv(
            DATA_PATH
        )

        return loaded_data, None

    except Exception as error:

        return None, str(error)


data, data_error = load_dataset()


# ============================================================
# DATASET TARGET DETECTION
# ============================================================

target_column = None

if data is not None:

    possible_targets = [
        "outcome(target)",
        "Outcome",
        "Outcome(Target)",
        "outcome",
        "target",
        "Target"
    ]

    for column in possible_targets:

        if column in data.columns:

            target_column = column

            break


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🩺 Diabetes AI"
    )

    st.caption(
        "Risk Intelligence Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔬 Risk Assessment",
            "📊 Dataset",
            "🧠 Model Intelligence",
            "⚖️ Evaluation",
            "🛡️ Responsible AI"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### ⚙️ Appearance"
    )

    selected_theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(
            st.session_state.theme
        ),
        label_visibility="collapsed"
    )

    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme

        st.rerun()

    st.divider()

    model_status = (
        "🟢 Online"
        if model is not None
        else "🔴 Offline"
    )

    data_status = (
        "🟢 Available"
        if data is not None
        else "🔴 Missing"
    )

    st.markdown(
        f"""
        **Model:** {model_status}

        **Dataset:** {data_status}
        """
    )

    st.divider()

    st.caption(
        "Built by"
    )

    st.markdown(
        "**Olalemi Olaoluwakintan Emmanuel**"
    )


# ============================================================
# TOP BAR
# ============================================================

top1, top2 = st.columns(
    [5, 1]
)

with top1:

    st.markdown(
        f"""
        <span class="status status-green">
        ● SYSTEM ONLINE
        </span>
        """,
        unsafe_allow_html=True
    )

with top2:

    if st.button(
        "💡",
        help="Change appearance"
    ):

        theme_names = list(
            THEMES.keys()
        )

        current_index = (
            theme_names.index(
                st.session_state.theme
            )
        )

        next_index = (
            current_index + 1
        ) % len(theme_names)

        st.session_state.theme = (
            theme_names[next_index]
        )

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-badge">
            🩺 AI-POWERED HEALTH ANALYTICS
        </div>

        <h1 class="hero-title">
            Diabetes Risk<br>
            Intelligence
        </h1>

        <p class="hero-subtitle">
            An interactive machine-learning screening prototype
            transforming patient-level health features into
            model-based diabetes risk estimates.
        </p>

        <br>

        <span class="status status-green">
            ● Research Prototype
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SAFETY NOTICE
# ============================================================

st.markdown(
    """
    <div class="safety-card">

    <strong>⚠️ Clinical Safety Notice</strong><br>

    This application is an educational and research prototype.
    It provides model-based screening estimates and does not
    provide a medical diagnosis. Results must not replace
    professional medical assessment, laboratory testing,
    or established clinical screening protocols.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        A high-level view of the dataset, model performance,
        optimization strategy and analytical insights.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DYNAMIC DATASET METRICS
    # --------------------------------------------------------

    total_patients = 768
    diabetic = 268
    non_diabetic = 500

    if (
        data is not None
        and target_column is not None
    ):

        total_patients = len(data)

        diabetic = int(
            (
                data[target_column] == 1
            ).sum()
        )

        non_diabetic = int(
            (
                data[target_column] == 0
            ).sum()
        )

    diabetic_pct = (
        diabetic / total_patients * 100
        if total_patients
        else 0
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    m1, m2, m3, m4, m5 = st.columns(5)

    metric_data = [
        (
            m1,
            "PATIENT RECORDS",
            f"{total_patients:,}",
            "Development dataset"
        ),
        (
            m2,
            "DIABETIC",
            f"{diabetic:,}",
            f"{diabetic_pct:.1f}% of records"
        ),
        (
            m3,
            "OPTIMIZED RECALL",
            "75.93%",
            "+25.93 pp"
        ),
        (
            m4,
            "F1 SCORE",
            "63.08%",
            "Optimized model"
        ),
        (
            m5,
            "ROC-AUC",
            "78.69%",
            "Discrimination"
        )
    ]

    for col, label, value, delta in metric_data:

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                    {label}
                    </div>

                    <div class="metric-value">
                    {value}
                    </div>

                    <div class="metric-delta">
                    {delta}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Why the optimized model matters</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🎯 Recall became the priority</h3>

            <p>
            The baseline model achieved approximately
            <strong>50.00%</strong> recall.
            </p>

            <p>
            After optimization, recall increased to
            <strong>75.93%</strong>.
            </p>

            <p class="muted">
            In a screening-oriented analytical setting,
            this means the optimized model identifies a
            larger proportion of positive cases in the
            evaluated test set.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🧠 Optimization strategy</h3>

            <p>
            <strong>01 — Scaling</strong><br>
            StandardScaler normalizes numerical feature magnitudes.
            </p>

            <p>
            <strong>02 — Balancing</strong><br>
            SMOTE addresses minority-class representation during training.
            </p>

            <p>
            <strong>03 — Search</strong><br>
            GridSearchCV evaluates Logistic Regression configurations
            using F1-oriented optimization.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # BASELINE COMPARISON
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Baseline → Optimized</div>',
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Baseline": [
            BASELINE_METRICS["Accuracy"],
            BASELINE_METRICS["Precision"],
            BASELINE_METRICS["Recall"],
            BASELINE_METRICS["F1 Score"]
        ],
        "Optimized": [
            OPTIMIZED_METRICS["Accuracy"],
            OPTIMIZED_METRICS["Precision"],
            OPTIMIZED_METRICS["Recall"],
            OPTIMIZED_METRICS["F1 Score"]
        ]
    })

    comparison["Change"] = (
        comparison["Optimized"]
        - comparison["Baseline"]
    )

    st.dataframe(
        comparison.style.format({
            "Baseline": "{:.2%}",
            "Optimized": "{:.2%}",
            "Change": "{:+.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The optimized model traded some accuracy and precision "
        "for a substantial improvement in recall."
    )

    # --------------------------------------------------------
    # FEATURE INSIGHT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📌 Feature Intelligence</div>',
        unsafe_allow_html=True
    )

    feature_df = pd.DataFrame(
        list(
            FEATURE_COEFFICIENTS.items()
        ),
        columns=[
            "Feature",
            "Coefficient"
        ]
    )

    feature_df = feature_df.sort_values(
        "Coefficient"
    )

    st.bar_chart(
        feature_df.set_index(
            "Feature"
        ),
        y="Coefficient",
        horizontal=True,
        height=450
    )

    st.caption(
        "Higher positive coefficients indicate stronger positive "
        "association with the model's predicted positive class. "
        "They do not establish causality."
    )


# ============================================================
# PAGE 2 — RISK ASSESSMENT
# ============================================================

elif page == "🔬 Risk Assessment":

    st.markdown(
        '<div class="section-title">🔬 Patient Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Enter the eight patient variables used by the trained model.
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:

        st.error(
            "The trained model could not be loaded."
        )

        st.info(
            f"Expected file: {MODEL_PATH.name}"
        )

        if model_error:

            st.code(
                model_error
            )

        st.stop()

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    with st.form(
        "patient_risk_form"
    ):

        st.markdown(
            "### Patient Characteristics"
        )

        c1, c2 = st.columns(2)

        with c1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                step=1
            )

            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0.0,
                max_value=300.0,
                value=120.0,
                step=1.0
            )

            blood_pressure = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=0.0,
                max_value=200.0,
                value=72.0,
                step=1.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0,
                max_value=100.0,
                value=23.0,
                step=1.0
            )

        with c2:

            insulin = st.number_input(
                "Insulin (μU/mL)",
                min_value=0.0,
                max_value=1000.0,
                value=125.0,
                step=1.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=80.0,
                value=32.0,
                step=0.1
            )

            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.47,
                step=0.01
            )

            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=33,
                step=1
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔎 Run Risk Assessment",
            use_container_width=True,
            type="primary"
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        patient = pd.DataFrame(
            [{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age
            }],
            columns=EXPECTED_FEATURES
        )

        try:

            prediction = int(
                model.predict(
                    patient
                )[0]
            )

            probability = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = float(
                    model.predict_proba(
                        patient
                    )[0][1]
                )

            elif hasattr(
                model,
                "decision_function"
            ):

                decision = float(
                    model.decision_function(
                        patient
                    )[0]
                )

                probability = float(
                    1 /
                    (
                        1 +
                        np.exp(-decision)
                    )
                )

            if probability is not None:

                probability = min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )

            st.session_state.assessment_result = {
                "prediction": prediction,
                "probability": probability,
                "patient": patient
            }

        except Exception as error:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(
                error
            )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    result = (
        st.session_state.assessment_result
    )

    if result is not None:

        prediction = result[
            "prediction"
        ]

        probability = result[
            "probability"
        ]

        patient = result[
            "patient"
        ]

        st.divider()

        st.markdown(
            "### Assessment Result"
        )

        if probability is not None:

            probability_percent = (
                probability * 100
            )

            if probability < 0.30:

                band = "Lower"

            elif probability < 0.60:

                band = "Intermediate"

            else:

                band = "Higher"

            a1, a2, a3 = st.columns(3)

            with a1:

                st.metric(
                    "Model Probability",
                    f"{probability_percent:.1f}%"
                )

            with a2:

                st.metric(
                    "Classification",
                    (
                        "Positive (1)"
                        if prediction == 1
                        else "Negative (0)"
                    )
                )

            with a3:

                st.metric(
                    "Risk Band",
                    band
                )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            st.markdown(
                "#### Model-Estimated Probability"
            )

            st.progress(
                probability
            )

            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="risk-high">

                    <h3>
                    ⚠️ Positive Model Classification
                    </h3>

                    <p>
                    The model classified this patient as
                    <strong>Class 1</strong>.
                    </p>

                    <p>
                    Model-estimated probability:
                    <strong>
                    {probability_percent:.1f}%
                    </strong>
                    </p>

                    <p>
                    This is a screening flag only and
                    should not be interpreted as a diagnosis.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif band == "Intermediate":

                st.markdown(
                    f"""
                    <div class="risk-medium">

                    <h3>
                    🟡 Intermediate Model Estimate
                    </h3>

                    <p>
                    The model classified this patient as
                    <strong>Class 0</strong>, but the estimated
                    positive-class probability is intermediate.
                    </p>

                    <p>
                    Estimated probability:
                    <strong>
                    {probability_percent:.1f}%
                    </strong>
                    </p>

                    <p>
                    Clinical assessment should determine
                    the appropriate next step.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="risk-low">

                    <h3>
                    ✓ Negative Model Classification
                    </h3>

                    <p>
                    The model classified this patient as
                    <strong>Class 0</strong>.
                    </p>

                    <p>
                    Model-estimated probability:
                    <strong>
                    {probability_percent:.1f}%
                    </strong>
                    </p>

                    <p>
                    A negative model result does not rule
                    out diabetes.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.markdown(
                "### Patient Input Summary"
            )

            st.dataframe(
                patient.T.rename(
                    columns={
                        0: "Patient Value"
                    }
                ),
                use_container_width=True
            )

            # ------------------------------------------------
            # REPORT
            # ------------------------------------------------

            report_probability = (
                f"{probability_percent:.2f}%"
                if probability is not None
                else "Unavailable"
            )

            report = f"""
DIABETES RISK INTELLIGENCE
MACHINE LEARNING SCREENING REPORT
============================================================

Assessment Date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MODEL RESULT
------------------------------------------------------------

Predicted Class:
{
    "Positive (1)"
    if prediction == 1
    else "Negative (0)"
}

Model Probability:
{report_probability}

Risk Band:
{band}


PATIENT INPUTS
------------------------------------------------------------

Pregnancies: {pregnancies}
Glucose: {glucose}
BloodPressure: {blood_pressure}
SkinThickness: {skin_thickness}
Insulin: {insulin}
BMI: {bmi}
DiabetesPedigreeFunction: {pedigree}
Age: {age}


MODEL PERFORMANCE
------------------------------------------------------------

Accuracy: 68.83%
Precision: 53.95%
Recall: 75.93%
F1 Score: 63.08%
ROC-AUC: 78.69%


CLINICAL SAFETY NOTICE
------------------------------------------------------------

This application is an educational and research prototype.

It is NOT a medical diagnosis.

The result must not replace professional medical
assessment, laboratory testing, or established
clinical screening protocols.

Model probability should not be interpreted as a
clinically calibrated probability unless appropriate
calibration and external validation have been performed.
"""

            st.download_button(
                "📄 Download Assessment Report",
                data=report,
                file_name=(
                    "diabetes_risk_assessment.txt"
                ),
                mime="text/plain",
                use_container_width=True
            )


# ============================================================
# PAGE 3 — DATASET
# ============================================================

elif page == "📊 Dataset":

    st.markdown(
        '<div class="section-title">📊 Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    if data is None:

        st.error(
            "Dataset could not be loaded."
        )

        if data_error:

            st.code(
                data_error
            )

    else:

        st.markdown(
            """
            <div class="section-description">
            Explore the dataset used during development
            of the predictive model.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        d1, d2, d3, d4 = st.columns(4)

        positive = 0
        negative = 0

        if target_column is not None:

            positive = int(
                (
                    data[target_column] == 1
                ).sum()
            )

            negative = int(
                (
                    data[target_column] == 0
                ).sum()
            )

        with d1:

            st.metric(
                "Records",
                f"{len(data):,}"
            )

        with d2:

            st.metric(
                "Columns",
                f"{len(data.columns):,}"
            )

        with d3:

            st.metric(
                "Positive",
                f"{positive:,}"
            )

        with d4:

            st.metric(
                "Negative",
                f"{negative:,}"
            )

        st.divider()

        # ----------------------------------------------------
        # DATA DISTRIBUTION
        # ----------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.markdown(
                "### Target Distribution"
            )

            if target_column:

                target_display = pd.DataFrame({
                    "Class": [
                        "No Diabetes",
                        "Diabetes"
                    ],
                    "Patients": [
                        negative,
                        positive
                    ]
                })

                st.bar_chart(
                    target_display.set_index(
                        "Class"
                    )
                )

        with right:

            st.markdown(
                "### Missing Values"
            )

            missing = (
                data.isnull()
                .sum()
            )

            missing = missing[
                missing > 0
            ].sort_values(
                ascending=False
            )

            if len(missing):

                st.bar_chart(
                    missing
                )

            else:

                st.success(
                    "✓ No missing values detected."
                )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        st.markdown(
            "### 🔍 Data Quality"
        )

        q1, q2, q3 = st.columns(3)

        with q1:

            st.metric(
                "Missing Cells",
                f"{int(data.isnull().sum().sum()):,}"
            )

        with q2:

            st.metric(
                "Duplicate Rows",
                f"{int(data.duplicated().sum()):,}"
            )

        with q3:

            st.metric(
                "Numeric Features",
                f"{len(data.select_dtypes(include=np.number).columns):,}"
            )

        st.divider()

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        st.markdown(
            "### Dataset Preview"
        )

        rows_to_show = st.slider(
            "Rows to display",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )

        st.dataframe(
            data.head(rows_to_show),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 4 — MODEL INTELLIGENCE
# ============================================================

elif page == "🧠 Model Intelligence":

    st.markdown(
        '<div class="section-title">🧠 Model Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Explore the architecture, optimization strategy and
        learned feature associations.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    status_left, status_right = st.columns(2)

    with status_left:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🤖 Model</h3>

            <h2>
            Logistic Regression
            </h2>

            <p class="muted">
            Interpretable linear classification model
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with status_right:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🔗 Pipeline</h3>

            <p>
            StandardScaler
            →
            SMOTE
            →
            Logistic Regression
            </p>

            <p class="muted">
            Designed to address feature scaling and
            class imbalance during model development.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # PIPELINE TABLE
    # --------------------------------------------------------

    st.markdown(
        "### Machine Learning Pipeline"
    )

    pipeline = pd.DataFrame({
        "Stage": [
            "Input",
            "Scaling",
            "Balancing",
            "Classifier",
            "Optimization",
            "Output"
        ],
        "Method": [
            "8 patient features",
            "StandardScaler",
            "SMOTE",
            "Logistic Regression",
            "GridSearchCV",
            "Class + Probability"
        ]
    })

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Selected Hyperparameters"
    )

    parameters = pd.DataFrame({
        "Parameter": [
            "C",
            "Solver",
            "Class Weight"
        ],
        "Value": [
            "0.001",
            "liblinear",
            "None"
        ]
    })

    st.dataframe(
        parameters,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Feature Associations"
    )

    feature_df = pd.DataFrame(
        list(
            FEATURE_COEFFICIENTS.items()
        ),
        columns=[
            "Feature",
            "Coefficient"
        ]
    )

    feature_df[
        "Absolute Importance"
    ] = feature_df[
        "Coefficient"
    ].abs()

    feature_df = feature_df.sort_values(
        "Absolute Importance",
        ascending=False
    )

    st.bar_chart(
        feature_df.set_index(
            "Feature"
        )[[
            "Absolute Importance"
        ]],
        horizontal=True,
        height=450
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Glucose has the largest positive coefficient in the "
        "reported Logistic Regression model, followed by BMI. "
        "These associations do not establish causality."
    )


# ============================================================
# PAGE 5 — EVALUATION
# ============================================================

elif page == "⚖️ Evaluation":

    st.markdown(
        '<div class="section-title">⚖️ Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Evaluation of the optimized model using the reported
        test-set performance.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    e1, e2, e3, e4, e5 = st.columns(5)

    evaluation_metrics = [
        (
            e1,
            "Accuracy",
            "68.83%"
        ),
        (
            e2,
            "Precision",
            "53.95%"
        ),
        (
            e3,
            "Recall",
            "75.93%"
        ),
        (
            e4,
            "F1 Score",
            "63.08%"
        ),
        (
            e5,
            "ROC-AUC",
            "78.69%"
        )
    ]

    for col, label, value in evaluation_metrics:

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                    {label}
                    </div>

                    <div class="metric-value">
                    {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown(
        "### Confusion Matrix"
    )

    tn = CONFUSION_MATRIX["TN"]
    fp = CONFUSION_MATRIX["FP"]
    fn = CONFUSION_MATRIX["FN"]
    tp = CONFUSION_MATRIX["TP"]

    cm = pd.DataFrame(
        [
            [tn, fp],
            [fn, tp]
        ],
        index=[
            "Actual Negative",
            "Actual Positive"
        ],
        columns=[
            "Predicted Negative",
            "Predicted Positive"
        ]
    )

    st.dataframe(
        cm,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CONFUSION DETAILS
    # --------------------------------------------------------

    st.markdown(
        "### Interpretation"
    )

    confusion_details = pd.DataFrame({
        "Outcome": [
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive"
        ],
        "Count": [
            tn,
            fp,
            fn,
            tp
        ],
        "Meaning": [
            "Correctly identified negative cases",
            "Negative cases incorrectly flagged",
            "Positive cases missed by the model",
            "Correctly identified positive cases"
        ]
    })

    st.dataframe(
        confusion_details,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CALCULATIONS
    # --------------------------------------------------------

    total = (
        tn +
        fp +
        fn +
        tp
    )

    accuracy = (
        (tn + tp) / total
        if total
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    f1 = (
        2 *
        precision *
        recall /
        (precision + recall)
        if (precision + recall)
        else 0
    )

    calculated = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "From Confusion Matrix": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    st.markdown(
        "### 📐 Reconstructed Metrics"
    )

    st.dataframe(
        calculated.style.format({
            "From Confusion Matrix":
                "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SCREENING TRADEOFF
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Screening Trade-off"
    )

    st.warning(
        """
        The optimized model places greater emphasis on identifying
        positive cases. The reported test result contains 13 false
        negatives and 35 false positives.

        Higher recall does not by itself establish clinical safety.
        Calibration, specificity, external validation, fairness,
        clinical thresholds and prospective evaluation would also
        be required before clinical use.
        """
    )


# ============================================================
# PAGE 6 — RESPONSIBLE AI
# ============================================================

elif page == "🛡️ Responsible AI":

    st.markdown(
        '<div class="section-title">🛡️ Responsible AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Understanding the boundary between a machine-learning
        research prototype and a clinical decision system.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # WHAT IT DOES
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="glass-card">

            <h3>✅ What this project demonstrates</h3>

            <ul>
                <li>Data preprocessing</li>
                <li>Exploratory analysis</li>
                <li>Feature scaling</li>
                <li>Class imbalance handling</li>
                <li>SMOTE</li>
                <li>Hyperparameter optimization</li>
                <li>Classification evaluation</li>
                <li>Model serialization</li>
                <li>Interactive ML deployment</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🚫 What it cannot claim</h3>

            <ul>
                <li>Medical diagnosis</li>
                <li>Clinical validation</li>
                <li>Autonomous treatment decisions</li>
                <li>Universal population performance</li>
                <li>Clinically calibrated probability</li>
                <li>Replacement for professional assessment</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # FUTURE REQUIREMENTS
    # --------------------------------------------------------

    st.markdown(
        "### 🔬 Before Real-World Clinical Deployment"
    )

    requirements = pd.DataFrame({
        "Requirement": [
            "External validation",
            "Population evaluation",
            "Probability calibration",
            "Clinical threshold selection",
            "Prospective evaluation",
            "Clinical expert review",
            "Privacy & security",
            "Regulatory governance"
        ],
        "Status": [
            "Required",
            "Required",
            "Required",
            "Required",
            "Required",
            "Required",
            "Required",
            "Required"
        ]
    })

    st.dataframe(
        requirements,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RESEARCH NOTE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="glass-card">

        <h3>🧑🏽‍💻 Researcher's Note</h3>

        <p>
        The purpose of this application is to demonstrate how
        a machine-learning model can move from experimentation
        in a notebook into an interactive analytical interface.
        </p>

        <p class="muted">
        It should be viewed as a portfolio and research prototype,
        rather than a clinical decision-making system.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>
        🩺 Diabetes Risk Intelligence
        </strong>

        <br>

        Diabetes Risk Prediction Using Machine Learning

        <br><br>

        Built by
        <strong>
        Olalemi Olaoluwakintan Emmanuel
        </strong>

        <br>

        <span>
        Educational & Research Prototype
        </span>

    </div>
    """,
    unsafe_allow_html=True
)
