# ============================================================
# 🩺 DIABETES RISK INTELLIGENCE PLATFORM
# Diabetes Risk Prediction Using Machine Learning
# Author: Olalemi Olaoluwakintan Emmanuel
# ============================================================

import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes AI | Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "diabetes_risk_prediction_model.pkl"
DATASET_PATH = "diabetes_nan.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

DISPLAY_NAMES = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age"
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(34, 211, 238, 0.08),
                transparent 25%
            ),
            radial-gradient(
                circle at 10% 20%,
                rgba(99, 102, 241, 0.06),
                transparent 25%
            ),
            #05070a;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #070b10 0%,
                #05070a 100%
            );
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero-container {
        padding: 2.5rem 0 1.5rem 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        border: 1px solid rgba(34,211,238,0.25);
        background: rgba(34,211,238,0.07);
        color: #67e8f9;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-size: clamp(2.8rem, 7vw, 6.5rem);
        line-height: 0.95;
        letter-spacing: -0.055em;
        font-weight: 850;
        margin: 0;
        color: #f8fafc;
    }

    .hero-title span {
        color: #67e8f9;
    }

    .hero-subtitle {
        max-width: 760px;
        color: #94a3b8;
        font-size: 1.08rem;
        line-height: 1.7;
        margin-top: 1.5rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 650;
        margin-top: 1rem;
    }

    .status-green {
        background: rgba(34,197,94,0.09);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.18);
    }

    .status-blue {
        background: rgba(59,130,246,0.09);
        color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.18);
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .glass-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.045),
                rgba(255,255,255,0.018)
            );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.35rem;
        box-shadow:
            0 20px 50px rgba(0,0,0,0.18);
    }

    .section-card {
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
    }

    .card-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .card-subtitle {
        color: #64748b;
        font-size: 0.84rem;
        line-height: 1.5;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.05),
                rgba(255,255,255,0.018)
            );
        border: 1px solid rgba(255,255,255,0.075);
        border-radius: 18px;
        padding: 1.25rem;
        min-height: 145px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        font-weight: 750;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-top: 0.4rem;
    }

    .metric-delta {
        color: #67e8f9;
        font-size: 0.76rem;
        margin-top: 0.25rem;
    }

    /* ========================================================
       RISK RESULT
       ======================================================== */

    .risk-safe {
        border: 1px solid rgba(34,197,94,0.3);
        background: rgba(34,197,94,0.075);
        border-radius: 22px;
        padding: 1.8rem;
    }

    .risk-warning {
        border: 1px solid rgba(245,158,11,0.32);
        background: rgba(245,158,11,0.075);
        border-radius: 22px;
        padding: 1.8rem;
    }

    .risk-danger {
        border: 1px solid rgba(239,68,68,0.35);
        background: rgba(239,68,68,0.075);
        border-radius: 22px;
        padding: 1.8rem;
    }

    .risk-label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        font-weight: 800;
        text-transform: uppercase;
        color: #94a3b8;
    }

    .risk-title {
        font-size: 2rem;
        font-weight: 850;
        margin: 0.35rem 0;
        letter-spacing: -0.04em;
    }

    .risk-text {
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.92rem;
    }

    /* ========================================================
       PROGRESS
       ======================================================== */

    .progress-track {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        overflow: hidden;
        margin-top: 0.75rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(
            90deg,
            #22d3ee,
            #6366f1
        );
    }

    /* ========================================================
       INFO
       ======================================================== */

    .info-box {
        border-left: 3px solid #22d3ee;
        background: rgba(34,211,238,0.045);
        padding: 1rem 1.15rem;
        border-radius: 0 12px 12px 0;
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.9rem;
    }

    .warning-box {
        border-left: 3px solid #f59e0b;
        background: rgba(245,158,11,0.045);
        padding: 1rem 1.15rem;
        border-radius: 0 12px 12px 0;
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.9rem;
    }

    .danger-box {
        border-left: 3px solid #ef4444;
        background: rgba(239,68,68,0.045);
        padding: 1rem 1.15rem;
        border-radius: 0 12px 12px 0;
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.9rem;
    }

    /* ========================================================
       TABLE
       ======================================================== */

    .simple-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
    }

    .simple-table th {
        text-align: left;
        color: #64748b;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .simple-table td {
        padding: 0.75rem;
        color: #cbd5e1;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* ========================================================
       STREAMLIT INPUTS
       ======================================================== */

    div[data-baseweb="input"] {
        background: rgba(255,255,255,0.035);
        border-radius: 10px;
    }

    div[data-baseweb="input"] input {
        color: #f8fafc;
    }

    label {
        color: #cbd5e1 !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: 1px solid rgba(34,211,238,0.22);
        background:
            linear-gradient(
                135deg,
                rgba(34,211,238,0.14),
                rgba(99,102,241,0.13)
            );
        color: #f8fafc;
        font-weight: 750;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(34,211,238,0.55);
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 45px;
        border-radius: 12px;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.07);
        color: #64748b;
        text-align: center;
        line-height: 1.7;
    }

    .footer strong {
        color: #cbd5e1;
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-container {
            padding-top: 1rem;
        }

        .hero-title {
            font-size: 3.2rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
        }

        .metric-card {
            min-height: 125px;
        }

        .metric-value {
            font-size: 1.65rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None

    try:
        return pd.read_csv(DATASET_PATH)
    except Exception:
        return None


model, model_error = load_model()
dataset = load_dataset()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.35rem;
            font-weight:800;
            color:#f8fafc;
            margin-bottom:0.2rem;
        ">
            🩺 Diabetes AI
        </div>

        <div style="
            color:#64748b;
            font-size:0.78rem;
            margin-bottom:1.5rem;
        ">
            Risk Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True
    )

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

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#64748b;
            font-size:0.72rem;
            text-transform:uppercase;
            letter-spacing:0.1em;
            font-weight:700;
            margin-bottom:0.6rem;
        ">
            System Status
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is not None:
        st.success("🟢 Model Online")
    else:
        st.error("🔴 Model Offline")

    if dataset is not None:
        st.success("🟢 Dataset Available")
    else:
        st.warning("🟡 Dataset Unavailable")

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#64748b;
            font-size:0.75rem;
            line-height:1.6;
        ">
        Built by<br>
        <strong style="color:#cbd5e1;">
        Olalemi Olaoluwakintan Emmanuel
        </strong>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-container">

        <div class="hero-badge">
            🩺 AI-POWERED HEALTH ANALYTICS
        </div>

        <h1 class="hero-title">
            Diabetes Risk<br>
            <span>Intelligence</span>
        </h1>

        <p class="hero-subtitle">
            An interactive machine-learning screening platform
            transforming patient-level health features into
            model-based diabetes risk estimates.
        </p>

        <span class="status-pill status-green">
            ● Research Model Online
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CLINICAL SAFETY NOTICE
# ============================================================

st.markdown(
    """
    <div class="warning-box">
        <strong>⚠️ Clinical Safety Notice</strong><br><br>

        This application provides a machine-learning risk estimate based
        on the information entered by the user. It is designed to support
        screening, education and research — not to independently diagnose
        diabetes.

        Results should be interpreted alongside professional medical
        assessment, appropriate laboratory testing and established
        clinical screening procedures.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "🏠 Overview":

    st.markdown("## Executive Overview")

    st.caption(
        "A high-level view of the dataset, model performance, "
        "optimization strategy and analytical insights."
    )

    total_records = 768
    diabetic_records = 268
    non_diabetic_records = 500

    metrics = [
        ("PATIENT RECORDS", f"{total_records}", "Development dataset"),
        ("DIABETIC", f"{diabetic_records}", "34.9% of records"),
        ("OPTIMIZED RECALL", "75.93%", "+25.93 percentage points"),
        ("F1 SCORE", "63.08%", "Optimized model"),
        ("ROC-AUC", "78.69%", "Discrimination")
    ]

    cols = st.columns(5)

    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-delta">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card section-card">

            <div class="card-title">
                🎯 Why the optimized model matters
            </div>

            <div class="card-subtitle">
                The project was optimized around the problem's actual
                evaluation needs rather than accuracy alone.
            </div>

            <br>

            <strong style="color:#f8fafc;">
                Recall became the priority
            </strong>

            <p style="color:#94a3b8; line-height:1.7;">
                The baseline Logistic Regression model achieved
                <strong style="color:#f8fafc;">50.00% recall</strong>.
                After applying feature scaling, SMOTE and hyperparameter
                tuning, recall increased to
                <strong style="color:#67e8f9;">75.93%</strong>.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 Optimization Strategy")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.4rem;">01</div>
                <h4>Scaling</h4>
                <p style="color:#94a3b8;">
                StandardScaler places numerical features on comparable
                scales before model training.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.4rem;">02</div>
                <h4>Balancing</h4>
                <p style="color:#94a3b8;">
                SMOTE generates synthetic minority-class training examples
                to improve representation during training.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:1.4rem;">03</div>
                <h4>Search</h4>
                <p style="color:#94a3b8;">
                GridSearchCV evaluates Logistic Regression configurations
                using F1 as the optimization objective.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            <strong>Key analytical lesson:</strong>
            The optimized model sacrificed some overall accuracy and
            precision in exchange for substantially higher recall.
            For a screening-oriented model, this is an intentional
            trade-off rather than a failure.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RISK ASSESSMENT PAGE
# ============================================================

elif page == "🔬 Risk Assessment":

    st.markdown("## 🔬 Diabetes Risk Assessment")

    st.caption(
        "Enter patient information to generate a model-based diabetes "
        "risk estimate."
    )

    if model is None:

        st.error(
            "The trained model could not be loaded. "
            "Please make sure diabetes_risk_prediction_model.pkl "
            "is in the same folder as app.py."
        )

    else:

        tab_single, tab_batch = st.tabs(
            [
                "👤 Single Patient",
                "📂 Batch Assessment"
            ]
        )

        # ====================================================
        # SINGLE PATIENT
        # ====================================================

        with tab_single:

            st.markdown(
                """
                <div class="glass-card">
                    <div class="card-title">
                        Patient Information
                    </div>
                    <div class="card-subtitle">
                        Enter the available health measurements below.
                        The model expects the same eight features used
                        during development.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                pregnancies = st.number_input(
                    "Pregnancies",
                    min_value=0,
                    max_value=20,
                    value=1,
                    step=1
                )

            with c2:
                glucose = st.number_input(
                    "Glucose (mg/dL)",
                    min_value=0.0,
                    max_value=300.0,
                    value=120.0,
                    step=1.0
                )

            with c3:
                blood_pressure = st.number_input(
                    "Blood Pressure (mmHg)",
                    min_value=0.0,
                    max_value=200.0,
                    value=70.0,
                    step=1.0
                )

            with c4:
                skin_thickness = st.number_input(
                    "Skin Thickness (mm)",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0
                )

            c5, c6, c7, c8 = st.columns(4)

            with c5:
                insulin = st.number_input(
                    "Insulin (µU/mL)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=80.0,
                    step=1.0
                )

            with c6:
                bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    max_value=80.0,
                    value=32.0,
                    step=0.1
                )

            with c7:
                pedigree = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.47,
                    step=0.01,
                    format="%.2f"
                )

            with c8:
                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=33,
                    step=1
                )

            st.markdown("<br>", unsafe_allow_html=True)

            predict_clicked = st.button(
                "🩺 Assess Diabetes Risk",
                use_container_width=True
            )

            if predict_clicked:

                patient_data = pd.DataFrame(
                    [[
                        pregnancies,
                        glucose,
                        blood_pressure,
                        skin_thickness,
                        insulin,
                        bmi,
                        pedigree,
                        age
                    ]],
                    columns=FEATURES
                )

                try:

                    prediction = int(
                        model.predict(patient_data)[0]
                    )

                    probability = float(
                        model.predict_proba(patient_data)[0][1]
                    )

                    probability_percent = probability * 100

                    st.markdown("<br>", unsafe_allow_html=True)

                    # -----------------------------------------
                    # RESULT
                    # -----------------------------------------

                    if prediction == 1:

                        st.markdown(
                            f"""
                            <div class="risk-danger">

                                <div class="risk-label">
                                    MODEL RESULT
                                </div>

                                <div class="risk-title">
                                    ⚠️ Elevated Diabetes Risk
                                </div>

                                <div class="risk-text">
                                    The model classified this input as
                                    <strong>positive for diabetes risk</strong>.
                                    The estimated model probability is
                                    <strong>{probability_percent:.1f}%</strong>.
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="risk-safe">

                                <div class="risk-label">
                                    MODEL RESULT
                                </div>

                                <div class="risk-title">
                                    ✓ Lower Predicted Risk
                                </div>

                                <div class="risk-text">
                                    The model classified this input as
                                    <strong>negative for diabetes risk</strong>.
                                    The estimated model probability is
                                    <strong>{probability_percent:.1f}%</strong>.
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # -----------------------------------------
                    # PROBABILITY
                    # -----------------------------------------

                    st.markdown("### 📈 Model Probability")

                    st.markdown(
                        f"""
                        <div class="glass-card">

                            <div style="
                                display:flex;
                                justify-content:space-between;
                                align-items:center;
                            ">
                                <span style="color:#94a3b8;">
                                    Estimated probability of positive class
                                </span>

                                <strong style="
                                    color:#67e8f9;
                                    font-size:1.25rem;
                                ">
                                    {probability_percent:.1f}%
                                </strong>
                            </div>

                            <div class="progress-track">
                                <div class="progress-fill"
                                     style="width:{min(probability_percent,100):.2f}%;">
                                </div>
                            </div>

                            <div style="
                                color:#64748b;
                                font-size:0.76rem;
                                margin-top:0.7rem;
                            ">
                                This is the model's estimated probability,
                                not a clinical probability or diagnosis.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # -----------------------------------------
                    # INPUT SUMMARY
                    # -----------------------------------------

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown("### 📋 Assessment Summary")

                    summary_cols = st.columns(4)

                    summary = [
                        ("Glucose", f"{glucose:.1f} mg/dL"),
                        ("BMI", f"{bmi:.1f}"),
                        ("Age", f"{age} years"),
                        ("Blood Pressure", f"{blood_pressure:.1f} mmHg")
                    ]

                    for col, (label, value) in zip(
                        summary_cols,
                        summary
                    ):
                        with col:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-label">
                                        {label}
                                    </div>
                                    <div class="metric-value"
                                         style="font-size:1.45rem;">
                                        {value}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown(
                        """
                        <div class="warning-box">
                            <strong>Important:</strong>
                            A model prediction should not be interpreted
                            as confirmation that a person does or does
                            not have diabetes. If you have health concerns,
                            discuss the result with a qualified healthcare
                            professional and use appropriate clinical tests.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        "Prediction could not be completed. "
                        "Please verify that the model file matches the "
                        "eight expected input features."
                    )

                    st.code(str(e))


        # ====================================================
        # BATCH ASSESSMENT
        # ====================================================

        with tab_batch:

            st.markdown(
                """
                <div class="glass-card">

                    <div class="card-title">
                        📂 Batch Risk Assessment
                    </div>

                    <div class="card-subtitle">
                        Upload a CSV containing patient records using
                        the same eight feature names expected by the model.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # -----------------------------------------------
            # TEMPLATE
            # -----------------------------------------------

            template_data = pd.DataFrame({
                "Pregnancies": [1],
                "Glucose": [120.0],
                "BloodPressure": [70.0],
                "SkinThickness": [20.0],
                "Insulin": [80.0],
                "BMI": [32.0],
                "DiabetesPedigreeFunction": [0.47],
                "Age": [33]
            })

            st.download_button(
                label="⬇️ Download CSV Template",
                data=template_data.to_csv(index=False),
                file_name="diabetes_prediction_template.csv",
                mime="text/csv"
            )

            uploaded_file = st.file_uploader(
                "Upload patient CSV",
                type=["csv"]
            )

            if uploaded_file is not None:

                try:

                    batch_data = pd.read_csv(uploaded_file)

                    missing_features = [
                        feature
                        for feature in FEATURES
                        if feature not in batch_data.columns
                    ]

                    if missing_features:

                        st.error(
                            "The uploaded CSV is missing these required "
                            f"columns: {', '.join(missing_features)}"
                        )

                    else:

                        input_data = batch_data[FEATURES].copy()

                        # Convert values to numeric
                        for feature in FEATURES:
                            input_data[feature] = pd.to_numeric(
                                input_data[feature],
                                errors="coerce"
                            )

                        invalid_rows = input_data.isnull().any(axis=1).sum()

                        if invalid_rows > 0:

                            st.warning(
                                f"{invalid_rows} row(s) contain missing or "
                                "non-numeric values. Those rows will be "
                                "excluded from prediction."
                            )

                        valid_mask = ~input_data.isnull().any(axis=1)

                        valid_input = input_data.loc[
                            valid_mask
                        ].copy()

                        if len(valid_input) == 0:

                            st.error(
                                "No valid patient records were found "
                                "after checking the uploaded file."
                            )

                        else:

                            batch_predictions = model.predict(
                                valid_input
                            )

                            batch_probabilities = model.predict_proba(
                                valid_input
                            )[:, 1]

                            results = batch_data.loc[
                                valid_input.index
                            ].copy()

                            results["Predicted Outcome"] = np.where(
                                batch_predictions == 1,
                                "Higher Diabetes Risk",
                                "Lower Diabetes Risk"
                            )

                            results["Risk Probability (%)"] = (
                                batch_probabilities * 100
                            ).round(2)

                            results["Model Prediction"] = batch_predictions

                            # --------------------------------
                            # SUMMARY
                            # --------------------------------

                            total = len(results)

                            positive = int(
                                (batch_predictions == 1).sum()
                            )

                            negative = int(
                                (batch_predictions == 0).sum()
                            )

                            positive_rate = (
                                positive / total * 100
                            )

                            st.markdown("<br>", unsafe_allow_html=True)

                            m1, m2, m3 = st.columns(3)

                            with m1:
                                st.metric(
                                    "Records Assessed",
                                    total
                                )

                            with m2:
                                st.metric(
                                    "Higher Risk",
                                    positive
                                )

                            with m3:
                                st.metric(
                                    "Higher Risk %",
                                    f"{positive_rate:.1f}%"
                                )

                            st.markdown("<br>", unsafe_allow_html=True)

                            st.markdown("### Prediction Results")

                            st.dataframe(
                                results,
                                use_container_width=True,
                                hide_index=True
                            )

                            csv_results = results.to_csv(
                                index=False
                            )

                            st.download_button(
                                label="⬇️ Download Prediction Results",
                                data=csv_results,
                                file_name=(
                                    "diabetes_risk_predictions.csv"
                                ),
                                mime="text/csv"
                            )

                            st.markdown(
                                """
                                <div class="warning-box">
                                    <strong>Batch assessment reminder:</strong>
                                    These predictions are model-generated
                                    screening estimates. They should not be
                                    treated as automated diagnoses.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                except Exception as e:

                    st.error(
                        "The uploaded file could not be processed."
                    )

                    st.code(str(e))


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.markdown("## 📊 Dataset Intelligence")

    st.caption(
        "Overview of the dataset used during model development."
    )

    if dataset is None:

        st.warning(
            "diabetes_nan.csv was not found in the application directory."
        )

    else:

        total = len(dataset)

        target_counts = dataset["outcome(target)"].value_counts()

        non_diabetic = int(target_counts.get(0, 0))
        diabetic = int(target_counts.get(1, 0))

        c1, c2, c3, c4 = st.columns(4)

        overview = [
            ("TOTAL RECORDS", total, "Patients"),
            ("FEATURES", 8, "Predictor variables"),
            ("NO DIABETES", non_diabetic, "65.1%"),
            ("DIABETES", diabetic, "34.9%")
        ]

        for col, (label, value, delta) in zip(
            [c1, c2, c3, c4],
            overview
        ):

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

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Dataset Preview")

        st.dataframe(
            dataset.head(20),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Missing Data Audit")

        missing = dataset.isnull().sum()

        missing_table = pd.DataFrame({
            "Feature": missing.index,
            "Missing Values": missing.values,
            "Missing %": (
                missing.values / len(dataset) * 100
            ).round(2)
        })

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            <div class="info-box">

                <strong>Preprocessing:</strong>
                Missing values in Glucose, BloodPressure,
                SkinThickness, Insulin and BMI were handled using
                median imputation during model development.

                The model itself is loaded from the trained pipeline,
                which includes the optimized preprocessing and learning
                stages.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "🧠 Model Intelligence":

    st.markdown("## 🧠 Model Intelligence")

    st.caption(
        "How the trained model was constructed and optimized."
    )

    st.markdown(
        """
        <div class="glass-card">

            <div class="card-title">
                Logistic Regression + Scaling + SMOTE + GridSearchCV
            </div>

            <p style="color:#94a3b8; line-height:1.7;">
                The final model is a Logistic Regression classifier
                embedded inside an imbalanced-learn pipeline.
                The pipeline standardizes features, applies SMOTE
                during training and selects model hyperparameters
                through cross-validated grid search.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    stages = [
        (
            "01",
            "Data Preparation",
            "Missing clinical values were addressed using median imputation."
        ),
        (
            "02",
            "Train/Test Split",
            "80% of records were used for training and 20% for final testing, with stratification."
        ),
        (
            "03",
            "Feature Scaling",
            "StandardScaler transformed the numerical features to comparable scales."
        ),
        (
            "04",
            "Class Balancing",
            "SMOTE generated synthetic minority-class examples during training."
        ),
        (
            "05",
            "Hyperparameter Search",
            "GridSearchCV tested Logistic Regression configurations using 5-fold cross-validation."
        ),
        (
            "06",
            "Final Evaluation",
            "The optimized pipeline was evaluated on the held-out test set."
        )
    ]

    for number, title, description in stages:

        st.markdown(
            f"""
            <div class="glass-card section-card">

                <div style="
                    color:#67e8f9;
                    font-weight:800;
                    font-size:0.75rem;
                    letter-spacing:0.1em;
                ">
                    {number}
                </div>

                <div style="
                    color:#f8fafc;
                    font-size:1.05rem;
                    font-weight:750;
                    margin-top:0.3rem;
                ">
                    {title}
                </div>

                <div style="
                    color:#94a3b8;
                    margin-top:0.4rem;
                    line-height:1.6;
                ">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🔑 Selected Configuration")

    config = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "C",
            "Solver",
            "Class Weight",
            "Cross-Validation",
            "Optimization Metric"
        ],
        "Selected Value": [
            "Logistic Regression",
            "0.001",
            "liblinear",
            "None",
            "5-fold",
            "F1 Score"
        ]
    })

    st.dataframe(
        config,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EVALUATION PAGE
# ============================================================

elif page == "⚖️ Evaluation":

    st.markdown("## ⚖️ Model Evaluation")

    st.caption(
        "Performance on the held-out test set."
    )

    evaluation = [
        ("Accuracy", "68.83%", "Overall classification accuracy"),
        ("Precision", "53.95%", "Positive predictions that were correct"),
        ("Recall", "75.93%", "Actual positive cases correctly identified"),
        ("F1 Score", "63.08%", "Balance between precision and recall"),
        ("ROC-AUC", "78.69%", "Ability to distinguish the two classes")
    ]

    cols = st.columns(5)

    for col, (label, value, description) in zip(
        cols,
        evaluation
    ):

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

                    <div style="
                        color:#64748b;
                        font-size:0.72rem;
                        line-height:1.4;
                        margin-top:0.35rem;
                    ">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Baseline → Optimized")

    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Baseline": [
            "70.13%",
            "58.70%",
            "50.00%",
            "54.00%"
        ],
        "Optimized": [
            "68.83%",
            "53.95%",
            "75.93%",
            "63.08%"
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">

            <strong>What changed?</strong><br><br>

            Recall increased from <strong>50.00%</strong> to
            <strong>75.93%</strong>, while F1 increased from
            <strong>54.00%</strong> to <strong>63.08%</strong>.

            The trade-off was a reduction in overall accuracy and
            precision. For a screening-oriented application, the
            improvement in recall is particularly important because
            fewer positive cases were missed in the evaluated test set.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Confusion Matrix")

    cm_data = pd.DataFrame(
        [
            ["True Negative", 65, "Correctly predicted negative"],
            ["False Positive", 35, "Negative cases predicted positive"],
            ["False Negative", 13, "Positive cases missed"],
            ["True Positive", 41, "Correctly predicted positive"]
        ],
        columns=[
            "Classification",
            "Count",
            "Meaning"
        ]
    )

    st.dataframe(
        cm_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="warning-box">

            <strong>Medical interpretation:</strong><br><br>

            The optimized model reduced false negatives from
            <strong>27 to 13</strong> on the evaluated test set.
            However, it also produced more false positives.

            This demonstrates the central screening trade-off:
            improving sensitivity can result in more patients being
            referred for additional assessment.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RESPONSIBLE AI PAGE
# ============================================================

elif page == "🛡️ Responsible AI":

    st.markdown("## 🛡️ Responsible AI & Safe Use")

    st.caption(
        "What users should understand before interpreting a prediction."
    )

    st.markdown(
        """
        <div class="danger-box">

            <strong>🚨 This model does not diagnose diabetes.</strong>

            <br><br>

            It produces a machine-learning prediction from the
            information supplied by the user. A prediction can be
            wrong, incomplete or affected by limitations in the
            underlying training data.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="glass-card">

                <div class="card-title">
                    ✅ Appropriate Use
                </div>

                <p style="color:#94a3b8; line-height:1.7;">
                    • Educational exploration<br>
                    • Research demonstrations<br>
                    • Portfolio demonstrations<br>
                    • Preliminary risk screening<br>
                    • Supporting discussion with healthcare professionals
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="glass-card">

                <div class="card-title">
                    ❌ Inappropriate Use
                </div>

                <p style="color:#94a3b8; line-height:1.7;">
                    • Confirming a diagnosis<br>
                    • Replacing laboratory testing<br>
                    • Making treatment decisions automatically<br>
                    • Ignoring professional medical advice<br>
                    • Using the prediction as the only clinical evidence
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🧠 Why can the model be wrong?")

    limitations = [
        (
            "Training Data",
            "The model learned from a relatively small dataset of 768 records."
        ),
        (
            "Generalization",
            "Performance on this dataset does not guarantee the same performance in another population or clinical environment."
        ),
        (
            "Input Quality",
            "Incorrect, outdated or unusual patient measurements can affect predictions."
        ),
        (
            "Model Limitations",
            "Logistic Regression captures linear relationships and may not represent every biological interaction."
        ),
        (
            "Clinical Context",
            "Diabetes assessment involves clinical history, laboratory measurements and professional judgment beyond these eight features."
        )
    ]

    for title, description in limitations:

        st.markdown(
            f"""
            <div class="glass-card section-card">

                <strong style="color:#f8fafc;">
                    {title}
                </strong>

                <div style="
                    color:#94a3b8;
                    margin-top:0.35rem;
                    line-height:1.6;
                ">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="info-box">

            <strong>Best practice:</strong>

            Treat the prediction as one additional piece of information,
            not the final answer. If a result raises concern, the
            appropriate next step is professional medical evaluation
            and clinically appropriate testing.

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

        <strong>🩺 Diabetes Risk Intelligence</strong>

        <br>

        Diabetes Risk Prediction Using Machine Learning

        <br><br>

        Built by
        <strong>
            Olalemi Olaoluwakintan Emmanuel
        </strong>

        <br>

        <span>
            Educational & Research Screening Prototype
        </span>

    </div>
    """,
    unsafe_allow_html=True
)
