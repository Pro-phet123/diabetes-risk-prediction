# ============================================================
# 🩺 DIABETES RISK INTELLIGENCE
# Premium Responsive Streamlit ML Screening Dashboard
#
# Author:
# Olalemi Olaoluwakintan Emmanuel
#
# Project:
# Diabetes Risk Prediction Using Machine Learning
#
# IMPORTANT:
# This application is an educational/research prototype.
# It is NOT a medical diagnostic system.
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
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_risk_prediction_model.pkl"
DATA_PATH = BASE_DIR / "diabetes_nan.csv"


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
# MODEL PERFORMANCE
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

CONFUSION_MATRIX = {
    "TN": 65,
    "FP": 35,
    "FN": 13,
    "TP": 41
}


# ============================================================
# BEST HYPERPARAMETERS
# ============================================================

BEST_PARAMETERS = {
    "C": 0.001,
    "Solver": "liblinear",
    "Class Weight": "None",
    "Cross-Validation": "5-Fold",
    "Optimization Metric": "F1 Score"
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
        "card": "#111114",
        "border": "rgba(255,255,255,0.09)",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "primary": "#60a5fa",
        "secondary": "#a78bfa",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#fb7185"
    },

    "Midnight Blue": {
        "background": "#020617",
        "surface": "#0f172a",
        "surface2": "#172033",
        "card": "#0f172a",
        "border": "rgba(148,163,184,0.12)",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "primary": "#38bdf8",
        "secondary": "#818cf8",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#fb7185"
    },

    "Emerald": {
        "background": "#020807",
        "surface": "#071311",
        "surface2": "#0c1d19",
        "card": "#071311",
        "border": "rgba(52,211,153,0.12)",
        "text": "#ecfdf5",
        "muted": "#94a3b8",
        "primary": "#34d399",
        "secondary": "#2dd4bf",
        "success": "#4ade80",
        "warning": "#fbbf24",
        "danger": "#fb7185"
    },

    "Light": {
        "background": "#f5f7fb",
        "surface": "#ffffff",
        "surface2": "#f8fafc",
        "card": "#ffffff",
        "border": "rgba(15,23,42,0.08)",
        "text": "#0f172a",
        "muted": "#64748b",
        "primary": "#2563eb",
        "secondary": "#7c3aed",
        "success": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626"
    }
}


theme = THEMES[st.session_state.theme]


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    f"""
    <style>

    * {{
        box-sizing: border-box;
    }}

    html, body {{
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
                rgba(37,99,235,0.09),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(124,58,237,0.08),
                transparent 25%
            ),
            {theme["background"]};
        color: {theme["text"]};
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 5rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}

    section[data-testid="stSidebar"] {{
        background: {theme["surface"]};
        border-right: 1px solid {theme["border"]};
    }}

    section[data-testid="stSidebar"] * {{
        color: {theme["text"]};
    }}

    h1, h2, h3, h4, p {{
        color: {theme["text"]};
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 3rem;
        border-radius: 30px;
        margin-bottom: 1.5rem;

        background:
            radial-gradient(
                circle at 85% 15%,
                rgba(96,165,250,0.22),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                {theme["surface"]},
                {theme["surface2"]}
            );

        border: 1px solid {theme["border"]};

        box-shadow:
            0 25px 80px rgba(0,0,0,0.25);
    }}

    .hero-badge {{
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(96,165,250,0.10);
        border: 1px solid rgba(96,165,250,0.22);
        color: {theme["primary"]};
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }}

    .hero-title {{
        margin: 0;
        font-size: clamp(2.4rem, 6vw, 5rem);
        line-height: 0.98;
        letter-spacing: -0.055em;
        font-weight: 850;
        color: {theme["text"]};
    }}

    .hero-subtitle {{
        max-width: 780px;
        margin-top: 1.3rem;
        color: {theme["muted"]};
        font-size: 1.05rem;
        line-height: 1.7;
    }}

    .status {{
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }}

    .status-green {{
        color: {theme["success"]};
        background: rgba(52,211,153,0.09);
        border: 1px solid rgba(52,211,153,0.18);
    }}

    .safety {{
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        background: rgba(251,191,36,0.07);
        border: 1px solid rgba(251,191,36,0.22);
        color: {theme["text"]};
        line-height: 1.65;
    }}

    .safety-title {{
        font-weight: 800;
        color: {theme["warning"]};
        margin-bottom: 0.35rem;
    }}

    .section-title {{
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        margin: 1.5rem 0 0.4rem 0;
    }}

    .section-description {{
        color: {theme["muted"]};
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }}

    .metric-card {{
        padding: 1.25rem;
        min-height: 135px;
        border-radius: 20px;
        background: {theme["card"]};
        border: 1px solid {theme["border"]};
        box-shadow: 0 10px 35px rgba(0,0,0,0.12);
    }}

    .metric-label {{
        color: {theme["muted"]};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }}

    .metric-value {{
        color: {theme["text"]};
        font-size: 2rem;
        font-weight: 850;
        margin-top: 0.3rem;
        letter-spacing: -0.04em;
    }}

    .metric-delta {{
        color: {theme["success"]};
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }}

    .card {{
        padding: 1.5rem;
        border-radius: 22px;
        background: {theme["card"]};
        border: 1px solid {theme["border"]};
        margin-bottom: 1rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.10);
    }}

    .card h3 {{
        margin-top: 0;
        color: {theme["text"]};
    }}

    .card p {{
        color: {theme["muted"]};
        line-height: 1.7;
    }}

    .feature-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid {theme["border"]};
    }}

    .feature-row:last-child {{
        border-bottom: none;
    }}

    .feature-name {{
        color: {theme["text"]};
        font-weight: 600;
    }}

    .feature-value {{
        color: {theme["primary"]};
        font-weight: 800;
        font-family: monospace;
    }}

    .result-positive {{
        padding: 1.5rem;
        border-radius: 22px;
        background: rgba(251,113,133,0.08);
        border: 1px solid rgba(251,113,133,0.28);
        margin-top: 1rem;
    }}

    .result-intermediate {{
        padding: 1.5rem;
        border-radius: 22px;
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.28);
        margin-top: 1rem;
    }}

    .result-negative {{
        padding: 1.5rem;
        border-radius: 22px;
        background: rgba(52,211,153,0.08);
        border: 1px solid rgba(52,211,153,0.28);
        margin-top: 1rem;
    }}

    .result-title {{
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}

    .result-text {{
        color: {theme["muted"]};
        line-height: 1.65;
    }}

    .footer {{
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        color: {theme["muted"]};
        border-top: 1px solid {theme["border"]};
    }}

    @media (max-width: 768px) {{

        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }}

        .hero {{
            padding: 1.7rem;
            border-radius: 22px;
        }}

        .hero-title {{
            font-size: 2.7rem;
        }}

        .hero-subtitle {{
            font-size: 0.95rem;
        }}

        .metric-card {{
            min-height: 110px;
        }}

        .metric-value {{
            font-size: 1.55rem;
        }}
    }}

    </style>
    """
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH.name}"

    try:
        loaded_model = joblib.load(MODEL_PATH)
        return loaded_model, None

    except Exception as error:
        return None, str(error)


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        return None, f"Dataset file not found: {DATA_PATH.name}"

    try:
        loaded_data = pd.read_csv(DATA_PATH)
        return loaded_data, None

    except Exception as error:
        return None, str(error)


model, model_error = load_model()
data, data_error = load_dataset()


# ============================================================
# TARGET DETECTION
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

    st.markdown("## 🩺 Diabetes AI")
    st.caption("Risk Intelligence Platform")

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

    st.markdown("### ⚙️ Appearance")

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

    dataset_status = (
        "🟢 Available"
        if data is not None
        else "🔴 Missing"
    )

    st.markdown(
        f"""
        **Model:** {model_status}

        **Dataset:** {dataset_status}
        """
    )

    st.divider()

    st.caption("Built by")
    st.markdown("**Olalemi Olaoluwakintan Emmanuel**")


# ============================================================
# TOP SYSTEM BAR
# ============================================================

st.html(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:1rem;
    ">

        <span class="status status-green">
            ● SYSTEM ONLINE
        </span>

        <span style="
            color:{theme["muted"]};
            font-size:0.8rem;
        ">
            Diabetes ML Screening Prototype
        </span>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
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
    """
)


# ============================================================
# SAFETY NOTICE
# ============================================================

st.html(
    f"""
    <div class="safety">

        <div class="safety-title">
            ⚠️ Clinical Safety Notice
        </div>

        This application is an educational and research prototype.
        It provides model-based screening estimates and does not
        provide a medical diagnosis. Results must not replace
        professional medical assessment, laboratory testing,
        or established clinical screening protocols.

    </div>
    """
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.html(
        """
        <div class="section-title">
            Executive Overview
        </div>

        <div class="section-description">
            A high-level view of the dataset, model performance,
            optimization strategy and analytical findings.
        </div>
        """
    )

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    total_patients = 768
    diabetic = 268
    non_diabetic = 500

    if data is not None and target_column is not None:

        total_patients = len(data)

        diabetic = int(
            (data[target_column] == 1).sum()
        )

        non_diabetic = int(
            (data[target_column] == 0).sum()
        )

    diabetic_pct = (
        diabetic / total_patients * 100
        if total_patients
        else 0
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    metric_cards = [

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
            "+25.93 percentage points"
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

    for col, label, value, delta in metric_cards:

        with col:

            st.html(
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
                """
            )

    # --------------------------------------------------------
    # KEY INSIGHTS
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-title">
            Why the optimized model matters
        </div>
        """
    )

    left, right = st.columns(2)

    with left:

        st.html(
            f"""
            <div class="card">

                <h3>🎯 Recall became the priority</h3>

                <p>
                    The baseline model achieved
                    <strong>50.00%</strong> recall.
                </p>

                <p>
                    After optimization, recall increased to
                    <strong>75.93%</strong>.
                </p>

                <p>
                    In this screening-oriented experiment,
                    the optimized model identified a larger
                    proportion of positive cases in the
                    evaluated test set.
                </p>

            </div>
            """
        )

    with right:

        st.html(
            f"""
            <div class="card">

                <h3>🧠 Optimization strategy</h3>

                <p>
                    <strong>01 — Scaling</strong><br>
                    StandardScaler standardized feature magnitudes.
                </p>

                <p>
                    <strong>02 — Balancing</strong><br>
                    SMOTE generated synthetic minority-class
                    training examples.
                </p>

                <p>
                    <strong>03 — Search</strong><br>
                    GridSearchCV searched Logistic Regression
                    configurations using F1 scoring.
                </p>

            </div>
            """
        )

    # --------------------------------------------------------
    # BASELINE VS OPTIMIZED
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-title">
            Baseline → Optimized
        </div>
        """
    )

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Baseline": [
            70.13,
            58.70,
            50.00,
            54.00
        ],

        "Optimized": [
            68.83,
            53.95,
            75.93,
            63.08
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The optimized model traded some accuracy and precision "
        "for a substantial increase in recall."
    )

    # --------------------------------------------------------
    # FEATURE INTELLIGENCE
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-title">
            📌 Feature Intelligence
        </div>

        <div class="section-description">
            Logistic Regression coefficients from the optimized
            model. Positive coefficients indicate association with
            the model's positive class; they do not establish causality.
        </div>
        """
    )

    feature_df = pd.DataFrame(
        FEATURE_COEFFICIENTS.items(),
        columns=["Feature", "Coefficient"]
    )

    feature_df = feature_df.sort_values(
        "Coefficient",
        ascending=False
    )

    st.bar_chart(
        feature_df.set_index("Feature")
    )


# ============================================================
# PAGE 2 — RISK ASSESSMENT
# ============================================================

elif page == "🔬 Risk Assessment":

    st.html(
        """
        <div class="section-title">
            🔬 Patient Risk Assessment
        </div>

        <div class="section-description">
            Enter the required patient-level features to generate
            a model-based classification estimate.
        </div>
        """
    )

    if model is None:

        st.error(
            "The trained model could not be loaded."
        )

        if model_error:
            st.code(model_error)

    else:

        st.html(
            f"""
            <div class="card">

                <h3>Model Pipeline</h3>

                <p>
                    Logistic Regression → StandardScaler →
                    SMOTE → GridSearchCV-selected configuration
                </p>

                <p>
                    The model was trained on eight patient-level
                    features from the project dataset.
                </p>

            </div>
            """
        )

        # ----------------------------------------------------
        # INPUT FORM
        # ----------------------------------------------------

        with st.form("risk_assessment_form"):

            st.markdown("### Patient Features")

            c1, c2, c3 = st.columns(3)

            with c1:

                pregnancies = st.number_input(
                    "Pregnancies",
                    min_value=0,
                    max_value=20,
                    value=3,
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

            with c2:

                skin_thickness = st.number_input(
                    "Skin Thickness (mm)",
                    min_value=0.0,
                    max_value=100.0,
                    value=23.0,
                    step=1.0
                )

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

            with c3:

                pedigree = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.47,
                    step=0.01
                )

                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=33,
                    step=1
                )

                st.write("")

                submitted = st.form_submit_button(
                    "🧠 Assess Risk",
                    use_container_width=True
                )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if submitted:

            patient = pd.DataFrame(
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
                columns=EXPECTED_FEATURES
            )

            try:

                prediction = int(
                    model.predict(patient)[0]
                )

                probability = None

                if hasattr(model, "predict_proba"):

                    probability = float(
                        model.predict_proba(patient)[0][1]
                    )

                st.session_state.assessment_result = {

                    "prediction": prediction,

                    "probability": probability,

                    "patient": patient
                }

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        result = st.session_state.assessment_result

        if result is not None:

            prediction = result["prediction"]
            probability = result["probability"]
            patient = result["patient"]

            st.divider()

            st.html(
                """
                <div class="section-title">
                    Assessment Result
                </div>
                """
            )

            if probability is not None:

                probability_percent = probability * 100

                if probability < 0.30:

                    band = "Lower"

                elif probability < 0.60:

                    band = "Intermediate"

                else:

                    band = "Higher"

                r1, r2, r3 = st.columns(3)

                with r1:

                    st.metric(
                        "Model Probability",
                        f"{probability_percent:.1f}%"
                    )

                with r2:

                    st.metric(
                        "Classification",
                        (
                            "Positive (1)"
                            if prediction == 1
                            else "Negative (0)"
                        )
                    )

                with r3:

                    st.metric(
                        "Risk Band",
                        band
                    )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                # ------------------------------------------------
                # RESULT MESSAGE
                # ------------------------------------------------

                if prediction == 1:

                    st.html(
                        f"""
                        <div class="result-positive">

                            <div class="result-title">
                                ⚠️ Positive Model Classification
                            </div>

                            <div class="result-text">

                                The model classified this patient
                                as <strong>Class 1</strong>.

                                <br><br>

                                Model-estimated positive-class
                                probability:
                                <strong>
                                    {probability_percent:.1f}%
                                </strong>

                                <br><br>

                                This is a screening flag only and
                                must not be interpreted as a diagnosis.

                            </div>

                        </div>
                        """
                    )

                elif band == "Intermediate":

                    st.html(
                        f"""
                        <div class="result-intermediate">

                            <div class="result-title">
                                🟡 Intermediate Model Estimate
                            </div>

                            <div class="result-text">

                                The model classified this patient
                                as <strong>Class 0</strong>, but the
                                estimated positive-class probability
                                is intermediate.

                                <br><br>

                                Estimated probability:
                                <strong>
                                    {probability_percent:.1f}%
                                </strong>

                                <br><br>

                                Clinical assessment should determine
                                the appropriate next step.

                            </div>

                        </div>
                        """
                    )

                else:

                    st.html(
                        f"""
                        <div class="result-negative">

                            <div class="result-title">
                                ✓ Negative Model Classification
                            </div>

                            <div class="result-text">

                                The model classified this patient
                                as <strong>Class 0</strong>.

                                <br><br>

                                Model-estimated positive-class
                                probability:
                                <strong>
                                    {probability_percent:.1f}%
                                </strong>

                                <br><br>

                                A negative model result does not
                                rule out diabetes.

                            </div>

                        </div>
                        """
                    )

            # ----------------------------------------------------
            # INPUT SUMMARY
            # ----------------------------------------------------

            st.markdown("### Patient Input Summary")

            st.dataframe(
                patient.T.rename(
                    columns={0: "Patient Value"}
                ),
                use_container_width=True
            )

            # ----------------------------------------------------
            # DOWNLOAD REPORT
            # ----------------------------------------------------

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
{"Positive (1)" if prediction == 1 else "Negative (0)"}

Model Probability:
{report_probability}

Risk Band:
{band if probability is not None else "Unavailable"}


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
assessment, laboratory testing, or established clinical
screening protocols.

Model probability should not be interpreted as a clinically
calibrated probability unless appropriate calibration and
external validation have been performed.
"""

            st.download_button(
                "📄 Download Assessment Report",
                data=report,
                file_name="diabetes_risk_assessment.txt",
                mime="text/plain",
                use_container_width=True
            )


# ============================================================
# PAGE 3 — DATASET
# ============================================================

elif page == "📊 Dataset":

    st.html(
        """
        <div class="section-title">
            📊 Dataset Explorer
        </div>

        <div class="section-description">
            Explore the dataset used during development of the
            diabetes risk prediction model.
        </div>
        """
    )

    if data is None:

        st.error(
            "Dataset could not be loaded."
        )

        if data_error:
            st.code(data_error)

    else:

        positive = 0
        negative = 0

        if target_column is not None:

            positive = int(
                (data[target_column] == 1).sum()
            )

            negative = int(
                (data[target_column] == 0).sum()
            )

        d1, d2, d3, d4 = st.columns(4)

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
                "Diabetes",
                f"{positive:,}"
            )

        with d4:
            st.metric(
                "No Diabetes",
                f"{negative:,}"
            )

        st.divider()

        # ----------------------------------------------------
        # TARGET DISTRIBUTION
        # ----------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.markdown("### Target Distribution")

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
                target_display.set_index("Class")
            )

        with right:

            st.markdown("### Missing Values")

            missing = (
                data.isnull()
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            missing_positive = missing[
                missing > 0
            ]

            if len(missing_positive):

                st.bar_chart(
                    missing_positive
                )

            else:

                st.success(
                    "✓ No missing values detected."
                )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        st.markdown("### 🔍 Data Quality")

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

        # ----------------------------------------------------
        # DATASET PREVIEW
        # ----------------------------------------------------

        st.divider()

        st.markdown("### Dataset Preview")

        rows_to_show = st.slider(
            "Rows to display",
            min_value=10,
            max_value=min(200, len(data)),
            value=min(50, len(data)),
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

    st.html(
        """
        <div class="section-title">
            🧠 Model Intelligence
        </div>

        <div class="section-description">
            Technical view of the machine-learning architecture,
            optimization strategy and learned feature coefficients.
        </div>
        """
    )

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="card">

            <h3>Model Architecture</h3>

            <p>
                <strong>Input Features</strong>
                →
                <strong>StandardScaler</strong>
                →
                <strong>SMOTE</strong>
                →
                <strong>Logistic Regression</strong>
            </p>

            <p>
                GridSearchCV was used with 5-fold cross-validation
                and F1 scoring to select the Logistic Regression
                configuration.
            </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    st.markdown("### ⚙️ Selected Configuration")

    parameter_df = pd.DataFrame({

        "Parameter": list(BEST_PARAMETERS.keys()),

        "Selected Value": list(
            BEST_PARAMETERS.values()
        )
    })

    st.dataframe(
        parameter_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE COEFFICIENTS
    # --------------------------------------------------------

    st.markdown("### 📌 Learned Feature Coefficients")

    coefficients = pd.DataFrame(
        FEATURE_COEFFICIENTS.items(),
        columns=[
            "Feature",
            "Coefficient"
        ]
    )

    coefficients = coefficients.sort_values(
        "Coefficient",
        ascending=False
    )

    st.bar_chart(
        coefficients.set_index("Feature")
    )

    st.caption(
        "These coefficients describe the direction and relative "
        "magnitude of association learned by the Logistic Regression "
        "model after feature scaling. They do not prove causation."
    )

    st.dataframe(
        coefficients,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 5 — EVALUATION
# ============================================================

elif page == "⚖️ Evaluation":

    st.html(
        """
        <div class="section-title">
            ⚖️ Model Evaluation
        </div>

        <div class="section-description">
            Evaluation of the optimized model on the held-out
            test set.
        </div>
        """
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    e1, e2, e3, e4, e5 = st.columns(5)

    evaluation_cards = [

        (
            e1,
            "ACCURACY",
            "68.83%"
        ),

        (
            e2,
            "PRECISION",
            "53.95%"
        ),

        (
            e3,
            "RECALL",
            "75.93%"
        ),

        (
            e4,
            "F1 SCORE",
            "63.08%"
        ),

        (
            e5,
            "ROC-AUC",
            "78.69%"
        )
    ]

    for col, label, value in evaluation_cards:

        with col:

            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-value">
                        {value}
                    </div>

                </div>
                """
            )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("### Confusion Matrix")

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

    st.markdown("### Interpretation")

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
    # METRIC RECONSTRUCTION
    # --------------------------------------------------------

    total = tn + fp + fn + tp

    calculated_accuracy = (
        (tn + tp) / total
    )

    calculated_precision = (
        tp / (tp + fp)
    )

    calculated_recall = (
        tp / (tp + fn)
    )

    calculated_f1 = (
        2 *
        calculated_precision *
        calculated_recall /
        (
            calculated_precision +
            calculated_recall
        )
    )

    reconstructed = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "From Confusion Matrix": [
            calculated_accuracy,
            calculated_precision,
            calculated_recall,
            calculated_f1
        ]
    })

    st.markdown("### 📐 Reconstructed Metrics")

    st.dataframe(
        reconstructed.style.format({
            "From Confusion Matrix": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SCREENING TRADE-OFF
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="card">

            <h3>🎯 Screening Trade-off</h3>

            <p>
                The optimized model produced
                <strong>{fn} false negatives</strong>
                and
                <strong>{fp} false positives</strong>
                on the evaluated test set.
            </p>

            <p>
                Recall increased from
                <strong>50.00%</strong>
                in the baseline model to
                <strong>75.93%</strong>
                after optimization.
            </p>

            <p>
                Higher recall does not by itself establish
                clinical safety. External validation,
                calibration, specificity, clinical thresholds,
                fairness assessment and prospective evaluation
                would still be required before clinical use.
            </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # BASELINE VS OPTIMIZED
    # --------------------------------------------------------

    st.markdown("### Baseline vs Optimized")

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Baseline": [
            70.13,
            58.70,
            50.00,
            54.00
        ],

        "Optimized": [
            68.83,
            53.95,
            75.93,
            63.08
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 6 — RESPONSIBLE AI
# ============================================================

elif page == "🛡️ Responsible AI":

    st.html(
        """
        <div class="section-title">
            🛡️ Responsible AI
        </div>

        <div class="section-description">
            Important limitations and requirements surrounding
            the use of this machine-learning prototype.
        </div>
        """
    )

    st.html(
        f"""
        <div class="card">

            <h3>⚠️ This is not a medical diagnostic system</h3>

            <p>
                This project demonstrates the application of
                machine learning to a diabetes classification
                problem. It has not been clinically validated
                and should not be used to diagnose, treat or
                clear a patient.
            </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown("### Known Limitations")

    limitations = pd.DataFrame({

        "Area": [
            "Dataset Size",
            "External Validation",
            "Clinical Calibration",
            "Population Generalization",
            "Threshold Selection",
            "Clinical Oversight"
        ],

        "Status": [
            "768 records",
            "Not performed",
            "Not performed",
            "Not established",
            "Not clinically selected",
            "Required"
        ]
    })

    st.dataframe(
        limitations,
        use_container_width=True,
        hide_index=True
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
    # RESEARCHER'S NOTE
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="card">

            <h3>🧑🏽‍💻 Researcher's Note</h3>

            <p>
                The purpose of this application is to demonstrate
                how a machine-learning model can move from
                experimentation in a notebook into an interactive
                analytical interface.
            </p>

            <p>
                The application should be viewed as a portfolio
                and research prototype rather than a clinical
                decision-making system.
            </p>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    f"""
    <div class="footer">

        <strong>
            🩺 Diabetes Risk Intelligence
        </strong>

        <br><br>

        Diabetes Risk Prediction Using Machine Learning

        <br><br>

        Built by
        <strong>
            Olalemi Olaoluwakintan Emmanuel
        </strong>

        <br>

        Educational & Research Prototype

    </div>
    """
)
