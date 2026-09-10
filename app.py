# ============================================================
# 🩺 DIABETES RISK INTELLIGENCE PLATFORM
# Interactive Diabetes Risk Prediction
# Author: Olalemi Olaoluwakintan Emmanuel
#
# Model:
# StandardScaler + SMOTE + GridSearchCV + Logistic Regression
#
# IMPORTANT:
# The saved .pkl file contains the complete trained pipeline.
# Do NOT manually scale or SMOTE individual prediction inputs.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

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
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_risk_prediction_model.pkl"
DATA_PATH = BASE_DIR / "diabetes_nan.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None


model = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None


data = load_dataset()


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(8, 145, 178, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at bottom left,
                rgba(109, 40, 217, 0.08),
                transparent 30%
            ),
            #080b10;
        color: #f8fafc;
    }

    .main {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] {
        background: #0b0f15;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero-container {
        padding: 3rem 1rem 2rem 1rem;
        max-width: 1100px;
        margin: auto;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(8,145,178,0.12);
        border: 1px solid rgba(8,145,178,0.30);
        color: #67e8f9;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2.8rem, 7vw, 6rem);
        line-height: 0.95;
        font-weight: 850;
        letter-spacing: -0.055em;
        margin: 0;
        color: #f8fafc;
    }

    .hero-subtitle {
        max-width: 760px;
        font-size: 1.12rem;
        line-height: 1.7;
        color: #94a3b8;
        margin-top: 1.5rem;
    }

    /* ======================================================
       CARDS
       ====================================================== */

    .glass-card {
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.20);
        backdrop-filter: blur(12px);
    }

    .section-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(15,23,42,0.95),
            rgba(15,23,42,0.65)
        );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.25rem;
        min-height: 145px;
    }

    .metric-label {
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #64748b;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.3rem;
    }

    .metric-delta {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    /* ======================================================
       STATUS
       ====================================================== */

    .status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .status-green {
        color: #86efac;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.20);
    }

    .status-yellow {
        color: #fde68a;
        background: rgba(234,179,8,0.10);
        border: 1px solid rgba(234,179,8,0.20);
    }

    .status-red {
        color: #fca5a5;
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.20);
    }

    /* ======================================================
       RESULT
       ====================================================== */

    .prediction-result {
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(255,255,255,0.09);
    }

    .prediction-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }

    .prediction-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .prediction-score {
        font-size: 3.2rem;
        font-weight: 850;
        margin: 0.5rem 0;
    }

    .prediction-note {
        color: #94a3b8;
        max-width: 650px;
        margin: auto;
        line-height: 1.7;
    }

    /* ======================================================
       INPUTS
       ====================================================== */

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div,
    .stFileUploader {
        border-radius: 12px !important;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #64748b;
        padding: 3rem 0 1rem 0;
        font-size: 0.82rem;
    }

    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 768px) {

        .hero-container {
            padding: 2rem 0.5rem;
        }

        .hero-title {
            font-size: 3rem;
        }

        .hero-subtitle {
            font-size: 0.98rem;
        }

        .metric-card {
            min-height: 120px;
        }

        .prediction-score {
            font-size: 2.5rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="
                font-size: 1.4rem;
                font-weight: 800;
                color: #f8fafc;
            ">
                🩺 Diabetes AI
            </div>

            <div style="
                color:#64748b;
                font-size:0.82rem;
                margin-top:4px;
            ">
                Risk Intelligence Platform
            </div>
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

    st.divider()

    st.markdown("### ⚙️ System")

    model_status = (
        "🟢 Online"
        if model is not None
        else "🔴 Model unavailable"
    )

    dataset_status = (
        "🟢 Available"
        if data is not None
        else "🟡 Not loaded"
    )

    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**Dataset:** {dataset_status}")

    st.divider()

    st.caption("Built by")
    st.markdown("**Olalemi Olaoluwakintan Emmanuel**")


# ============================================================
# COMMON DATA
# ============================================================

TOTAL_RECORDS = 768
HEALTHY_COUNT = 500
DIABETIC_COUNT = 268

BASELINE_ACCURACY = 0.7013
BASELINE_PRECISION = 0.5870
BASELINE_RECALL = 0.5000
BASELINE_F1 = 0.5400

OPTIMIZED_ACCURACY = 0.6883
OPTIMIZED_PRECISION = 0.5395
OPTIMIZED_RECALL = 0.7593
OPTIMIZED_F1 = 0.6308
ROC_AUC = 0.7869


# ============================================================
# HERO
# ============================================================

def render_hero():

    st.markdown(
        """
        <div class="hero-container">

            <div class="hero-badge">
                🩺 AI-POWERED HEALTH ANALYTICS
            </div>

            <h1 class="hero-title">
                Diabetes Risk<br>
                Intelligence
            </h1>

            <p class="hero-subtitle">
                An interactive machine-learning screening platform
                transforming patient-level health features into
                model-based diabetes risk estimates.
            </p>

            <br>

            <span class="status status-green">
                ● RESEARCH & SCREENING PROTOTYPE
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SAFETY NOTICE
# ============================================================

def render_safety_notice():

    st.warning(
        """
        ⚠️ **Clinical Safety Notice**

        This application provides a machine-learning risk estimate,
        not a medical diagnosis.

        Results should be interpreted alongside professional clinical
        assessment, laboratory testing, patient history and established
        screening protocols.

        **Do not make treatment or medication decisions based solely
        on this application.**
        """
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    render_hero()

    render_safety_notice()

    st.markdown("## Executive Overview")

    st.markdown(
        """
        A high-level view of the dataset, model performance,
        optimization strategy and analytical findings.
        """
    )

    cols = st.columns(5)

    metrics = [
        ("PATIENT RECORDS", "768", "Development dataset"),
        ("DIABETIC", "268", "34.9% of records"),
        ("OPTIMIZED RECALL", "75.93%", "+25.93 percentage points"),
        ("F1 SCORE", "63.08%", "Optimized model"),
        ("ROC-AUC", "78.69%", "Discrimination")
    ]

    for col, (label, value, delta) in zip(cols, metrics):

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

    st.markdown("")

    # --------------------------------------------------------
    # WHY OPTIMIZED MODEL MATTERS
    # --------------------------------------------------------

    st.markdown("## 🎯 Why the optimized model matters")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🎯 Recall became the priority</h3>

            <p style="color:#94a3b8; line-height:1.7;">
            The baseline model achieved <strong>50.00%</strong> recall.
            </p>

            <p style="color:#94a3b8; line-height:1.7;">
            After optimization, recall increased to
            <strong>75.93%</strong>.
            </p>

            <p style="color:#94a3b8; line-height:1.7;">
            In a screening-oriented setting, the optimized model
            identifies a larger proportion of positive cases in
            the evaluated test set.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="glass-card">

            <h3>🧠 Optimization strategy</h3>

            <p style="color:#94a3b8; line-height:1.8;">
            <strong>01 — Scaling</strong><br>
            StandardScaler normalizes feature magnitudes.
            </p>

            <p style="color:#94a3b8; line-height:1.8;">
            <strong>02 — Balancing</strong><br>
            SMOTE improves minority-class representation
            during model training.
            </p>

            <p style="color:#94a3b8; line-height:1.8;">
            <strong>03 — Search</strong><br>
            GridSearchCV evaluates Logistic Regression
            configurations using F1-oriented optimization.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.markdown("## 📈 Baseline → Optimized")

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "Baseline": [
                BASELINE_ACCURACY,
                BASELINE_PRECISION,
                BASELINE_RECALL,
                BASELINE_F1
            ],
            "Optimized": [
                OPTIMIZED_ACCURACY,
                OPTIMIZED_PRECISION,
                OPTIMIZED_RECALL,
                OPTIMIZED_F1
            ]
        }
    )

    st.dataframe(
        comparison.style.format(
            {
                "Baseline": "{:.2%}",
                "Optimized": "{:.2%}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The optimized model traded some accuracy and precision "
        "for a substantial improvement in recall."
    )

    # --------------------------------------------------------
    # FEATURE INTELLIGENCE
    # --------------------------------------------------------

    st.markdown("## 📌 Feature Intelligence")

    feature_values = {
        "Glucose": 0.156972,
        "BMI": 0.102375,
        "SkinThickness": 0.062338,
        "Insulin": 0.058959,
        "Age": 0.058537,
        "Pregnancies": 0.053139,
        "BloodPressure": 0.045498,
        "DiabetesPedigreeFunction": 0.043508
    }

    feature_df = pd.DataFrame(
        {
            "Feature": list(feature_values.keys()),
            "Coefficient": list(feature_values.values())
        }
    ).sort_values(
        "Coefficient",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        feature_df["Feature"],
        feature_df["Coefficient"]
    )

    ax.set_xlabel("Model Coefficient")
    ax.set_title("Logistic Regression Feature Coefficients")

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        "Higher positive coefficients indicate stronger positive "
        "association with the model's predicted positive class. "
        "They do not establish causality."
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "🔬 Risk Assessment":

    render_hero()

    render_safety_notice()

    st.markdown("## 🔬 Patient Risk Assessment")

    st.markdown(
        """
        Enter the patient's available health measurements below.
        The trained machine-learning pipeline will generate a
        **model-based diabetes risk classification**.
        """
    )

    if model is None:

        st.error(
            "The trained model could not be loaded. "
            "Make sure `diabetes_risk_prediction_model.pkl` "
            "is in the same folder as this Streamlit application."
        )

        st.stop()

    # ========================================================
    # ASSESSMENT MODE
    # ========================================================

    assessment_mode = st.radio(
        "Assessment mode",
        [
            "👤 Single Patient",
            "📂 Batch CSV Prediction"
        ],
        horizontal=True
    )

    # ========================================================
    # SINGLE PATIENT
    # ========================================================

    if assessment_mode == "👤 Single Patient":

        st.markdown("### Patient Information")

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
                help="Number of pregnancies."
            )

            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0.0,
                max_value=400.0,
                value=120.0,
                step=1.0,
                help="Plasma glucose concentration."
            )

            blood_pressure = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=0.0,
                max_value=250.0,
                value=70.0,
                step=1.0,
                help="Diastolic blood pressure measurement used by the training dataset."
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Triceps skin fold thickness."
            )

        with col2:

            insulin = st.number_input(
                "Insulin (μU/mL)",
                min_value=0.0,
                max_value=1000.0,
                value=80.0,
                step=1.0,
                help="Serum insulin level."
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=80.0,
                value=25.0,
                step=0.1,
                help="Body Mass Index."
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.47,
                step=0.01,
                help="Diabetes pedigree function value."
            )

            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=30,
                step=1,
                help="Patient age."
            )

        st.markdown("---")

        st.markdown(
            """
            **Before predicting**

            Ensure that the measurements entered correspond to the
            definitions and units expected by the training dataset.
            """
        )

        predict_button = st.button(
            "🔍 Assess Diabetes Risk",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            patient_data = pd.DataFrame(
                [
                    {
                        "Pregnancies": pregnancies,
                        "Glucose": glucose,
                        "BloodPressure": blood_pressure,
                        "SkinThickness": skin_thickness,
                        "Insulin": insulin,
                        "BMI": bmi,
                        "DiabetesPedigreeFunction": dpf,
                        "Age": age
                    }
                ]
            )

            try:

                prediction = model.predict(patient_data)[0]

                # ------------------------------------------------
                # PROBABILITY
                # ------------------------------------------------

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        patient_data
                    )[0]

                    # Probability of positive class
                    risk_probability = float(probabilities[1])

                else:

                    risk_probability = None

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                if int(prediction) == 1:

                    st.markdown(
                        """
                        <div class="prediction-result">

                            <div class="prediction-icon">
                                ⚠️
                            </div>

                            <div class="prediction-title">
                                Elevated Diabetes Risk
                            </div>

                            <div class="prediction-note">
                                The model classified this input as
                                belonging to the positive class.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="prediction-result">

                            <div class="prediction-icon">
                                🟢
                            </div>

                            <div class="prediction-title">
                                Lower Predicted Risk
                            </div>

                            <div class="prediction-note">
                                The model classified this input as
                                belonging to the negative class.
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if risk_probability is not None:

                    st.markdown("### Model Probability")

                    probability_percent = risk_probability * 100

                    st.progress(
                        risk_probability
                    )

                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            font-size:2.4rem;
                            font-weight:800;
                            margin:0.5rem 0;
                        ">
                            {probability_percent:.2f}%
                        </div>

                        <div style="
                            text-align:center;
                            color:#94a3b8;
                        ">
                            Estimated probability of the positive
                            model class
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # INPUT SUMMARY
                # ------------------------------------------------

                with st.expander("📋 View submitted measurements"):

                    st.dataframe(
                        patient_data,
                        use_container_width=True,
                        hide_index=True
                    )

                # ------------------------------------------------
                # CLINICAL INTERPRETATION
                # ------------------------------------------------

                st.markdown("### 🩺 Interpretation")

                if int(prediction) == 1:

                    st.warning(
                        """
                        The model identified an elevated predicted risk
                        based on the information supplied.

                        This result should be treated as a prompt for
                        appropriate professional assessment and,
                        where clinically indicated, confirmatory testing.
                        """
                    )

                else:

                    st.info(
                        """
                        The model did not classify the supplied inputs
                        as belonging to the positive class.

                        A lower model-predicted risk does not guarantee
                        the absence of diabetes. Clinical symptoms,
                        medical history and appropriate testing should
                        still be considered.
                        """
                    )

                st.caption(
                    "Model output is an analytical screening estimate "
                    "and must not be interpreted as a diagnosis."
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {str(e)}"
                )

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    else:

        st.markdown("### 📂 Batch Risk Prediction")

        st.markdown(
            """
            Upload a CSV containing multiple patient records and the
            application will generate model predictions for each row.

            Your CSV should contain these **8 feature columns**:
            """
        )

        required_features = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        st.code(
            ", ".join(required_features),
            language="text"
        )

        st.info(
            """
            💡 The uploaded CSV should contain patient feature values.
            You do not need to add `outcome(target)` because that is the
            value the model is predicting.
            """
        )

        uploaded_file = st.file_uploader(
            "Upload patient CSV",
            type=["csv"],
            help="CSV containing the 8 model input features."
        )

        if uploaded_file is not None:

            try:

                batch_data = pd.read_csv(uploaded_file)

                st.markdown("### Uploaded Data")

                st.dataframe(
                    batch_data.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                missing_features = [
                    feature
                    for feature in required_features
                    if feature not in batch_data.columns
                ]

                if missing_features:

                    st.error(
                        "The following required columns are missing: "
                        + ", ".join(missing_features)
                    )

                else:

                    prediction_data = batch_data[
                        required_features
                    ].copy()

                    # --------------------------------------------
                    # NUMERIC VALIDATION
                    # --------------------------------------------

                    for feature in required_features:

                        prediction_data[feature] = pd.to_numeric(
                            prediction_data[feature],
                            errors="coerce"
                        )

                    invalid_rows = prediction_data[
                        prediction_data.isnull().any(axis=1)
                    ]

                    if len(invalid_rows) > 0:

                        st.error(
                            f"{len(invalid_rows)} row(s) contain "
                            "missing or non-numeric values. "
                            "Please clean the CSV before prediction."
                        )

                    else:

                        batch_predict_button = st.button(
                            "🚀 Run Batch Prediction",
                            type="primary",
                            use_container_width=True
                        )

                        if batch_predict_button:

                            try:

                                predictions = model.predict(
                                    prediction_data
                                )

                                if hasattr(model, "predict_proba"):

                                    probabilities = model.predict_proba(
                                        prediction_data
                                    )[:, 1]

                                else:

                                    probabilities = np.full(
                                        len(predictions),
                                        np.nan
                                    )

                                results = batch_data.copy()

                                results[
                                    "Predicted Outcome"
                                ] = predictions

                                results[
                                    "Prediction Label"
                                ] = np.where(
                                    predictions == 1,
                                    "Elevated Risk",
                                    "Lower Predicted Risk"
                                )

                                results[
                                    "Positive Class Probability"
                                ] = probabilities

                                results[
                                    "Positive Class Probability"
                                ] = (
                                    results[
                                        "Positive Class Probability"
                                    ] * 100
                                ).round(2)

                                # --------------------------------
                                # SUMMARY
                                # --------------------------------

                                total = len(results)

                                positive_count = int(
                                    np.sum(predictions == 1)
                                )

                                negative_count = int(
                                    np.sum(predictions == 0)
                                )

                                c1, c2, c3 = st.columns(3)

                                with c1:

                                    st.metric(
                                        "Records Processed",
                                        total
                                    )

                                with c2:

                                    st.metric(
                                        "Elevated Risk",
                                        positive_count
                                    )

                                with c3:

                                    st.metric(
                                        "Lower Predicted Risk",
                                        negative_count
                                    )

                                # --------------------------------
                                # RESULTS
                                # --------------------------------

                                st.markdown(
                                    "### 📊 Prediction Results"
                                )

                                st.dataframe(
                                    results,
                                    use_container_width=True,
                                    hide_index=True
                                )

                                # --------------------------------
                                # DOWNLOAD
                                # --------------------------------

                                csv_output = results.to_csv(
                                    index=False
                                ).encode("utf-8")

                                st.download_button(
                                    label="⬇️ Download Prediction Results",
                                    data=csv_output,
                                    file_name="diabetes_risk_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                                st.success(
                                    "Batch prediction completed successfully."
                                )

                                st.warning(
                                    """
                                    Batch predictions are model-generated
                                    screening estimates. They should not
                                    be used as automated medical diagnoses
                                    or treatment decisions.
                                    """
                                )

                            except Exception as e:

                                st.error(
                                    f"Batch prediction failed: {str(e)}"
                                )

            except Exception as e:

                st.error(
                    f"Unable to read the uploaded CSV: {str(e)}"
                )


# ============================================================
# DATASET
# ============================================================

elif page == "📊 Dataset":

    render_hero()

    st.markdown("## 📊 Dataset Intelligence")

    st.markdown(
        """
        The model was developed using a dataset containing
        **768 patient records** and eight predictive features.
        """
    )

    cols = st.columns(4)

    dataset_metrics = [
        ("TOTAL RECORDS", "768"),
        ("FEATURES", "8"),
        ("NO DIABETES", "500"),
        ("DIABETES", "268")
    ]

    for col, (label, value) in zip(cols, dataset_metrics):

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

    st.markdown("")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("### Target Distribution")

    distribution = pd.DataFrame(
        {
            "Outcome": [
                "No Diabetes",
                "Diabetes"
            ],
            "Patients": [
                HEALTHY_COUNT,
                DIABETIC_COUNT
            ]
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        distribution["Outcome"],
        distribution["Patients"]
    )

    ax.set_ylabel("Number of Patients")
    ax.set_title("Diabetes Outcome Distribution")

    plt.tight_layout()

    st.pyplot(fig)

    st.markdown(
        """
        **Class 0:** 500 patients — 65.10%

        **Class 1:** 268 patients — 34.90%

        The dataset therefore contains a moderate class imbalance,
        with the non-diabetic class representing the majority.
        """
    )

    # --------------------------------------------------------
    # DATASET TABLE
    # --------------------------------------------------------

    if data is not None:

        st.markdown("### Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Missing Values Before Cleaning")

        missing_data = pd.DataFrame(
            {
                "Feature": [
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI"
                ],
                "Missing Values": [
                    5,
                    35,
                    227,
                    374,
                    11
                ]
            }
        )

        st.dataframe(
            missing_data,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        """
        Missing numerical values were handled using median imputation.
        This preserved all 768 patient records instead of removing rows
        containing incomplete measurements.
        """
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "🧠 Model Intelligence":

    render_hero()

    st.markdown("## 🧠 Model Intelligence")

    st.markdown(
        """
        The final model is an optimized Logistic Regression pipeline
        developed to improve classification performance while placing
        greater emphasis on identifying positive cases.
        """
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown("### ⚙️ Final Model Pipeline")

    pipeline_steps = [
        ("01", "StandardScaler", "Standardizes numerical features."),
        ("02", "SMOTE", "Generates synthetic minority-class training samples."),
        ("03", "Logistic Regression", "Performs binary classification."),
        ("04", "GridSearchCV", "Selects the best tested hyperparameter configuration.")
    ]

    for number, title, description in pipeline_steps:

        st.markdown(
            f"""
            <div class="section-card">

                <div style="
                    color:#67e8f9;
                    font-weight:800;
                ">
                    {number}
                </div>

                <h4 style="margin:5px 0;">
                    {title}
                </h4>

                <div style="color:#94a3b8;">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # BEST PARAMETERS
    # --------------------------------------------------------

    st.markdown("### 🔧 Selected Hyperparameters")

    parameters = pd.DataFrame(
        {
            "Parameter": [
                "C",
                "Solver",
                "Class Weight"
            ],
            "Selected Value": [
                "0.001",
                "liblinear",
                "None"
            ]
        }
    )

    st.dataframe(
        parameters,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These parameters were selected by the GridSearchCV process "
        "using 5-fold cross-validation and F1 scoring."
    )

    # --------------------------------------------------------
    # WHY SMOTE?
    # --------------------------------------------------------

    st.markdown("### ⚖️ Why SMOTE?")

    st.markdown(
        """
        The target distribution was not perfectly balanced:

        **500 non-diabetic records vs 268 diabetic records.**

        SMOTE was therefore applied **inside the training pipeline**
        to improve representation of the minority class during model
        training.

        Importantly, SMOTE was not applied directly to the test set.
        This prevents synthetic samples from contaminating the
        independent evaluation data.
        """
    )

    # --------------------------------------------------------
    # WHY SCALING?
    # --------------------------------------------------------

    st.markdown("### 📏 Why StandardScaler?")

    st.markdown(
        """
        The features operate on very different numerical scales.

        For example, Age is measured in years, while Insulin can have
        values in the hundreds.

        StandardScaler puts the numerical features onto a comparable
        scale before Logistic Regression learns its coefficients.
        """
    )


# ============================================================
# EVALUATION
# ============================================================

elif page == "⚖️ Evaluation":

    render_hero()

    st.markdown("## ⚖️ Model Evaluation")

    st.markdown(
        """
        The model was evaluated on a held-out test set that was not
        used during final model fitting.
        """
    )

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    st.markdown("### Optimized Model Performance")

    cols = st.columns(5)

    optimized_metrics = [
        ("ACCURACY", "68.83%"),
        ("PRECISION", "53.95%"),
        ("RECALL", "75.93%"),
        ("F1 SCORE", "63.08%"),
        ("ROC-AUC", "78.69%")
    ]

    for col, (label, value) in zip(
        cols,
        optimized_metrics
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

                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("### Confusion Matrix")

    cm = np.array(
        [
            [65, 35],
            [13, 41]
        ]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No Diabetes",
            "Actual Diabetes"
        ],
        columns=[
            "Predicted No Diabetes",
            "Predicted Diabetes"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("True Negatives", "65")

    with c2:
        st.metric("False Positives", "35")

    with c3:
        st.metric("False Negatives", "13")

    with c4:
        st.metric("True Positives", "41")

    st.markdown("### 🩺 Medical Context Interpretation")

    st.markdown(
        """
        **The optimized model shows:**

        - **Higher Recall:** 75.93% of actual positive cases in the
          test set were correctly identified.
        - **Lower Precision:** Some non-diabetic patients were also
          classified as positive, resulting in 35 false positives.
        - **Key Trade-off:** The model accepts additional false
          positives in exchange for substantially fewer false negatives.
        - **Screening Perspective:** In a screening-oriented use case,
          improving recall can be more meaningful than maximizing
          overall accuracy.
        """
    )

    # --------------------------------------------------------
    # BASELINE VS OPTIMIZED
    # --------------------------------------------------------

    st.markdown("### 📈 Baseline vs Optimized")

    comparison_df = pd.DataFrame(
        {
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
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "The most notable improvement was Recall, increasing "
        "from 50.00% to 75.93%."
    )


# ============================================================
# RESPONSIBLE AI
# ============================================================

elif page == "🛡️ Responsible AI":

    render_hero()

    st.markdown("## 🛡️ Responsible AI")

    st.markdown(
        """
        ### What this application can do

        This application can be used to generate **machine-learning
        based diabetes risk estimates** from the required patient
        features.

        It can support:

        - Educational demonstrations
        - Research experimentation
        - Portfolio demonstrations
        - Preliminary risk-screening workflows
        - Batch analytical exploration

        ### What this application cannot do

        The model cannot independently establish that a person has
        diabetes.

        It does not replace:

        - Laboratory testing
        - Physician assessment
        - Medical history
        - Physical examination
        - Established clinical guidelines
        - Professional diagnosis or treatment

        ### ⚠️ Understanding the prediction

        A positive prediction means the model classified the supplied
        information as belonging to the positive class.

        A negative prediction does **not** prove that diabetes is absent.

        Machine-learning systems can make mistakes, and this model was
        developed and evaluated on a relatively small dataset.

        ### 🎯 Why Recall matters

        The optimized model achieved **75.93% recall** on the evaluated
        test set.

        That means the model correctly identified 75.93% of the actual
        positive cases in that test set.

        However, it also produced false positives.

        Therefore, the output should be treated as a **screening signal
        requiring appropriate human interpretation**, not as a final
        clinical decision.

        ### 🔐 Data responsibility

        Avoid entering personally identifiable information into the
        application.

        When using the batch prediction feature, upload only the
        measurements necessary for the analysis.

        ### 🧠 Human oversight

        The safest way to use this system is:

        **Patient information → Model estimate → Professional review →
        Appropriate clinical testing/decision**

        The model supports the process. It does not replace the
        professional responsible for the patient's care.
        """
    )

    st.warning(
        """
        **Important:** This application is an educational and research
        screening prototype. Any real-world clinical deployment would
        require additional validation, appropriate clinical governance,
        privacy controls, regulatory consideration and prospective
        evaluation before use with patients.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong style="color:#94a3b8;">
            🩺 Diabetes Risk Intelligence
        </strong>

        <br><br>

        Diabetes Risk Prediction Using Machine Learning

        <br><br>

        Built by
        <strong style="color:#cbd5e1;">
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
