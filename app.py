# ============================================================
# DIABETES RISK INTELLIGENCE
# Streamlit Clinical Screening Dashboard
#
# Author: Olalemi Olaoluwakintan Emmanuel
# Project: Diabetes Risk Prediction Using Machine Learning
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
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_risk_prediction_model.pkl"
DATA_PATH = BASE_DIR / "diabetes_nan.csv"


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
# EXPECTED FEATURES
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f7f9fc;
    }

    .main {
        padding-top: 1rem;
    }

    /* ========================================================
       HEADER
       ======================================================== */

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a5f 55%,
            #2563eb 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .info-card h3 {
        margin-top: 0;
        color: #111827;
    }

    .metric-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.1rem;
        text-align: center;
        min-height: 120px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 800;
        color: #2563eb;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* ========================================================
       RESULT CARDS
       ======================================================== */

    .risk-high {
        padding: 1.4rem;
        border-radius: 16px;
        background: #fff1f2;
        border-left: 6px solid #dc2626;
        margin: 1rem 0;
    }

    .risk-low {
        padding: 1.4rem;
        border-radius: 16px;
        background: #f0fdf4;
        border-left: 6px solid #16a34a;
        margin: 1rem 0;
    }

    .risk-medium {
        padding: 1.4rem;
        border-radius: 16px;
        background: #fffbeb;
        border-left: 6px solid #d97706;
        margin: 1rem 0;
    }

    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: #fff8e1;
        border-left: 5px solid #f59e0b;
        color: #713f12;
        margin: 1rem 0;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0;
        font-size: 0.85rem;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None, (
            f"Model file not found: {MODEL_PATH.name}. "
            f"Make sure it is in the same folder as app.py."
        )

    try:

        loaded_model = joblib.load(MODEL_PATH)

        return loaded_model, None

    except Exception as error:

        return None, str(error)


model, model_error = load_model()


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():

        return None, (
            f"Dataset file not found: {DATA_PATH.name}."
        )

    try:

        loaded_data = pd.read_csv(DATA_PATH)

        return loaded_data, None

    except Exception as error:

        return None, str(error)


data, data_error = load_dataset()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")

    st.caption(
        "Machine Learning Screening Prototype"
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Executive Dashboard",
            "🔬 Patient Risk Assessment",
            "📊 Dataset Explorer",
            "🧠 Model Intelligence",
            "⚖️ Model Evaluation",
            "🛡️ Responsible AI"
        ]
    )

    st.divider()

    st.markdown("### Model")

    st.write(
        "**Algorithm:** Logistic Regression"
    )

    st.write(
        "**Pipeline:**\n"
        "StandardScaler → SMOTE → Logistic Regression"
    )

    st.divider()

    model_status = (
        "🟢 Loaded"
        if model is not None
        else "🔴 Not Loaded"
    )

    st.write(
        f"**Model Status:** {model_status}"
    )

    st.divider()

    st.caption(
        "Project by\n"
        "**Olalemi Olaoluwakintan Emmanuel**"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🩺 Diabetes Risk Intelligence</h1>

        <p>
        Machine-learning powered diabetes risk screening dashboard
        built from an end-to-end predictive analytics pipeline.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RESPONSIBLE USE NOTICE
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <strong>⚠️ Clinical Safety Notice</strong><br>

    This application is an educational and research prototype.
    It estimates model-based diabetes risk and does not provide
    a medical diagnosis. Results should not replace laboratory
    testing, physician assessment, or established clinical
    screening protocols.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL ERROR NOTICE
# ============================================================

if model_error is not None:

    with st.expander("⚠️ Model loading information"):

        st.error(
            f"Unable to load the model: {model_error}"
        )

        st.code(
            str(MODEL_PATH)
        )


# ============================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.subheader("Executive Dashboard")

    st.write(
        "A high-level view of the dataset, model performance, "
        "and the main analytical findings."
    )

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    total_patients = 768
    diabetic = 268
    non_diabetic = 500

    # If the dataset is available, use actual values
    if data is not None:

        target_column = None

        possible_target_columns = [
            "outcome(target)",
            "Outcome",
            "Outcome(Target)",
            "outcome",
            "target",
            "Target"
        ]

        for column in possible_target_columns:

            if column in data.columns:

                target_column = column
                break

        if target_column is not None:

            total_patients = len(data)

            diabetic = int(
                (data[target_column] == 1).sum()
            )

            non_diabetic = int(
                (data[target_column] == 0).sum()
            )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Patient Records",
            f"{total_patients:,}"
        )

    with c2:

        diabetic_percentage = (
            diabetic / total_patients * 100
            if total_patients > 0
            else 0
        )

        st.metric(
            "Diabetic",
            f"{diabetic:,}",
            f"{diabetic_percentage:.2f}%"
        )

    with c3:

        non_diabetic_percentage = (
            non_diabetic / total_patients * 100
            if total_patients > 0
            else 0
        )

        st.metric(
            "Non-Diabetic",
            f"{non_diabetic:,}",
            f"{non_diabetic_percentage:.2f}%"
        )

    with c4:

        st.metric(
            "Optimized Recall",
            "75.93%",
            "+25.93 pp"
        )

    with c5:

        st.metric(
            "ROC-AUC",
            "78.69%"
        )

    st.divider()

    # --------------------------------------------------------
    # TWO COLUMN DASHBOARD
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="info-card">

            <h3>🎯 Primary Analytical Insight</h3>

            <p>
            The most important improvement was not overall accuracy.
            It was <strong>Recall</strong>.
            </p>

            <p>
            Recall increased from <strong>50.00%</strong> in the
            baseline model to <strong>75.93%</strong> after optimization.
            </p>

            <p>
            For a screening-oriented use case, this means the optimized
            model identifies a substantially larger proportion of the
            diabetic cases present in the test set.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="info-card">

            <h3>🔬 Optimization Pipeline</h3>

            <p><strong>1.</strong> Feature Scaling</p>

            <p>
            StandardScaler brings features onto comparable
            numerical scales.
            </p>

            <p><strong>2.</strong> Class Balancing</p>

            <p>
            SMOTE generates synthetic minority-class training examples.
            </p>

            <p><strong>3.</strong> Hyperparameter Optimization</p>

            <p>
            GridSearchCV evaluates multiple Logistic Regression
            configurations using F1 scoring.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # BASELINE VS OPTIMIZED
    # --------------------------------------------------------

    st.subheader("Baseline → Optimized")

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
        width="stretch",
        hide_index=True
    )

    st.info(
        "The optimized model traded some accuracy and precision "
        "for a substantial improvement in recall. This reflects "
        "the project's screening-oriented objective."
    )

    # --------------------------------------------------------
    # FEATURE INSIGHT
    # --------------------------------------------------------

    st.subheader("📌 Feature Insight")

    feature_values = pd.DataFrame({
        "Feature": [
            "Glucose",
            "BMI",
            "Age",
            "Pregnancies",
            "SkinThickness",
            "Insulin",
            "BloodPressure",
            "DiabetesPedigreeFunction"
        ],
        "Model Coefficient": [
            0.156972,
            0.102375,
            0.058537,
            0.053139,
            0.062338,
            0.058959,
            0.045498,
            0.043508
        ]
    })

    feature_values = feature_values.sort_values(
        "Model Coefficient",
        ascending=True
    )

    # Native Streamlit chart
    chart_data = feature_values.set_index(
        "Feature"
    )

    st.bar_chart(
        chart_data,
        y="Model Coefficient",
        horizontal=True,
        height=450
    )

    st.caption(
        "Higher positive coefficients indicate a stronger positive "
        "association with the model's predicted positive class. "
        "These coefficients should not be interpreted as causal effects."
    )


# ============================================================
# PAGE 2 — PATIENT RISK ASSESSMENT
# ============================================================

elif page == "🔬 Patient Risk Assessment":

    st.subheader("🔬 Patient Risk Assessment")

    st.write(
        "Enter the eight variables used by the trained model."
    )

    if model is None:

        st.error(
            "The trained model could not be loaded."
        )

        st.info(
            "Place diabetes_risk_prediction_model.pkl "
            "in the same directory as app.py."
        )

        if model_error:

            st.exception(
                Exception(model_error)
            )

        st.stop()

    with st.form("patient_form"):

        st.markdown("### Patient Characteristics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

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

        with col2:

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

        with col3:

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

        with col4:

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
            width="stretch",
            type="primary"
        )

    if submitted:

        # ----------------------------------------------------
        # CREATE PATIENT DATAFRAME
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            prediction = int(
                model.predict(patient)[0]
            )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    patient
                )

                probability = float(
                    probabilities[0][1]
                )

            elif hasattr(model, "decision_function"):

                decision = float(
                    model.decision_function(patient)[0]
                )

                probability = float(
                    1 / (1 + np.exp(-decision))
                )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            st.subheader("Assessment Result")

            if probability is not None:

                probability = min(
                    max(probability, 0.0),
                    1.0
                )

                probability_percent = (
                    probability * 100
                )

                # Risk band
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
                        "Predicted Class",
                        (
                            "Positive (1)"
                            if prediction == 1
                            else "Negative (0)"
                        )
                    )

                with r3:

                    st.metric(
                        "Model Risk Band",
                        band
                    )

                # ------------------------------------------------
                # PROBABILITY BAR
                # ------------------------------------------------

                st.markdown(
                    "### Model-Estimated Probability"
                )

                st.progress(
                    probability
                )

                st.caption(
                    f"Estimated probability of positive class: "
                    f"{probability_percent:.2f}%"
                )

                # ------------------------------------------------
                # RESULT MESSAGE
                # ------------------------------------------------

                if prediction == 1:

                    st.markdown(
                        f"""
                        <div class="risk-high">

                        <h3>⚠️ Positive Model Classification</h3>

                        <p>
                        The model classified this patient as
                        <strong>Class 1</strong>.
                        </p>

                        <p>
                        Model-estimated probability:
                        <strong>{probability_percent:.1f}%</strong>
                        </p>

                        <p>
                        This result should be treated as a
                        <strong>screening flag</strong>, not a diagnosis.
                        Appropriate clinical evaluation should determine
                        the next step.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="risk-low">

                        <h3>✓ Negative Model Classification</h3>

                        <p>
                        The model classified this patient as
                        <strong>Class 0</strong>.
                        </p>

                        <p>
                        Model-estimated probability:
                        <strong>{probability_percent:.1f}%</strong>
                        </p>

                        <p>
                        A negative model result does not rule out diabetes.
                        Clinical assessment should remain the deciding factor.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------
                # INPUT SUMMARY
                # ------------------------------------------------

                st.subheader(
                    "Patient Input Summary"
                )

                summary = patient.T.rename(
                    columns={0: "Patient Value"}
                )

                st.dataframe(
                    summary,
                    width="stretch"
                )

                # ------------------------------------------------
                # REPORT
                # ------------------------------------------------

                report = f"""
DIABETES RISK INTELLIGENCE
Machine Learning Screening Report
============================================================

Assessment Date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MODEL RESULT
------------------------------------------------------------

Predicted Class:
{"Positive (1)" if prediction == 1 else "Negative (0)"}

Model Probability:
{probability_percent:.2f}%

Model Risk Band:
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


IMPORTANT NOTICE
------------------------------------------------------------

This output is generated by a machine-learning research prototype.

It is NOT a medical diagnosis.

It must not replace professional clinical assessment,
laboratory testing, or established medical screening protocols.

The model probability is a machine-learning estimate and should
not be interpreted as a clinically calibrated probability unless
the model has undergone appropriate probability calibration
and external clinical validation.
"""

                st.download_button(
                    label="📄 Download Assessment Report",
                    data=report,
                    file_name=(
                        "diabetes_risk_assessment.txt"
                    ),
                    mime="text/plain",
                    width="stretch"
                )

            else:

                st.warning(
                    "The model generated a classification, but "
                    "a probability estimate is not available."
                )

                st.metric(
                    "Predicted Class",
                    (
                        "Positive (1)"
                        if prediction == 1
                        else "Negative (0)"
                    )
                )

        except Exception as error:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(error)

            st.warning(
                """
                This usually indicates that the deployed model's
                expected input format does not match the eight
                features supplied by the application.

                Verify that the saved .pkl model was trained using:

                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age
                """
            )


# ============================================================
# PAGE 3 — DATASET EXPLORER
# ============================================================

elif page == "📊 Dataset Explorer":

    st.subheader("📊 Dataset Explorer")

    if data is None:

        st.warning(
            "diabetes_nan.csv was not found beside the application."
        )

        if data_error:

            st.exception(
                Exception(data_error)
            )

    else:

        st.write(
            "Explore the dataset used during development "
            "of the model."
        )

        # ----------------------------------------------------
        # IDENTIFY TARGET COLUMN
        # ----------------------------------------------------

        target_column = None

        possible_target_columns = [
            "outcome(target)",
            "Outcome",
            "Outcome(Target)",
            "outcome",
            "target",
            "Target"
        ]

        for column in possible_target_columns:

            if column in data.columns:

                target_column = column
                break

        # ----------------------------------------------------
        # DATASET METRICS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Records",
                f"{len(data):,}"
            )

        with c2:

            st.metric(
                "Features",
                "8"
            )

        if target_column is not None:

            positive_cases = int(
                (data[target_column] == 1).sum()
            )

            negative_cases = int(
                (data[target_column] == 0).sum()
            )

        else:

            positive_cases = 0
            negative_cases = 0

        with c3:

            st.metric(
                "Positive Cases",
                f"{positive_cases:,}"
            )

        with c4:

            st.metric(
                "Negative Cases",
                f"{negative_cases:,}"
            )

        st.divider()

        # ----------------------------------------------------
        # TARGET DISTRIBUTION
        # ----------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.markdown(
                "### Target Distribution"
            )

            if target_column is not None:

                target_counts = (
                    data[target_column]
                    .value_counts()
                    .sort_index()
                )

                target_display = pd.DataFrame({
                    "Class": [
                        "No Diabetes",
                        "Diabetes"
                    ],
                    "Patients": [
                        target_counts.get(0, 0),
                        target_counts.get(1, 0)
                    ]
                })

                st.bar_chart(
                    target_display.set_index(
                        "Class"
                    )
                )

            else:

                st.warning(
                    "Target column could not be identified."
                )

        with right:

            st.markdown(
                "### Missing Values — Original Dataset"
            )

            missing = data.isnull().sum()

            missing = (
                missing[
                    missing > 0
                ]
                .sort_values(
                    ascending=False
                )
            )

            if len(missing) > 0:

                st.bar_chart(
                    missing
                )

            else:

                st.success(
                    "No missing values detected."
                )

        st.divider()

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        st.markdown(
            "### 🔍 Data Quality Overview"
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
                "Columns",
                f"{len(data.columns):,}"
            )

        st.divider()

        # ----------------------------------------------------
        # DATA TABLE
        # ----------------------------------------------------

        st.markdown(
            "### Dataset Preview"
        )

        st.dataframe(
            data.head(100),
            width="stretch",
            hide_index=True
        )


# ============================================================
# PAGE 4 — MODEL INTELLIGENCE
# ============================================================

elif page == "🧠 Model Intelligence":

    st.subheader("🧠 Model Intelligence")

    st.write(
        "Understand how the model was constructed and which "
        "features contributed most strongly to its predictions."
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown(
        "### 🔗 Machine Learning Pipeline"
    )

    pipeline_steps = pd.DataFrame({
        "Stage": [
            "Input Data",
            "Feature Scaling",
            "Class Balancing",
            "Model",
            "Hyperparameter Search",
            "Prediction"
        ],
        "Method": [
            "8 patient features",
            "StandardScaler",
            "SMOTE",
            "Logistic Regression",
            "GridSearchCV",
            "Class + Probability"
        ],
        "Purpose": [
            "Capture patient characteristics",
            "Normalize feature magnitudes",
            "Address minority-class representation",
            "Learn classification boundary",
            "Select strong hyperparameters",
            "Generate screening estimate"
        ]
    })

    st.dataframe(
        pipeline_steps,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # BEST PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Selected Hyperparameters"
    )

    params = pd.DataFrame({
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
    })

    st.dataframe(
        params,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE COEFFICIENTS
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Learned Feature Coefficients"
    )

    feature_values = pd.DataFrame({
        "Feature": [
            "Glucose",
            "BMI",
            "SkinThickness",
            "Insulin",
            "Age",
            "Pregnancies",
            "BloodPressure",
            "DiabetesPedigreeFunction"
        ],
        "Coefficient": [
            0.156972,
            0.102375,
            0.062338,
            0.058959,
            0.058537,
            0.053139,
            0.045498,
            0.043508
        ]
    })

    feature_values["Absolute Importance"] = (
        feature_values["Coefficient"].abs()
    )

    feature_values = feature_values.sort_values(
        "Absolute Importance",
        ascending=False
    )

    st.dataframe(
        feature_values,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        "### Feature Importance"
    )

    importance_chart = (
        feature_values[
            ["Feature", "Absolute Importance"]
        ]
        .set_index("Feature")
        .sort_values(
            "Absolute Importance"
        )
    )

    st.bar_chart(
        importance_chart,
        horizontal=True,
        height=450
    )

    st.info(
        "Glucose has the largest positive coefficient in the "
        "trained Logistic Regression model, followed by BMI. "
        "These values describe the model's learned associations; "
        "they do not establish causation."
    )


# ============================================================
# PAGE 5 — MODEL EVALUATION
# ============================================================

elif page == "⚖️ Model Evaluation":

    st.subheader("⚖️ Model Evaluation")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.markdown(
        "### Optimized Model Performance"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        ("Accuracy", "68.83%"),
        ("Precision", "53.95%"),
        ("Recall", "75.93%"),
        ("F1 Score", "63.08%"),
        ("ROC-AUC", "78.69%")
    ]

    for column, (label, value) in zip(
        [c1, c2, c3, c4, c5],
        metrics
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-box">

                    <div class="metric-number">
                    {value}
                    </div>

                    <div class="metric-label">
                    {label}
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

    cm = np.array([
        [
            CONFUSION_MATRIX["TN"],
            CONFUSION_MATRIX["FP"]
        ],
        [
            CONFUSION_MATRIX["FN"],
            CONFUSION_MATRIX["TP"]
        ]
    ])

    cm_df = pd.DataFrame(
        cm,
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
        cm_df,
        width="stretch"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX DETAILS
    # --------------------------------------------------------

    confusion_table = pd.DataFrame({
        "Outcome": [
            "True Negative",
            "False Positive",
            "False Negative",
            "True Positive"
        ],
        "Count": [
            CONFUSION_MATRIX["TN"],
            CONFUSION_MATRIX["FP"],
            CONFUSION_MATRIX["FN"],
            CONFUSION_MATRIX["TP"]
        ],
        "Interpretation": [
            "Correctly identified non-diabetic cases",
            "Non-diabetic cases flagged positive",
            "Diabetic cases missed",
            "Correctly identified diabetic cases"
        ]
    })

    st.dataframe(
        confusion_table,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # CALCULATED METRICS
    # --------------------------------------------------------

    tn = CONFUSION_MATRIX["TN"]
    fp = CONFUSION_MATRIX["FP"]
    fn = CONFUSION_MATRIX["FN"]
    tp = CONFUSION_MATRIX["TP"]

    total = tn + fp + fn + tp

    calculated_accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )

    calculated_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    calculated_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    calculated_f1 = (
        2 * calculated_precision * calculated_recall
        / (calculated_precision + calculated_recall)
        if (calculated_precision + calculated_recall) > 0
        else 0
    )

    st.markdown(
        "### 📐 Metrics Reconstructed From Confusion Matrix"
    )

    calculated_metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Calculated": [
            calculated_accuracy,
            calculated_precision,
            calculated_recall,
            calculated_f1
        ]
    })

    st.dataframe(
        calculated_metrics.style.format({
            "Calculated": "{:.2%}"
        }),
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # MEDICAL CONTEXT
    # --------------------------------------------------------

    st.markdown(
        "### 🏥 Medical Context"
    )

    st.warning(
        """
        In a screening-oriented setting, false negatives can be more
        consequential than false positives because a missed high-risk
        patient may not receive timely follow-up.

        The optimized model has 13 false negatives in the reported
        test result, alongside 35 false positives.

        This reflects a deliberate trade-off toward higher recall,
        not proof of clinical safety.
        """
    )


# ============================================================
# PAGE 6 — RESPONSIBLE AI
# ============================================================

elif page == "🛡️ Responsible AI":

    st.subheader(
        "🛡️ Responsible AI & Clinical Safety"
    )

    st.markdown(
        """
        ### What this model can demonstrate

        - End-to-end machine-learning workflow design
        - Data cleaning and preprocessing
        - Exploratory data analysis
        - Feature scaling
        - Synthetic minority oversampling
        - Hyperparameter optimization
        - Classification evaluation
        - Model serialization
        - Interactive ML deployment

        ### What this model cannot claim

        This project does **not** establish that the model is clinically
        validated or suitable for autonomous diagnosis.

        Before any real-world clinical deployment, the model would require:

        1. External validation on independent populations.
        2. Evaluation across relevant demographic groups.
        3. Probability calibration.
        4. Clinically justified decision thresholds.
        5. Prospective evaluation.
        6. Clinical expert review.
        7. Privacy and security controls.
        8. Appropriate regulatory and governance review.

        ### Why recall was emphasized

        The baseline model achieved approximately **50% recall**.

        The optimized model achieved approximately **75.93% recall**.

        In this project, improving recall was prioritized because the
        analytical objective was screening-oriented.

        However, higher recall alone does not make a model clinically safe.
        Precision, calibration, specificity, external validity, fairness,
        workflow integration, and clinical consequences must also be evaluated.
        """
    )

    st.divider()

    st.markdown(
        "### 🔎 Model Limitations"
    )

    limitations = pd.DataFrame({
        "Area": [
            "Clinical Validation",
            "External Validation",
            "Calibration",
            "Population Generalization",
            "Decision Threshold",
            "Data Quality",
            "Clinical Deployment"
        ],
        "Current Status": [
            "Not performed",
            "Not performed",
            "Not established",
            "Limited to development dataset",
            "Research threshold",
            "Dataset dependent",
            "Not intended"
        ]
    })

    st.dataframe(
        limitations,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        "### Researcher's Note"
    )

    st.info(
        """
        The purpose of this application is to demonstrate how a machine
        learning model can move from experimentation in a notebook to an
        interactive analytical interface.

        It should be viewed as a portfolio and research prototype rather
        than a clinical decision-making system.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <strong>Diabetes Risk Intelligence</strong><br>

    Diabetes Risk Prediction Using Machine Learning<br>

    Olalemi Olaoluwakintan Emmanuel

    </div>
    """,
    unsafe_allow_html=True
        )
