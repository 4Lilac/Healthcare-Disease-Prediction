
import importlib.util
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import pandas as pd

# Install joblib if missing
try:
    if importlib.util.find_spec("joblib") is None:
        subprocess.check_call([
            sys.executable, "-m", "pip",
            "install", "joblib==1.6.0"
        ])
    import joblib
except Exception as exc:
    st.error("Could not install or import joblib.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()


st.set_page_config(
    page_title="Healthcare Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

CLASSIFICATION_FEATURES = [
    "age_level", "gender", "smoking", "age",
    "age_normalized", "bmi", "HbA1c_level",
    "glucose", "cholesterol", "sleep_hours",
    "triglycerides", "physical_activity",
    "family_history", "stress_level",
    "low_hdl_cholesterol", "high_ldl_cholesterol",
    "blood_pressure", "high_blood_pressure",
    "sugar_consumption", "crp_level",
    "homocysteine_level", "systolic_bp",
    "diastolic_bp", "alcohol_intake", "salt_intake",
    "heart_rate", "hdl", "ldl", "education_level",
    "employment_status"
]

REGRESSION_FEATURES = [
    c for c in CLASSIFICATION_FEATURES if c != "bmi"
]


# Load trained models
@st.cache_resource(show_spinner="Loading trained models...")
def load_models():

    zip_path = (
        Path(__file__).resolve().parent
        / "models"
        / "healthcare_models.zip"
    )

    if not zip_path.exists():
        raise FileNotFoundError("Model ZIP file not found.")

    model_dir = (
        Path(tempfile.gettempdir())
        / "healthcare_prediction_models"
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "classification_model.pkl",
        "regression_model.pkl"
    ]

    with zipfile.ZipFile(zip_path, "r") as archive:

        archive_names = set(archive.namelist())

        missing = [
            name for name in required_files
            if name not in archive_names
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}"
            )

        for name in required_files:
            target = model_dir / name
            target.write_bytes(archive.read(name))

    classifier = joblib.load(
        model_dir / "classification_model.pkl"
    )

    regressor = joblib.load(
        model_dir / "regression_model.pkl"
    )

    return classifier, regressor


try:
    classifier, regressor = load_models()

except Exception as exc:
    st.error("The trained models could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info("Please check the Streamlit application logs.")
    st.stop()


# Determine age group from age
def age_group(age):

    if age <= 23:
        return "Young"
    elif age <= 45:
        return "Middle"
    elif age <= 67:
        return "Mature"
    else:
        return "Senior"


# Application heading
st.title("🏥 Healthcare Disease Prediction")

st.write(
    "Enter patient information to predict a health "
    "classification or estimate BMI."
)

st.caption(
    "For educational and analytical purposes only. "
    "This application does not provide medical diagnoses."
)

st.divider()


# Choose prediction
task = st.radio(
    "Choose prediction type",
    ["Health classification", "BMI estimation"],
    horizontal=True
)


# Categorical inputs
categorical_fields = [
    (
        "gender", "Gender",
        ["Female", "Male"], "Female"
    ),
    (
        "smoking", "Smoking Status",
        ["Never", "Former", "Current"], "Never"
    ),
    (
        "physical_activity", "Physical Activity",
        ["High", "Moderate", "Low"], "Moderate"
    ),
    (
        "family_history", "Family History",
        ["No", "Yes"], "No"
    ),
    (
        "stress_level", "Stress Level",
        ["High", "Medium", "Low"], "Medium"
    ),
    (
        "low_hdl_cholesterol", "Low HDL Cholesterol",
        ["No", "Yes"], "No"
    ),
    (
        "high_ldl_cholesterol", "High LDL Cholesterol",
        ["No", "Yes"], "No"
    ),
    (
        "high_blood_pressure", "High Blood Pressure",
        ["No", "Yes"], "No"
    ),
    (
        "sugar_consumption", "Sugar Consumption",
        ["High", "Medium", "Low"], "Medium"
    ),
    (
        "education_level", "Education Level",
        ["Primary", "Secondary", "Tertiary"], "Tertiary"
    ),
    (
        "employment_status", "Employment Status",
        ["Employed", "Retired", "Unemployed"], "Employed"
    )
]


# Numerical inputs: field, label, minimum, maximum, default, step
numeric_fields = [
    ("age", "Age", 2, 89, 50, 1),
    ("HbA1c_level", "HbA1c Level", 3.5, 9.0, 5.8, 0.1),
    ("glucose", "Glucose", 70.0, 300.0, 138.0, 1.0),
    ("cholesterol", "Cholesterol", 150.0, 300.0, 225.0, 1.0),
    ("sleep_hours", "Sleep Hours", 4.0, 10.0, 6.98, 0.1),
    ("triglycerides", "Triglycerides", 50.0, 400.0, 179.0, 1.0),
    ("blood_pressure", "Blood Pressure", 120.0, 180.0, 149.7, 0.1),
    ("crp_level", "CRP Level", 0.003647, 14.997087, 7.46, 0.01),
    ("homocysteine_level", "Homocysteine Level", 5.000236, 19.999037, 12.49, 0.01),
    ("systolic_bp", "Systolic Blood Pressure", 90.0, 179.0, 134.2, 0.1),
    ("diastolic_bp", "Diastolic Blood Pressure", 60.0, 119.0, 89.5, 0.1),
    ("alcohol_intake", "Alcohol Intake", 0.0, 30.0, 15.0, 0.1),
    ("salt_intake", "Salt Intake", 2.0, 15.0, 8.5, 0.1),
    ("heart_rate", "Heart Rate", 50.0, 99.0, 74.5, 0.1),
    ("hdl", "HDL Cholesterol", 30.0, 99.0, 64.7, 0.1),
    ("ldl", "LDL Cholesterol", 70.0, 189.0, 130.0, 0.1)
]


# Patient input form
with st.form("patient_inputs"):

    st.subheader("Patient Information")

    st.markdown("**Personal and Lifestyle Information**")

    categorical_values = {}
    columns = st.columns(3)

    for i, (key, label, options, default) in enumerate(
        categorical_fields
    ):
        with columns[i % 3]:
            categorical_values[key] = st.selectbox(
                label,
                options,
                index=options.index(default)
            )

    st.markdown("**Health Measurements**")

    numeric_values = {}
    columns = st.columns(3)

    for i, (key, label, minimum, maximum, default, step) in enumerate(
        numeric_fields
    ):
        with columns[i % 3]:
            numeric_values[key] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step
            )

    current_bmi = None

    if task == "Health classification":
        current_bmi = st.number_input(
            "Current BMI",
            min_value=10.01,
            max_value=95.69,
            value=27.32,
            step=0.1
        )

    submitted = st.form_submit_button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    )


# Generate prediction
if submitted:

    age = numeric_values["age"]

    input_values = {
        **categorical_values,
        **numeric_values,
        "age_level": age_group(age),
        "age_normalized": (age - 2) / 87
    }

    if task == "Health classification":

        input_values["bmi"] = current_bmi

        X = pd.DataFrame([
            {
                name: input_values[name]
                for name in CLASSIFICATION_FEATURES
            }
        ])

        prediction = classifier.predict(X)[0]

        probabilities = classifier.predict_proba(X)[0]

        classes = list(
            classifier.named_steps["model"].classes_
        )

        abnormal_probability = float(
            probabilities[classes.index("Abnormal")]
        )

        st.subheader("Health Classification Result")

        if prediction == "Abnormal":
            st.error(f"Predicted Condition: {prediction}")
        else:
            st.success(f"Predicted Condition: {prediction}")

        st.metric(
            "Probability of Abnormal Classification",
            f"{abnormal_probability:.1%}"
        )

    else:

        X = pd.DataFrame([
            {
                name: input_values[name]
                for name in REGRESSION_FEATURES
            }
        ])

        predicted_bmi = float(regressor.predict(X)[0])

        st.subheader("BMI Prediction Result")

        st.metric(
            "Estimated BMI",
            f"{predicted_bmi:.2f}"
        )


st.divider()

st.caption(
    "Disclaimer: Predictions are for educational and analytical "
    "purposes only. They are not medical diagnoses and must not "
    "replace professional medical advice."
)
