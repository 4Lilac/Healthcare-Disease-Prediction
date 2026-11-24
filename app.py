import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import shutil
import tempfile

from pathlib import Path


st.set_page_config(
    page_title="Healthcare Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load the trained models
@st.cache_resource
def load_models():

    root = Path(__file__).resolve().parent
    zip_path = root / "models" / "healthcare_models.zip"

    if not zip_path.exists():
        raise FileNotFoundError("Model ZIP file not found.")

    model_dir = Path(tempfile.gettempdir()) / "healthcare_models"
    model_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:

        required_files = [
            "classification_model.pkl",
            "regression_model.pkl"
        ]

        for name in required_files:
            if name not in archive.namelist():
                raise FileNotFoundError(f"{name} is missing from the ZIP.")

            target = model_dir / name

            if not target.exists():
                with archive.open(name) as source:
                    with open(target, "wb") as destination:
                        shutil.copyfileobj(source, destination)

    classifier = joblib.load(
        model_dir / "classification_model.pkl"
    )

    regressor = joblib.load(
        model_dir / "regression_model.pkl"
    )

    return classifier, regressor


try:
    classifier, regressor = load_models()
except Exception as error:
    st.error("Unable to load the prediction models.")
    st.exception(error)
    st.stop()


# Website heading
st.title("🏥 Healthcare Disease Prediction")
st.write(
    "Enter patient health information to obtain "
    "a health classification or estimated BMI."
)

st.divider()


# Categorical inputs
categorical_fields = [
    ("age_level", "Age Group",
     ["Mature", "Middle", "Senior", "Young"], "Mature"),

    ("gender", "Gender",
     ["Female", "Male"], "Female"),

    ("smoking", "Smoking Status",
     ["Never", "Former", "Current"], "Never"),

    ("physical_activity", "Physical Activity",
     ["High", "Moderate", "Low"], "High"),

    ("family_history", "Family History",
     ["No", "Yes"], "No"),

    ("stress_level", "Stress Level",
     ["High", "Medium", "Low"], "High"),

    ("low_hdl_cholesterol", "Low HDL Cholesterol",
     ["No", "Yes"], "No"),

    ("high_ldl_cholesterol", "High LDL Cholesterol",
     ["Yes", "No"], "Yes"),

    ("high_blood_pressure", "High Blood Pressure",
     ["Yes", "No"], "Yes"),

    ("sugar_consumption", "Sugar Consumption",
     ["High", "Medium", "Low"], "Low"),

    ("education_level", "Education Level",
     ["Primary", "Secondary", "Tertiary"], "Tertiary"),

    ("employment_status", "Employment Status",
     ["Employed", "Retired", "Unemployed"], "Unemployed")
]


# Numerical inputs
numeric_fields = [
    ("age", "Age", 2.0, 89.0, 50.0, 1.0),
    ("bmi", "Current BMI (classification only)", 10.01, 95.69, 27.32, 0.1),
    ("HbA1c_level", "HbA1c Level", 3.5, 9.0, 5.8, 0.1),
    ("glucose", "Glucose", 70.0, 300.0, 137.0, 1.0),
    ("cholesterol", "Cholesterol", 150.0, 300.0, 225.0, 1.0),
    ("sleep_hours", "Sleep Hours", 4.0, 10.0, 7.0, 0.1),
    ("triglycerides", "Triglycerides", 50.0, 400.0, 178.0, 1.0),
    ("blood_pressure", "Blood Pressure", 120.0, 180.0, 149.7, 0.1),
    ("crp_level", "CRP Level", 0.003647, 14.997087, 7.46, 0.01),
    ("homocysteine_level", "Homocysteine Level", 5.000236, 19.999037, 12.49, 0.01),
    ("systolic_bp", "Systolic Blood Pressure", 90.0, 179.0, 134.0, 1.0),
    ("diastolic_bp", "Diastolic Blood Pressure", 60.0, 119.0, 89.0, 1.0),
    ("alcohol_intake", "Alcohol Intake", 0.0, 30.0, 15.0, 0.1),
    ("salt_intake", "Salt Intake", 2.0, 15.0, 8.5, 0.1),
    ("heart_rate", "Heart Rate", 50.0, 99.0, 74.0, 1.0),
    ("hdl", "HDL Cholesterol", 30.0, 99.0, 65.0, 1.0),
    ("ldl", "LDL Cholesterol", 70.0, 189.0, 130.0, 1.0)
]


# Input form
with st.form("health_form"):

    st.subheader("Patient Information")

    categorical_values = {}
    numeric_values = {}

    st.markdown("**Personal and Lifestyle Information**")

    columns = st.columns(3)

    for i, (name, label, options, default) in enumerate(categorical_fields):
        with columns[i % 3]:
            categorical_values[name] = st.selectbox(
                label,
                options,
                index=options.index(default)
            )

    st.divider()
    st.markdown("**Health Measurements**")

    columns = st.columns(3)

    for i, (name, label, minimum, maximum, default, step) in enumerate(numeric_fields):

        with columns[i % 3]:
            numeric_values[name] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        predict_class = st.form_submit_button(
            "Predict Health Condition",
            type="primary",
            use_container_width=True
        )

    with col2:
        predict_bmi = st.form_submit_button(
            "Predict BMI",
            use_container_width=True
        )


# Prepare model inputs
if predict_class or predict_bmi:

    input_values = {
        **categorical_values,
        **numeric_values
    }

    # Engineered features
    input_values["age_normalized"] = (
        numeric_values["age"] - 2
    ) / 87

    input_values["pulse_pressure"] = (
        numeric_values["systolic_bp"]
        - numeric_values["diastolic_bp"]
    )

    input_values["cholesterol_ratio"] = (
        numeric_values["cholesterol"]
        / numeric_values["hdl"]
    )

    classification_features = [
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
        "employment_status", "pulse_pressure",
        "cholesterol_ratio"
    ]

    regression_features = [
        column for column in classification_features
        if column != "bmi"
    ]

    # Classification prediction
    if predict_class:

        class_input = pd.DataFrame(
            [{col: input_values[col] for col in classification_features}]
        )

        prediction = classifier.predict(class_input)[0]
        probabilities = classifier.predict_proba(class_input)[0]

        classes = list(classifier.named_steps["model"].classes_)
        abnormal_probability = probabilities[
            classes.index("Abnormal")
        ]

        st.divider()
        st.subheader("Health Classification Result")

        if prediction == "Abnormal":
            st.error(f"Predicted Condition: {prediction}")
        else:
            st.success(f"Predicted Condition: {prediction}")

        st.metric(
            "Probability of Abnormal Classification",
            f"{abnormal_probability:.2%}"
        )

    # BMI prediction
    if predict_bmi:

        regression_input = pd.DataFrame(
            [{col: input_values[col] for col in regression_features}]
        )

        bmi_prediction = regressor.predict(regression_input)[0]

        st.divider()
        st.subheader("BMI Prediction Result")

        st.metric(
            "Predicted BMI",
            f"{bmi_prediction:.2f}"
        )


# Disclaimer
st.divider()

st.caption(
    "Disclaimer: This application is developed for educational "
    "and analytical purposes only. Predictions are not medical "
    "diagnoses and should not replace professional medical advice."
)
