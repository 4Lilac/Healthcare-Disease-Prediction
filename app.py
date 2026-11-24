
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Healthcare Disease Prediction",
    page_icon="🏥",
    layout="wide",
)


# Predictor columns used to train the saved models.
CLASSIFICATION_FEATURES = [
    "age_level",
    "gender",
    "smoking",
    "age",
    "age_normalized",
    "bmi",
    "HbA1c_level",
    "glucose",
    "cholesterol",
    "sleep_hours",
    "triglycerides",
    "physical_activity",
    "family_history",
    "stress_level",
    "low_hdl_cholesterol",
    "high_ldl_cholesterol",
    "blood_pressure",
    "high_blood_pressure",
    "sugar_consumption",
    "crp_level",
    "homocysteine_level",
    "systolic_bp",
    "diastolic_bp",
    "alcohol_intake",
    "salt_intake",
    "heart_rate",
    "hdl",
    "ldl",
    "education_level",
    "employment_status",
    "pulse_pressure",
    "cholesterol_ratio",
]

REGRESSION_FEATURES = [
    feature for feature in CLASSIFICATION_FEATURES if feature != "bmi"
]


# Load trained models from the ZIP file.
@st.cache_resource(show_spinner="Loading trained models...")
def load_models():

    zip_path = (
        Path(__file__).resolve().parent
        / "models"
        / "healthcare_models.zip"
    )

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Model ZIP file was not found at: {zip_path}"
        )

    model_dir = Path(tempfile.gettempdir()) / "healthcare_prediction_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    required_files = (
        "classification_model.pkl",
        "regression_model.pkl",
    )

    with zipfile.ZipFile(zip_path, "r") as archive:

        archive_names = set(archive.namelist())

        missing_files = [
            name for name in required_files if name not in archive_names
        ]

        if missing_files:
            raise FileNotFoundError(
                f"The model ZIP is missing these files: {missing_files}"
            )

        for filename in required_files:
            destination = model_dir / filename
            destination.write_bytes(archive.read(filename))

    classifier = joblib.load(
        model_dir / "classification_model.pkl"
    )

    regressor = joblib.load(
        model_dir / "regression_model.pkl"
    )

    # Confirm that the model columns match the app inputs.
    classifier_columns = list(classifier.feature_names_in_)
    regressor_columns = list(regressor.feature_names_in_)

    if set(classifier_columns) != set(CLASSIFICATION_FEATURES):
        raise ValueError(
            "Classification model features do not match the app. "
            f"Model columns: {classifier_columns}"
        )

    if set(regressor_columns) != set(REGRESSION_FEATURES):
        raise ValueError(
            "Regression model features do not match the app. "
            f"Model columns: {regressor_columns}"
        )

    return classifier, regressor


try:
    classifier, regressor = load_models()

except Exception as exc:
    st.error("The trained models could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Check that models/healthcare_models.zip is in the GitHub "
        "repository and that the required packages are installed."
    )
    st.stop()


# Age groups used in the dataset.
def age_group(age):

    if age <= 23:
        return "Young"

    if age <= 45:
        return "Middle"

    if age <= 67:
        return "Mature"

    return "Senior"


# Application heading.
st.title("🏥 Healthcare Disease Prediction")

st.write(
    "Enter patient information to generate a health "
    "classification or estimate BMI."
)

st.caption(
    "For educational and analytical purposes only. "
    "This application does not provide medical diagnoses."
)

st.divider()


# Choose prediction type.
prediction_type = st.radio(
    "Choose prediction type",
    ["Health classification", "BMI estimation"],
    horizontal=True,
)


# Categorical inputs.
categorical_fields = [

    ("gender", "Gender", ["Female", "Male"], "Female"),

    ("smoking", "Smoking Status",
     ["Never", "Former", "Current"], "Never"),

    ("physical_activity", "Physical Activity",
     ["High", "Moderate", "Low"], "Moderate"),

    ("family_history", "Family History",
     ["No", "Yes"], "No"),

    ("stress_level", "Stress Level",
     ["High", "Medium", "Low"], "Medium"),

    ("low_hdl_cholesterol", "Low HDL Cholesterol",
     ["No", "Yes"], "No"),

    ("high_ldl_cholesterol", "High LDL Cholesterol",
     ["No", "Yes"], "No"),

    ("high_blood_pressure", "High Blood Pressure",
     ["No", "Yes"], "No"),

    ("sugar_consumption", "Sugar Consumption",
     ["High", "Medium", "Low"], "Medium"),

    ("education_level", "Education Level",
     ["Primary", "Secondary", "Tertiary"], "Tertiary"),

    ("employment_status", "Employment Status",
     ["Employed", "Retired", "Unemployed"], "Employed"),
]


# Numerical inputs: field, label, minimum, maximum, default, step.
numeric_fields = [

    ("age", "Age", 2, 89, 50, 1),

    ("HbA1c_level", "HbA1c Level",
     3.5, 9.0, 5.8, 0.1),

    ("glucose", "Glucose",
     70.0, 300.0, 138.0, 1.0),

    ("cholesterol", "Cholesterol",
     150.0, 300.0, 225.0, 1.0),

    ("sleep_hours", "Sleep Hours",
     4.0, 10.0, 6.98, 0.1),

    ("triglycerides", "Triglycerides",
     50.0, 400.0, 179.0, 1.0),

    ("blood_pressure", "Blood Pressure",
     120.0, 180.0, 149.7, 0.1),

    ("crp_level", "CRP Level",
     0.003647, 14.997087, 7.46, 0.01),

    ("homocysteine_level", "Homocysteine Level",
     5.000236, 19.999037, 12.49, 0.01),

    ("systolic_bp", "Systolic Blood Pressure",
     90.0, 179.0, 134.2, 0.1),

    ("diastolic_bp", "Diastolic Blood Pressure",
     60.0, 119.0, 89.5, 0.1),

    ("alcohol_intake", "Alcohol Intake",
     0.0, 30.0, 15.0, 0.1),

    ("salt_intake", "Salt Intake",
     2.0, 15.0, 8.5, 0.1),

    ("heart_rate", "Heart Rate",
     50.0, 99.0, 74.5, 0.1),

    ("hdl", "HDL Cholesterol",
     30.0, 99.0, 64.7, 0.1),

    ("ldl", "LDL Cholesterol",
     70.0, 189.0, 130.0, 0.1),
]


# Patient input form.
with st.form("patient_inputs"):

    st.subheader("Patient Information")

    st.markdown("**Personal and Lifestyle Information**")

    categorical_values = {}
    categorical_columns = st.columns(3)

    for index, (key, label, options, default) in enumerate(
        categorical_fields
    ):

        with categorical_columns[index % 3]:

            categorical_values[key] = st.selectbox(
                label,
                options,
                index=options.index(default),
            )

    st.markdown("**Health Measurements**")

    numeric_values = {}
    numeric_columns_ui = st.columns(3)

    for index, (
        key, label, minimum, maximum, default, step
    ) in enumerate(numeric_fields):

        with numeric_columns_ui[index % 3]:

            numeric_values[key] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step,
            )

    current_bmi = None

    if prediction_type == "Health classification":

        current_bmi = st.number_input(
            "Current BMI",
            min_value=10.01,
            max_value=95.69,
            value=27.32,
            step=0.1,
        )

    submitted = st.form_submit_button(
        "Generate Prediction",
        type="primary",
        use_container_width=True,
    )


# Generate prediction.
if submitted:

    age = int(numeric_values["age"])
    hdl_value = float(numeric_values["hdl"])

    # Prepare engineered features used during model training.
    input_values = {
        **categorical_values,
        **numeric_values,

        "age_level": age_group(age),

        "age_normalized": (age - 2) / 87,

        "pulse_pressure": (
            numeric_values["systolic_bp"]
            - numeric_values["diastolic_bp"]
        ),

        "cholesterol_ratio": (
            numeric_values["cholesterol"] / hdl_value
            if hdl_value != 0
            else np.nan
        ),
    }

    try:

        if prediction_type == "Health classification":

            input_values["bmi"] = float(current_bmi)

            model_columns = list(classifier.feature_names_in_)

            X = pd.DataFrame(
                [
                    {
                        column: input_values[column]
                        for column in model_columns
                    }
                ],
                columns=model_columns,
            )

            prediction = str(classifier.predict(X)[0])

            probabilities = classifier.predict_proba(X)[0]

            classes = list(
                classifier.named_steps["model"].classes_
            )

            abnormal_index = classes.index("Abnormal")

            abnormal_probability = float(
                probabilities[abnormal_index]
            )

            st.subheader("Health Classification Result")

            if prediction == "Abnormal":
                st.error("Predicted Classification: Abnormal")

            else:
                st.success("Predicted Classification: Normal")

            st.metric(
                "Probability of Abnormal Classification",
                f"{abnormal_probability:.1%}",
            )

        else:

            model_columns = list(regressor.feature_names_in_)

            X = pd.DataFrame(
                [
                    {
                        column: input_values[column]
                        for column in model_columns
                    }
                ],
                columns=model_columns,
            )

            predicted_bmi = float(regressor.predict(X)[0])

            st.subheader("BMI Estimation Result")

            st.metric(
                "Estimated BMI",
                f"{predicted_bmi:.2f}",
            )

    except Exception as exc:

        st.error("The prediction could not be generated.")

        st.code(f"{type(exc).__name__}: {exc}")


st.divider()

st.caption(
    "Disclaimer: Predictions are for educational and analytical "
    "purposes only. They are not medical diagnoses and must not "
    "replace professional medical advice."
)
