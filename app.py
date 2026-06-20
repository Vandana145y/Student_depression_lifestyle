import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

st.set_page_config(
    page_title="Student Depression Lifestyle Predictor",
    page_icon="🎓",
    layout="centered",
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CSV_PATH = "student_lifestyle_100k.csv"
import os

print("CSV_PATH:", CSV_PATH)
print("Exists:", os.path.exists(CSV_PATH))
MODEL_CACHE_PATH = "student_depression_model.joblib"

GENDER_OPTIONS = ["Female", "Male"]
DEPARTMENT_OPTIONS = ["Arts", "Business", "Engineering", "Medical", "Science"]

FEATURE_ORDER = [
    "Age",
    "Gender",
    "Department",
    "CGPA",
    "Sleep_Duration",
    "Study_Hours",
    "Social_Media_Hours",
    "Physical_Activity",
    "Stress_Level",
]


# ─────────────────────────────────────────────
# Data loading + training (cached so it only runs once per session)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    """
    Loads the dataset, trains a RandomForest pipeline on it, and caches
    the fitted model + test metrics + the label encoders used, so the
    same encoding is applied at inference time.
    """
    df = pd.read_csv(CSV_PATH)

    df = df.drop(columns=["Student_ID"])

    encoders = {}
    for col in ["Gender", "Department"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    df["Depression"] = df["Depression"].astype(int)
    df = df.drop_duplicates()

    X = df.drop(columns=["Depression"])
    y = df["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("Scaler", StandardScaler()),
            (
                "Model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=11,
                    min_samples_split=9,
                    min_samples_leaf=17,
                    max_features="sqrt",
                    bootstrap=True,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_auc": roc_auc_score(y_test, y_proba),
    }

    joblib.dump({"pipeline": pipeline, "encoders": encoders, "metrics": metrics}, MODEL_CACHE_PATH)

    return pipeline, encoders, metrics


@st.cache_resource
def get_model():
    # Reuse a previously-trained model on disk if present, otherwise train fresh
    if os.path.exists(MODEL_CACHE_PATH):
        bundle = joblib.load(MODEL_CACHE_PATH)
        return bundle["pipeline"], bundle["encoders"], bundle["metrics"]
    return load_and_train()


def build_input_dataframe(
    age, gender, department, cgpa, sleep_duration,
    study_hours, social_media_hours, physical_activity, stress_level,
    encoders,
):
    row = {
        "Age": age,
        "Gender": encoders["Gender"].transform([gender])[0],
        "Department": encoders["Department"].transform([department])[0],
        "CGPA": cgpa,
        "Sleep_Duration": sleep_duration,
        "Study_Hours": study_hours,
        "Social_Media_Hours": social_media_hours,
        "Physical_Activity": physical_activity,
        "Stress_Level": stress_level,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ─────────────────────────────────────────────
# Load model (trains automatically on first run, no manual setup needed)
# ─────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    st.error(
        f"Could not find `{CSV_PATH}`. Place the dataset CSV in the same "
        "folder as this app.py and refresh the page."
    )
    st.stop()

with st.spinner("Loading model..."):
    model, encoders, metrics = get_model()

st.sidebar.header("Model Info")
st.sidebar.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
st.sidebar.metric("Test F1", f"{metrics['test_f1']:.3f}")
st.sidebar.metric("Test ROC-AUC", f"{metrics['test_auc']:.3f}")
st.sidebar.caption(
    "Model trains automatically from the CSV on first run and is cached "
    "afterward — no run ID needed."
)
if st.sidebar.button("Retrain model"):
    if os.path.exists(MODEL_CACHE_PATH):
        os.remove(MODEL_CACHE_PATH)
    st.cache_resource.clear()
    st.rerun()

# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.title("🎓 Student Depression Predictor")
st.write(
    "Fill in the student's lifestyle details below to predict the likelihood "
    "of depression."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=15, max_value=40, value=20)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        department = st.selectbox("Department", DEPARTMENT_OPTIONS)
        cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
        sleep_duration = st.slider("Sleep Duration (hours/day)", 0.0, 12.0, 7.0, step=0.1)

    with col2:
        study_hours = st.slider("Study Hours (per day)", 0.0, 16.0, 4.0, step=0.1)
        social_media_hours = st.slider("Social Media Hours (per day)", 0.0, 12.0, 2.0, step=0.1)
        physical_activity = st.number_input(
            "Physical Activity (minutes/day)", min_value=0, max_value=300, value=60
        )
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

    submitted = st.form_submit_button("Predict")

if submitted:
    X_input = build_input_dataframe(
        age, gender, department, cgpa, sleep_duration,
        study_hours, social_media_hours, physical_activity, stress_level,
        encoders,
    )

    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ Predicted: **Depression risk detected**")
    else:
        st.success("✅ Predicted: **No depression risk detected**")

    st.metric("Predicted probability of depression", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    with st.expander("See input data sent to the model"):
        st.dataframe(X_input)

st.divider()
