from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Risk Screener",
    layout="wide",
)

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "heart_disease_model.pkl")
    columns = joblib.load(BASE_DIR / "model_columns.pkl")
    return model, columns


model, model_columns = load_model()

st.title("Heart Disease Risk Screener")
st.caption(
    "**Disclaimer:** This tool is for educational and research purposes only. "
    "It does not constitute medical advice. Consult a qualified healthcare professional for diagnosis and treatment."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)

with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

if st.button("Analyze Risk", type="primary", use_container_width=True):
    input_data = pd.DataFrame(0, index=[0], columns=model_columns).astype(float)

    input_data["age"] = float(age)
    input_data["chol"] = float(chol)
    input_data["thalch"] = float(thalch)
    input_data["trestbps"] = float(trestbps)
    input_data["oldpeak"] = float(oldpeak)

    if sex == "Male" and "sex_Male" in input_data.columns:
        input_data["sex_Male"] = 1.0
    if exang == "Yes" and "exang_True" in input_data.columns:
        input_data["exang_True"] = 1.0

    cp_map = {
        "Atypical Angina": "cp_atypical angina",
        "Non-anginal Pain": "cp_non-anginal pain",
        "Asymptomatic": "cp_asymptomatic",
    }
    cp_col = cp_map.get(cp)
    if cp_col and cp_col in input_data.columns:
        input_data[cp_col] = 1.0

    prediction = model.predict(input_data)[0]
    probability = float(model.predict_proba(input_data)[0][1])

    st.divider()
    result_col, chart_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.error(f"**HIGH RISK** — {probability:.1%} probability of heart disease")
        else:
            st.success(f"**LOW RISK** — {probability:.1%} probability of heart disease")

        st.metric("Risk Probability", f"{probability:.1%}")
        st.progress(probability)
        st.caption("Score reflects model probability; not a clinical diagnosis.")

    with chart_col:
        importances = (
            pd.Series(model.feature_importances_, index=model_columns)
            .nlargest(8)
            .sort_values()
        )
        st.write("**Top 8 Predictive Features**")
        st.bar_chart(importances)
