import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Custom CSS for slightly lighter dark gray background and modern look
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #343a40;
        color: #f8f9fa;
    }
    .main {
        background-color: #343a40;
    }
    .stButton>button {
        background-color: #343a40;
        color: #f8f9fa;
        font-weight: bold;
        border-radius: 8px;
        border: 1.5px solid #495057;
        padding: 0.5em 2em;
    }
    .stButton>button:hover {
        background-color: #495057;
        color: #fff;
    }
    /* Number input styling - simple dark gray */
    .stNumberInput>div>input {
        background-color: #343a40;
        color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #495057;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: none;
        transition: border 0.2s, background 0.2s;
    }
    .stNumberInput>div>input:focus {
        border: 1.5px solid #00b4d8;
        background-color: #3a3f44;
        outline: none;
    }
    .stNumberInput>div>input:hover {
        border: 1.5px solid #00b4d8;
        background-color: #3a3f44;
    }
    .stTextInput>div>input {
        background-color: #343a40;
        color: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #495057;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🩺 Diabetes Risk Prediction App")
st.markdown("""
##### Take control of your health! 
Please fill in the required information below to instantly discover your risk of diabetes. All results are analyzed by a professional machine learning model. 

*Your privacy is respected. For medical advice, always consult a healthcare professional.*
""")

# User input fields with helpful explanations
pregnancies = st.number_input(
    "Number of Pregnancies",
    min_value=0, max_value=20, value=1,
    help="Number of times the patient has been pregnant."
)
glucose = st.number_input(
    "Glucose Level",
    min_value=0, max_value=200, value=120,
    help="Plasma glucose concentration after fasting (mg/dL)."
)
blood_pressure = st.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0, max_value=150, value=70,
    help="Diastolic blood pressure (mm Hg)."
)
skin_thickness = st.number_input(
    "Skin Thickness (mm)",
    min_value=0, max_value=100, value=20,
    help="Thickness of the skin fold at the triceps (mm), indicates subcutaneous fat."
)
insulin = st.number_input(
    "Insulin (mu U/ml)",
    min_value=0, max_value=900, value=80,
    help="Fasting serum insulin level (mu U/ml)."
)
bmi = st.number_input(
    "BMI (Body Mass Index)",
    min_value=0.0, max_value=70.0, value=25.0,
    help="Body Mass Index: weight (kg) / [height (m)]²."
)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0, max_value=3.0, value=0.5,
    help="A score indicating the likelihood of diabetes based on family history."
)
age = st.number_input(
    "Age",
    min_value=1, max_value=120, value=33,
    help="Age of the patient in years."
)

# Load model and scaler
@st.cache_data
def load_data_and_model():
    data = pd.read_csv("C:/Users/kubra/OneDrive/Masaüstü/diabetes.csv")
    X = data.drop(columns="Outcome", axis=1)
    y = data["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GaussianNB()
    model.fit(X_scaled, y)
    return scaler, model

scaler, model = load_data_and_model()

# Prediction button
if st.button("Predict My Risk"): 
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"**High risk:** You are at high risk of having diabetes! (Risk: %{round(probability*100,2)})")
    else:
        st.success(f"**Low risk:** You are at low risk of having diabetes. (Risk: %{round(probability*100,2)})")
    st.markdown("---")
    st.markdown("**Note:** These results are based solely on a machine learning model. For a definitive diagnosis, please consult a medical professional.") 