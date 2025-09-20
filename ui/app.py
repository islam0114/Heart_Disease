import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

model_path = os.path.join("models", "final_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

model_path = os.path.join("models", "scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("❤️ Heart Disease Prediction (Encoded Features)")
st.write("Enter your health data to predict the likelihood of heart disease:")

age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No","Yes"])
restecg = st.selectbox("Resting ECG", [0,1,2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina?", ["Yes","No"])
oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping","flat","downsloping"])
ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", [0,1,2,3])
thal = st.selectbox("Thalassemia", ["normal","fixed defect","reversable defect"])


columns = ['thal_reversable defect', 'oldpeak', 'thal_normal', 'cp_asymptomatic',
           'chol', 'slope_upsloping', 'trestbps', 'cp_non-anginal pain',
           'exang_no', 'sex_female', 'thalach', 'age', 'ca']

input_df = pd.DataFrame(np.zeros((1,len(columns))), columns=columns)

input_df['age'] = age
input_df['trestbps'] = trestbps
input_df['chol'] = chol
input_df['thalach'] = thalach
input_df['oldpeak'] = oldpeak
input_df['ca'] = ca

input_df['sex_female'] = 1 if sex=="Female" else 0
input_df['cp_asymptomatic'] = 1 if cp=="asymptomatic" else 0
input_df['cp_non-anginal pain'] = 1 if cp=="non-anginal pain" else 0
input_df['slope_upsloping'] = 1 if slope=="upsloping" else 0
input_df['exang_no'] = 1 if exang=="No" else 0
input_df['thal_normal'] = 1 if thal=="normal" else 0
input_df['thal_reversable defect'] = 1 if thal=="reversable defect" else 0
fbs_value = 1 if fbs=="Yes" else 0

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred_proba = model.predict_proba(input_scaled)[0][1]
    pred_class = model.predict(input_scaled)[0]
    st.write(f"### Predicted Probability of Heart Disease: {pred_proba*100:.2f}%")
    st.write(f"### Predicted Class: {'Has Heart Disease' if pred_class==1 else 'No Heart Disease'}")
