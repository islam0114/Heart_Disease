# Heart Disease Prediction Project

## Overview
This project aims to **predict heart disease** using patient health data. The project covers **data preprocessing, exploratory data analysis (EDA), dimensionality reduction, feature selection, supervised and unsupervised machine learning, hyperparameter tuning, and deployment via Streamlit UI**.

- **Dataset:** Heart disease dataset including age, sex, chest pain type, blood pressure, cholesterol, max heart rate, and more.
- **Goal:** Predict whether a patient has heart disease (`num`) based on input features.

---

## Dataset

Columns include:
`id`, `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `num`.

Target variable: `num` (Heart disease diagnosis).

---

## Project Structure

Heart_Disease_Project/  
│── data/  
│ ├── heart_disease.csv  
│ ├── heart_disease_clean.csv   
│ └── heart_disease_selected_features.csv   
│── notebooks/  
│ ├── 01_data_preprocessing.ipynb  
│ ├── 02_pca_analysis.ipynb  
│ ├── 03_feature_selection.ipynb  
│ ├── 04_supervised_learning.ipynb  
│ ├── 05_unsupervised_learning.ipynb  
│ └── 06_hyperparameter_tuning.ipynb  
│── models/  
│ ├── Decision Tree_model.pkl  
│ ├── Logistic Regression_model.pkl  
│ ├── svm_model.pkl   
│ ├── Random Forest_model.pkl    
│ ├── scaler_model.pkl  
│ └── final_model.pkl  
│── ui/  
│ └── app.py   
│── deployment/  
│ └── streamlit_deployment.txt   
│── README.md  
│── requirements.txt  
│── .gitignore  
└── LICENSE  

---

## Features

- **Data Preprocessing & EDA:** Handle missing values, visualize distributions, correlations, and categorical features.
- **Dimensionality Reduction:** PCA to reduce feature dimensionality while retaining variance.
- **Feature Selection:** Random Forest / XGBoost feature importance, RFE, and Chi-Square tests to select key predictors.
- **Supervised Learning Models:**
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, ROC Curve & AUC score.
- **Unsupervised Learning:** K-Means clustering and hierarchical clustering with dendrograms.
- **Hyperparameter Tuning:** GridSearchCV & RandomizedSearchCV to optimize model performance.
- **Streamlit Web UI:** User-friendly interface for real-time predictions and visualization.

---

## Tools & Technologies

| Category         | Tools                         |
| ---------------- | ----------------------------- |
| Language         | Python                        |
| Data Processing  | pandas, NumPy                 |
| Machine Learning | scikit-learn, XGBoost         |
| Visualization    | Matplotlib, Seaborn, Power BI |

---

## Deployment
Demo: https://heartdisease-demo.streamlit.app/
