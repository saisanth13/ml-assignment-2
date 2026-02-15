import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification App")
st.write(
    "This app allows you to upload a dataset and evaluate different "
    "machine learning classification models."
)

# ===============================
# Load preprocessing objects
# ===============================
imputer = joblib.load("model/imputer.pkl")
scaler = joblib.load("model/scaler.pkl")

# ===============================
# Load models
# ===============================
models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl"),
}

# ===============================
# Model selection
# ===============================
model_name = st.selectbox("Select a Model", list(models.keys()))
model = models[model_name]

# ===============================
# File upload
# ===============================
uploaded_file = st.file_uploader(
    "Upload CSV file (Breast Cancer Dataset)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Drop completely empty columns (e.g. Unnamed: 32)
    df = df.dropna(axis=1, how="all")

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Separate target if present
    if "diagnosis" in df.columns:
        X = df.drop("diagnosis", axis=1)
        y_true = df["diagnosis"].map({"B": 0, "M": 1})
    else:
        X = df.copy()
        y_true = None

    # ===============================
    # Preprocessing (same as training)
    # ===============================
    X = imputer.transform(X)
    X_scaled = scaler.transform(X)

    # ===============================
    # Predictions
    # ===============================
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Add predictions to dataframe
    df["Prediction"] = y_pred
    df["Probability"] = y_prob

    st.subheader("Predictions")
    st.dataframe(df.head())

    # ===============================
    # Evaluation metrics (only if target exists)
    # ===============================
    if y_true is not None:
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            st.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")

        with col2:
            st.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
            st.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")

        with col3:
            st.metric("AUC", f"{roc_auc_score(y_true, y_prob):.4f}")
            st.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        # ===============================
        # Confusion Matrix
        # ===============================
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"]
        )
        st.dataframe(cm_df)

    # ===============================
    # Download predictions
    # ===============================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions as CSV",
        csv,
        "predictions.csv",
        "text/csv"
    )
