import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns

# Set the title of the app
st.title("🍷 Wine Quality Prediction App")

# Debugging information
st.write("### Debugging Information")
st.write(f"Current working directory: `{os.getcwd()}`")
st.write(f"Files in directory: `{os.listdir()}`")

# Define paths
model_path = 'best_wine_quality_model.joblib'
scaler_path = 'scaler.joblib'
data_path = 'winequality.csv'

# File checks
st.write("#### File Path Checks")
st.write(f"Model path: `{os.path.abspath(model_path)}`")
st.write(f"Scaler path: `{os.path.abspath(scaler_path)}`")
st.write(f"Data path: `{os.path.abspath(data_path)}`")

# Check for required files
if not all([os.path.exists(model_path), os.path.exists(scaler_path), os.path.exists(data_path)]):
    st.error("❌ Missing required files. Please check:")
    st.error(f"- Model: {'Found' if os.path.exists(model_path) else 'Missing'}")
    st.error(f"- Scaler: {'Found' if os.path.exists(scaler_path) else 'Missing'}")
    st.error(f"- Data: {'Found' if os.path.exists(data_path) else 'Missing'}")
    st.stop()

# Load model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("✅ Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model/scaler: {str(e)}")
    st.stop()

# Load dataset
try:
    df = pd.read_csv(data_path)
    st.write(f"✅ Dataset loaded successfully (shape: {df.shape})")
except Exception as e:
    st.error(f"❌ Error loading dataset: {str(e)}")
    st.stop()

# Feature alignment
try:
    if hasattr(scaler, 'feature_names_in_'):
        required_features = list(scaler.feature_names_in_)
        X = df[required_features].copy()
    else:
        X = df.drop('quality', axis=1)
        
    y = df['quality'] - df['quality'].min()
except Exception as e:
    st.error(f"❌ Feature processing error: {str(e)}")
    st.stop()

# Model processing
try:
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
except Exception as e:
    st.error(f"❌ Prediction error: {str(e)}")
    st.stop()

# Sidebar inputs
st.sidebar.header("⚙️ Input Features")
input_features = {}
for feature in scaler.feature_names_in_:
    input_features[feature] = st.sidebar.number_input(
        feature, 
        value=0.0,
        min_value=0.0,
        format="%.2f"
    )

# Prediction
try:
    input_df = pd.DataFrame([input_features])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0] + df['quality'].min()
    st.subheader("📊 Prediction Result")
    st.metric(label="Predicted Quality", value=f"{prediction}/10")
except Exception as e:
    st.error(f"❌ Prediction failed: {str(e)}")
    st.stop()

# Visualization section
st.subheader("📈 Model Performance")

# Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y, y_pred), 
            annot=True, fmt="d", 
            cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Classification Report
st.write("#### Classification Metrics")
st.code(classification_report(y, y_pred))

# Accuracy
accuracy = accuracy_score(y, y_pred)
st.write(f"#### Overall Accuracy: {accuracy:.2%}")
st.progress(accuracy)

# ROC Curve
if hasattr(model, "predict_proba"):
    fig_roc = plt.figure()
    for i in range(len(np.unique(y))):
        y_true = (y == i).astype(int)
        probas = model.predict_proba(X_scaled)[:, i]
        fpr, tpr, _ = roc_curve(y_true, probas)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(fig_roc)
