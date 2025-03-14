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

# Debugging: Print current working directory and file paths
st.write("### Debugging Information")
st.write(f"Current working directory: `{os.getcwd()}`")
st.write(f"Files in directory: `{os.listdir()}`")

# Define paths (update these if needed)
model_path = 'best_wine_quality_model.joblib'
scaler_path = 'scaler.joblib'
data_path = 'winequality.csv'

# Check if files exist (with absolute paths)
st.write("#### File Path Checks")
st.write(f"Looking for model at: `{os.path.abspath(model_path)}`")
st.write(f"Looking for scaler at: `{os.path.abspath(scaler_path)}`")
st.write(f"Looking for dataset at: `{os.path.abspath(data_path)}`")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: `{os.path.abspath(model_path)}`. Please ensure the file exists.")
    st.stop()

if not os.path.exists(scaler_path):
    st.error(f"❌ Scaler file not found at: `{os.path.abspath(scaler_path)}`. Please ensure the file exists.")
    st.stop()

# Load the model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("✅ Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# Load dataset
try:
    df = pd.read_csv(data_path)
    st.write(f"✅ Dataset `{data_path}` loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# Feature alignment check
if hasattr(scaler, 'feature_names_in_'):
    required_features = list(scaler.feature_names_in_)
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        st.error(f"❌ Missing features in dataset: {missing_features}")
        st.stop()
    X = df[required_features].copy()
else:
    X = df.drop('quality', axis=1)

y = df['quality']
y = y - y.min()  # Normalize labels to start from 0

# Transform features
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Sidebar for user input
st.sidebar.header("⚙️ User Input Features")
input_features = {}
for feature in scaler.feature_names_in_:
    input_features[feature] = st.sidebar.number_input(
        feature, 
        value=0.0,
        help=f"Enter value for {feature}"
    )

# Prediction
input_df = pd.DataFrame([input_features])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0] + df['quality'].min()

# Display prediction
st.subheader("📊 Prediction Result")
st.metric(label="Predicted Wine Quality", value=prediction)

# Model evaluation section
st.subheader("📈 Model Performance Metrics")

# Confusion Matrix
st.write("### Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix(y, y_pred), 
            annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), 
            yticklabels=np.unique(y),
            ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# Classification Report
st.write("### Classification Report")
st.code(classification_report(y, y_pred))

# Accuracy
st.write("### Model Accuracy")
accuracy = accuracy_score(y, y_pred)
st.progress(accuracy)
st.write(f"Accuracy: **{accuracy:.2%}**")

# ROC Curve (if applicable)
if hasattr(model, "predict_proba") and len(np.unique(y)) > 2:
    st.write("### ROC Curve")
    fig_roc = plt.figure()
    for i in range(len(np.unique(y))):
        fpr, tpr, _ = roc_curve((y == i), model.predict_proba(X_scaled)[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    st.pyplot(fig_roc)

# Precision-Recall Curve (if applicable)
if hasattr(model, "predict_proba"):
    st.write("### Precision-Recall Curve")
    fig_pr = plt.figure()
    for i in range(len(np.unique(y))):
        precision, recall, _ = precision_recall_curve((y == i), model.predict_proba(X_scaled)[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    st.pyplot(fig_pr)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
