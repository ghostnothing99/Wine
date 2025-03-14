import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns

# Set the title of the app
st.title("Wine Quality Prediction App")

# Load trained model and scaler
# Correct paths (modify as needed)
model_path = 'models/best_wine_quality_model.joblib'
scaler_path = 'models/scaler.joblib'

# Check if model and scaler files exist
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}. Please ensure the file exists.")
    st.stop()

if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found at: {scaler_path}. Please ensure the file exists.")
    st.stop()

# Load the model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

data_path = 'data/winequality.csv'
if not os.path.exists(data_path):
    st.error(f"Dataset file not found at: {data_path}. Please ensure the file exists.")
    st.stop()

try:
    df = pd.read_csv(data_path)
    st.write("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Fix for feature name mismatch:
# Get X and y from the dataset
X = df.drop('quality', axis=1)
y = df['quality']
y = y - y.min()  # Normalize labels to start from 0

# Ensure we have the same feature names in the same order as the scaler
if hasattr(scaler, 'feature_names_in_'):
    required_features = list(scaler.feature_names_in_)
    missing_features = [feat for feat in required_features if feat not in X.columns]
    if missing_features:
        st.error(f"Missing required features in the dataset: {missing_features}")
        st.stop()
    X = X[required_features]

# Transform the features
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Sidebar for user input
st.sidebar.header("User Input Features")
input_features = {}

# Collect user input for each feature
for feature in scaler.feature_names_in_:
    input_features[feature] = st.sidebar.number_input(feature, value=0.0)

# Create a DataFrame with the input features
input_df = pd.DataFrame([input_features])

# Scale the input features
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)[0]

# If we normalized the labels before, add back the minimum value
prediction = prediction + df['quality'].min()

# Display prediction
st.subheader("Prediction")
st.write(f"The predicted wine quality is: **{prediction}**")

# Display model evaluation graphs
st.subheader("Model Evaluation")

# 1️⃣ Confusion Matrix
st.write("### Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# 2️⃣ Classification Report (Bar Chart)
st.write("### Classification Report Metrics")
report = classification_report(y, y_pred, output_dict=True)
df_report = pd.DataFrame(report).T.drop("support", axis=1)
fig_cr, ax_cr = plt.subplots(figsize=(10, 5))
df_report.plot(kind="bar", colormap="viridis", ax=ax_cr)
ax_cr.set_title("Classification Report Metrics")
ax_cr.set_ylabel("Score")
ax_cr.set_ylim(0, 1)
plt.xticks(rotation=45)
st.pyplot(fig_cr)

# 3️⃣ Model Accuracy
st.write("### Overall Model Accuracy")
accuracy = accuracy_score(y, y_pred)
fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
ax_acc.bar(["Model Accuracy"], [accuracy], color=["blue"])
ax_acc.set_ylim(0, 1)
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Overall Model Accuracy")
st.pyplot(fig_acc)

# 4️⃣ ROC Curve (One-vs-Rest for Multi-Class)
if hasattr(model, "predict_proba") and len(np.unique(y)) > 2:
    st.write("### ROC Curve (One-vs-Rest)")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for i in range(len(np.unique(y))):
        y_true = (y == i).astype(int)
        y_probs = model.predict_proba(X_scaled)[:, i]
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (One-vs-Rest)")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# 5️⃣ Precision-Recall Curve (One-vs-Rest for Multi-Class)
if hasattr(model, "predict_proba"):
    st.write("### Precision-Recall Curve (One-vs-Rest)")
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for i in range(len(np.unique(y))):
        y_true = (y == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, model.predict_proba(X_scaled)[:, i])
        ax_pr.plot(recall, precision, label=f'Class {i}')
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve (One-vs-Rest)")
    ax_pr.legend()
    st.pyplot(fig_pr)
