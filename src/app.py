# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd
# import os
# from sklearn.metrics import accuracy_score
# import seaborn as sns
# import io
# import base64
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# app = Flask(__name__)

# # Load trained model and scaler
# model_path = 'models/best_wine_quality_model.joblib'
# scaler_path = 'models/scaler.joblib'

# if os.path.exists(model_path) and os.path.exists(scaler_path):
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     # Print feature names used during training (for debugging)
#     if hasattr(scaler, 'feature_names_in_'):
#         print("Model was trained with these features:", scaler.feature_names_in_)
# else:
#     raise FileNotFoundError('Model or Scaler not found. Please train the model first.')

# # Load dataset (used for model evaluation)
# df = pd.read_csv('D:/My/Wine/data/winequality.csv')

# # Fix for feature name mismatch:
# # Get X and y from the dataset
# X = df.drop('quality', axis=1)
# y = df['quality']
# y = y - y.min()  # Normalize labels to start from 0

# # Option 1: If there's an 'Id' column that wasn't in the training set, drop it
# if 'Id' in X.columns and 'Id' not in scaler.feature_names_in_:
#     X = X.drop('Id', axis=1)

# # Option 2: Ensure we have the same feature names in the same order
# if hasattr(scaler, 'feature_names_in_'):
#     # Check if we have all the required features
#     required_features = list(scaler.feature_names_in_)
    
#     # If column names differ only in case or spacing, try to standardize them
#     current_columns = list(X.columns)
#     column_mapping = {}
    
#     for req_feature in required_features:
#         # Try to find a matching column regardless of case
#         for col in current_columns:
#             if col.lower().replace(" ", "") == req_feature.lower().replace(" ", ""):
#                 column_mapping[col] = req_feature
    
#     # Rename columns if needed
#     if column_mapping:
#         X = X.rename(columns=column_mapping)
    
#     # Ensure all required features exist
#     missing_features = [feat for feat in required_features if feat not in X.columns]
#     if missing_features:
#         raise ValueError(f"Missing required features: {missing_features}")
    
#     # Select only the features used during training, in the same order
#     X = X[required_features]

# # Transform the features
# X_scaled = scaler.transform(X)
# y_pred = model.predict(X_scaled)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/graphs')
# def graphs():
#     # Function to convert plot to base64 string
#     def plot_to_base64(fig):
#         img = io.BytesIO()
#         fig.savefig(img, format='png')
#         img.seek(0)
#         return base64.b64encode(img.getvalue()).decode()

#     plot_urls = {}

#     # 1️⃣ Confusion Matrix
#     fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
#     cm = confusion_matrix(y, y_pred)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax_cm)
#     ax_cm.set_xlabel("Predicted")
#     ax_cm.set_ylabel("Actual")
#     ax_cm.set_title("Confusion Matrix")
#     plot_urls['confusion_matrix'] = plot_to_base64(fig_cm)
#     plt.close(fig_cm)

#     # 2️⃣ Classification Report (Bar Chart)
#     report = classification_report(y, y_pred, output_dict=True)
#     df_report = pd.DataFrame(report).T.drop("support", axis=1)
#     fig_cr, ax_cr = plt.subplots(figsize=(10, 5))
#     df_report.plot(kind="bar", colormap="viridis", ax=ax_cr)
#     ax_cr.set_title("Classification Report Metrics")
#     ax_cr.set_ylabel("Score")
#     ax_cr.set_ylim(0, 1)
#     plt.xticks(rotation=45)
#     plot_urls['classification_report'] = plot_to_base64(fig_cr)
#     plt.close(fig_cr)

#     # 3️⃣ Model Accuracy
#     accuracy = accuracy_score(y, y_pred)
#     fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
#     ax_acc.bar(["Model Accuracy"], [accuracy], color=["blue"])
#     ax_acc.set_ylim(0, 1)
#     ax_acc.set_ylabel("Accuracy")
#     ax_acc.set_title("Overall Model Accuracy")
#     plot_urls['model_accuracy'] = plot_to_base64(fig_acc)
#     plt.close(fig_acc)

#     # 4️⃣ ROC Curve (One-vs-Rest for Multi-Class)
#     if hasattr(model, "predict_proba") and len(np.unique(y)) > 2:
#         fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
#         for i in range(len(np.unique(y))):
#             y_true = (y == i).astype(int)
#             y_probs = model.predict_proba(X_scaled)[:, i]
#             fpr, tpr, _ = roc_curve(y_true, y_probs)
#             roc_auc = auc(fpr, tpr)
#             ax_roc.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
#         ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--")
#         ax_roc.set_xlabel("False Positive Rate")
#         ax_roc.set_ylabel("True Positive Rate")
#         ax_roc.set_title("ROC Curve (One-vs-Rest)")
#         ax_roc.legend(loc="lower right")
#         plot_urls['roc_curve'] = plot_to_base64(fig_roc)
#         plt.close(fig_roc)

#     # 5️⃣ Precision-Recall Curve (One-vs-Rest for Multi-Class)
#     if hasattr(model, "predict_proba"):
#         fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
#         for i in range(len(np.unique(y))):
#             y_true = (y == i).astype(int)
#             precision, recall, _ = precision_recall_curve(y_true, model.predict_proba(X_scaled)[:, i])
#             ax_pr.plot(recall, precision, label=f'Class {i}')
#         ax_pr.set_xlabel("Recall")
#         ax_pr.set_ylabel("Precision")
#         ax_pr.set_title("Precision-Recall Curve (One-vs-Rest)")
#         ax_pr.legend()
#         plot_urls['precision_recall_curve'] = plot_to_base64(fig_pr)
#         plt.close(fig_pr)

#     return render_template('graphs.html', plot_urls=plot_urls)

# @app.route('/about')
# def about():
#     # Evaluate the best model
#     y_train_pred = model.predict(X_scaled)
#     test_accuracy = accuracy_score(y, y_train_pred)
    
#     return render_template('about.html', test_accuracy=test_accuracy)

# # Add a predict route for user input
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get feature values from form
#         if request.method == 'POST':
#             # Extract features from form
#             input_features = {}
            
#             # Use the expected feature names from the model
#             for feature in scaler.feature_names_in_:
#                 input_features[feature] = float(request.form.get(feature, 0))
            
#             # Create a DataFrame with the input features
#             input_df = pd.DataFrame([input_features])
            
#             # Scale the input features
#             input_scaled = scaler.transform(input_df)
            
#             # Make prediction
#             prediction = model.predict(input_scaled)[0]
            
#             # If we normalized the labels before, add back the minimum value
#             prediction = prediction + df['quality'].min()
            
#             return render_template('result.html', prediction=prediction)
#     except Exception as e:
#         return render_template('error.html', error=str(e))

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=True, host='0.0.0.0', port=port)


import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

# Load trained model and scaler
model_path = 'models/best_wine_quality_model.joblib'
scaler_path = 'models/scaler.joblib'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    # Print feature names used during training (for debugging)
    if hasattr(scaler, 'feature_names_in_'):
        st.write("Model was trained with these features:", scaler.feature_names_in_)
else:
    raise FileNotFoundError('Model or Scaler not found. Please train the model first.')

# Load dataset (used for model evaluation)
df = pd.read_csv('D:/My/Wine/data/winequality.csv')

# Fix for feature name mismatch:
# Get X and y from the dataset
X = df.drop('quality', axis=1)
y = df['quality']
y = y - y.min()  # Normalize labels to start from 0

# Option 1: If there's an 'Id' column that wasn't in the training set, drop it
if 'Id' in X.columns and 'Id' not in scaler.feature_names_in_:
    X = X.drop('Id', axis=1)

# Option 2: Ensure we have the same feature names in the same order
if hasattr(scaler, 'feature_names_in_'):
    # Check if we have all the required features
    required_features = list(scaler.feature_names_in_)
    
    # If column names differ only in case or spacing, try to standardize them
    current_columns = list(X.columns)
    column_mapping = {}
    
    for req_feature in required_features:
        # Try to find a matching column regardless of case
        for col in current_columns:
            if col.lower().replace(" ", "") == req_feature.lower().replace(" ", ""):
                column_mapping[col] = req_feature
    
    # Rename columns if needed
    if column_mapping:
        X = X.rename(columns=column_mapping)
    
    # Ensure all required features exist
    missing_features = [feat for feat in required_features if feat not in X.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the features used during training, in the same order
    X = X[required_features]

# Transform the features
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Streamlit app
st.title('Wine Quality Prediction')

# Sidebar for user input
st.sidebar.header('User Input Features')

# Collect user input features
input_features = {}
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
st.subheader('Prediction')
st.write(f'The predicted wine quality is: {prediction}')

# Display model evaluation graphs
st.subheader('Model Evaluation Graphs')

# 1️⃣ Confusion Matrix
st.write('### Confusion Matrix')
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# 2️⃣ Classification Report (Bar Chart)
st.write('### Classification Report Metrics')
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
st.write('### Overall Model Accuracy')
accuracy = accuracy_score(y, y_pred)
fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
ax_acc.bar(["Model Accuracy"], [accuracy], color=["blue"])
ax_acc.set_ylim(0, 1)
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Overall Model Accuracy")
st.pyplot(fig_acc)

# 4️⃣ ROC Curve (One-vs-Rest for Multi-Class)
if hasattr(model, "predict_proba") and len(np.unique(y)) > 2:
    st.write('### ROC Curve (One-vs-Rest)')
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
    st.write('### Precision-Recall Curve (One-vs-Rest)')
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

# About section
st.sidebar.header('About')
st.sidebar.write('This app predicts wine quality based on input features.')
st.sidebar.write('Model Accuracy:', accuracy_score(y, y_pred))