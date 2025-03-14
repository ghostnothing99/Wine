import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import joblib

# 🔹 Load trained model and scaler
model = joblib.load('models/best_wine_quality_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# 🔹 Load dataset (Ensure it's the same dataset used for training/testing)
df = pd.read_csv('D:/My/Wine/data/winequality.csv')

# 🔹 Split into features and labels
X = df.drop('quality', axis=1)
y = df['quality']
y = y - y.min()  # Normalize labels to start from 0

# 🔹 Scale features
X_scaled = scaler.transform(X)

# 🔹 Predict on dataset
y_pred = model.predict(X_scaled)

# 🔹 Classification Report
report = classification_report(y, y_pred, output_dict=True)
df_report = pd.DataFrame(report).T.drop("support", axis=1)

# 📊 **1️⃣ Confusion Matrix**
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 📊 **2️⃣ Classification Report (Bar Chart)**
df_report.plot(kind="bar", figsize=(10, 5), colormap="viridis")
plt.title("Classification Report Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# 📊 **3️⃣ Model Accuracy**
accuracy = accuracy_score(y, y_pred)
plt.figure(figsize=(6, 4))
plt.bar(["Model Accuracy"], [accuracy], color=["blue"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Overall Model Accuracy")
plt.show()

# 📊 **4️⃣ ROC Curve (One-vs-Rest for Multi-Class)**
if hasattr(model, "predict_proba") and len(np.unique(y)) > 2:
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y))):
        y_true = (y == i).astype(int)  # Convert to binary format
        y_probs = model.predict_proba(X_scaled)[:, i]
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.show()

# 📊 **5️⃣ Precision-Recall Curve (One-vs-Rest for Multi-Class)**
if hasattr(model, "predict_proba"):
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y))):
        y_true = (y == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, model.predict_proba(X_scaled)[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (One-vs-Rest)")
    plt.legend()
    plt.show()

# 📊 **6️⃣ Average Precision Score**
average_precision = average_precision_score(y, model.predict_proba(X_scaled), average="macro")
print(f"\n📌 Average Precision-Recall Score (Macro-Averaged): {average_precision:.4f}")

print("\n✅ All performance metrics visualized successfully!")
