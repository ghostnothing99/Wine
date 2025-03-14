import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load dataset
df = pd.read_csv('D:/My/Wine/data/winequality.csv')  # Ensure this path is correct

# Split data into features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Detect Overfitting or Underfitting
overfitting_warning = None
if train_accuracy > test_accuracy + 0.1:
    overfitting_warning = "⚠️ Warning: Potential Overfitting detected!"
elif test_accuracy > train_accuracy + 0.1:
    overfitting_warning = "⚠️ Warning: Potential Underfitting detected!"
else:
    overfitting_warning = "✅ Model is well-balanced (No overfitting/underfitting detected)."

# Print Insights
print("\n🔍 Model Evaluation:")
print(f"🎯 Training Accuracy: {train_accuracy:.4f}")
print(f"📊 Test Accuracy: {test_accuracy:.4f}")
print(overfitting_warning)

print("\n📌 Classification Report:\n", classification_report(y_test, y_test_pred))
print("\n🛠️ Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Ensure 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model & scaler
joblib.dump(model, 'models/wine_quality_predictor_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

print("\n✅ Model & Scaler saved successfully!")
