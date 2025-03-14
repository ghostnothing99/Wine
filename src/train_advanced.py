# import pandas as pd
# import numpy as np
# import joblib
# import optuna
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
# from imblearn.over_sampling import SMOTE
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load dataset
# df = pd.read_csv('D:/My/Wine/data/winequality.csv')

# # Check for missing values
# print("Missing values:\n", df.isnull().sum())

# # Handle missing values (if any)
# df = df.dropna()

# # Check for class imbalance
# print("Class distribution:\n", df['quality'].value_counts())

# # Normalize class labels to start from 0
# y = df['quality']
# y = y - y.min()  # Normalize labels to start from 0

# # Handle class imbalance using SMOTE
# X = df.drop('quality', axis=1)
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Correlation analysis
# plt.figure(figsize=(10, 8))
# sns.heatmap(X_resampled.corr(), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

# # Drop highly correlated features
# correlation_threshold = 0.9
# correlation_matrix = X_resampled.corr().abs()
# upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
# high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
# X_resampled = X_resampled.drop(columns=high_corr_features)

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_resampled)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# # Define models
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
#     "LightGBM": LGBMClassifier(random_state=42),
#     "SVM": SVC(probability=True, random_state=42)
# }

# # Cross-validation and model evaluation
# cv_results = {}
# for name, model in models.items():
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
#     cv_results[name] = scores.mean()
#     print(f"{name} Cross-Validation Accuracy: {scores.mean():.4f}")

# # Select the best model
# best_model_name = max(cv_results, key=cv_results.get)
# best_model = models[best_model_name]

# # Hyperparameter tuning using Optuna
# def objective(trial):
#     if best_model_name == "XGBoost":
#         params = {
#             "n_estimators": trial.suggest_int("n_estimators", 50, 300),
#             "max_depth": trial.suggest_int("max_depth", 3, 15),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
#             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
#         }
#         model = XGBClassifier(**params, eval_metric='mlogloss', random_state=42)
#     elif best_model_name == "LightGBM":
#         params = {
#             "n_estimators": trial.suggest_int("n_estimators", 50, 300),
#             "max_depth": trial.suggest_int("max_depth", 3, 15),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
#             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
#         }
#         model = LGBMClassifier(**params, random_state=42)
#     elif best_model_name == "Random Forest":
#         params = {
#             "n_estimators": trial.suggest_int("n_estimators", 50, 300),
#             "max_depth": trial.suggest_int("max_depth", 3, 15),
#             "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
#         }
#         model = RandomForestClassifier(**params, random_state=42)
#     elif best_model_name == "SVM":
#         params = {
#             "C": trial.suggest_float("C", 0.1, 10.0),
#             "gamma": trial.suggest_float("gamma", 0.01, 1.0)
#         }
#         model = SVC(**params, probability=True, random_state=42)
#     else:
#         params = {
#             "C": trial.suggest_float("C", 0.1, 10.0),
#             "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
#         }
#         model = LogisticRegression(**params, max_iter=1000, random_state=42)

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     score = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
#     return score.mean()

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)
# best_params = study.best_params

# # Train the best model with optimized hyperparameters
# if best_model_name == "XGBoost":
#     best_model = XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42)
# elif best_model_name == "LightGBM":
#     best_model = LGBMClassifier(**best_params, random_state=42)
# elif best_model_name == "Random Forest":
#     best_model = RandomForestClassifier(**best_params, random_state=42)
# elif best_model_name == "SVM":
#     best_model = SVC(**best_params, probability=True, random_state=42)
# else:
#     best_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)

# best_model.fit(X_train, y_train)

# # Evaluate the best model
# y_train_pred = best_model.predict(X_train)
# y_test_pred = best_model.predict(X_test)

# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# print(f"\n🔍 Best Model: {best_model_name}")
# print(f"🎯 Training Accuracy: {train_accuracy:.4f}")
# print(f"📊 Test Accuracy: {test_accuracy:.4f}")
# print("\n📌 Classification Report:\n", classification_report(y_test, y_test_pred))
# print("\n🛠️ Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# # Save the best model and scaler
# joblib.dump(best_model, 'models/best_wine_quality_model.joblib')
# joblib.dump(scaler, 'models/scaler.joblib')
# print("\n✅ Best Model & Scaler saved successfully!")


import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping  # Import early_stopping for LightGBM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('D:/My/Wine/data/winequality.csv')

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Handle missing values (if any)
df = df.dropna()

# Check for class imbalance
print("Class distribution:\n", df['quality'].value_counts())

# Normalize class labels to start from 0
y = df['quality']
y = y - y.min()  # Normalize labels to start from 0

# Handle class imbalance using SMOTE
X = df.drop('quality', axis=1)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Correlation analysis
plt.figure(figsize=(10, 8))
sns.heatmap(X_resampled.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Drop highly correlated features
correlation_threshold = 0.9
correlation_matrix = X_resampled.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
X_resampled = X_resampled.drop(columns=high_corr_features)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),  # Set verbosity to suppress warnings
    "SVM": SVC(probability=True, random_state=42)
}

# Cross-validation and model evaluation
cv_results = {}
for name, model in models.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_results[name] = scores.mean()
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.4f}")

# Select the best model
best_model_name = max(cv_results, key=cv_results.get)
best_model = models[best_model_name]

# Hyperparameter tuning using Optuna
def objective(trial):
    if best_model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # Reduced n_estimators
            "max_depth": trial.suggest_int("max_depth", 3, 8),  # Reduced max_depth
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),  # Increased reg_alpha
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.0)  # Increased reg_lambda
        }
        model = XGBClassifier(**params, eval_metric='mlogloss', random_state=42)
    elif best_model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # Reduced n_estimators
            "max_depth": trial.suggest_int("max_depth", 3, 8),  # Reduced max_depth
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),  # Increased reg_alpha
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.0),  # Increased reg_lambda
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50)  # Added min_child_samples
        }
        model = LGBMClassifier(**params, random_state=42, verbosity=-1)
    elif best_model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # Reduced n_estimators
            "max_depth": trial.suggest_int("max_depth", 3, 10),  # Reduced max_depth
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }
        model = RandomForestClassifier(**params, random_state=42)
    elif best_model_name == "SVM":
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0)
        }
        model = SVC(**params, probability=True, random_state=42)
    else:
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        }
        model = LogisticRegression(**params, max_iter=1000, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    return score.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params

# Train the best model with optimized hyperparameters
if best_model_name == "XGBoost":
    best_model = XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
elif best_model_name == "LightGBM":
    best_model = LGBMClassifier(**best_params, random_state=42, verbosity=-1)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping(10)])
elif best_model_name == "Random Forest":
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
elif best_model_name == "SVM":
    best_model = SVC(**best_params, probability=True, random_state=42)
    best_model.fit(X_train, y_train)
else:
    best_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
    best_model.fit(X_train, y_train)

# Function to check overfitting, underfitting, or balanced model
def check_model_balance(train_accuracy, test_accuracy, threshold=0.05):
    """
    Determines if the model is overfitting, underfitting, or balanced.
    
    Parameters:
        train_accuracy (float): Accuracy on the training set.
        test_accuracy (float): Accuracy on the test set.
        threshold (float): Threshold to determine significant difference.
    
    Returns:
        str: "Overfitting", "Underfitting", or "Balanced".
    """
    if train_accuracy > test_accuracy + threshold:
        return "Overfitting"
    elif test_accuracy > train_accuracy + threshold:
        return "Underfitting"
    else:
        return "Balanced"

# Evaluate the best model
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Check if the model is overfitting, underfitting, or balanced
model_status = check_model_balance(train_accuracy, test_accuracy)

# Print evaluation results
print(f"\n🔍 Best Model: {best_model_name}")
print(f"🎯 Training Accuracy: {train_accuracy:.4f}")
print(f"📊 Test Accuracy: {test_accuracy:.4f}")
print(f"🔄 Model Status: {model_status}")
print("\n📌 Classification Report:\n", classification_report(y_test, y_test_pred))
print("\n🛠️ Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Save the best model and scaler
joblib.dump(best_model, 'models/best_wine_quality_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
print("\n✅ Best Model & Scaler saved successfully!")