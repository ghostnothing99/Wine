from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def train_model(X_train, y_train):
    """Train a random forest model"""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, 'models/best_wine_quality_model.joblib')  # Save model
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data"""
    y_pred = model.predict(X_test)
    
    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    # Classification report
    print(classification_report(y_test, y_pred))
    
    return accuracy