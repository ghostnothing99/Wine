import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    """Load the wine quality dataset"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Handle missing values, and split data into features (X) and target (y)"""
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'models/scaler.joblib')  # Save in the models directory
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the dataset into training and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)