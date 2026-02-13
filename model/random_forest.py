"""
FIXED Random Forest Classifier (Ensemble)
ML Assignment 2 - BITS Pilani

FIXES:
- Added class_weight='balanced' for better handling of imbalanced data
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath='credit_card_data.csv'):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    
    df = df.dropna()
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    return X, y, label_encoders


def train_random_forest_model(X, y):
    """Train Random Forest model with class balancing"""
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST CLASSIFIER (ENSEMBLE)")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Random Forest model with 100 estimators and class balancing...")
    # FIX: Added class_weight='balanced'
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # ← THIS HELPS WITH IMBALANCED DATA
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Training complete!")
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    print("\nCalculating metrics...")
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    
    try:
        if len(np.unique(y_test)) == 2:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        metrics['auc'] = 0.0
    
    print("\n" + "="*70)
    print("RANDOM FOREST MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    print("="*70)
    
    return model, scaler, metrics, X_test_scaled, y_test


def save_model(model, scaler, filename='model/random_forest_model.pkl'):
    """Save trained model and scaler"""
    model_data = {
        'model': model,
        'scaler': scaler
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to {filename}")


if __name__ == "__main__":
    X, y, label_encoders = load_and_preprocess_data('credit_card_data.csv')
    model, scaler, metrics, X_test, y_test = train_random_forest_model(X, y)
    save_model(model, scaler)
    print("\n✓ Random Forest model training complete!")
