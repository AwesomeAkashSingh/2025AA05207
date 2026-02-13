"""
FIXED Naive Bayes Classifier (Gaussian)
ML Assignment 2 - BITS Pilani

FIXES:
- Added SMOTE to handle class imbalance
- This prevents zero precision/recall/F1 scores
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
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


def train_naive_bayes_model(X, y):
    """Train Naive Bayes model with SMOTE for handling class imbalance"""
    print("\n" + "="*70)
    print("TRAINING NAIVE BAYES CLASSIFIER")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # FIX: Apply SMOTE to balance classes (only on training data)
    print("\nApplying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Original training samples: {len(y_train)}")
    print(f"After SMOTE: {len(y_train_balanced)}")
    
    print("\nTraining Naive Bayes (Gaussian) model on balanced data...")
    model = GaussianNB()
    model.fit(X_train_balanced, y_train_balanced)  # ← Use balanced data!
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
    print("NAIVE BAYES MODEL PERFORMANCE")
    print("="*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    print("="*70)
    
    return model, scaler, metrics, X_test_scaled, y_test


def save_model(model, scaler, filename='model/naive_bayes_model.pkl'):
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
    model, scaler, metrics, X_test, y_test = train_naive_bayes_model(X, y)
    save_model(model, scaler)
    print("\n✓ Naive Bayes model training complete!")
