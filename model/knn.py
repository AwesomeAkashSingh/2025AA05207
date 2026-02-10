import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['Ind_ID', 'EMAIL_ID'], axis=1, errors='ignore')
    
    # ROBUST FILLING: Fill all numerical NaNs with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col in df.columns and col != 'label':
            df[col] = df[col].fillna(df[col].median())
            
    # ROBUST FILLING: Fill all categorical NaNs with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Label Encoding
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y, le

def train_knn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    return model, scaler, metrics, X_test, y_test

def save_model(model, scaler):
    with open(f'model/knn_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)