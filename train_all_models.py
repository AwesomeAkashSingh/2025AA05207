"""
Master Training Script - Trains All 6 Models
ML Assignment 2 - BITS Pilani

Run this script to train all 6 models at once and generate comparison table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import knn, logistic_regression, decision_tree, naive_bayes, random_forest, xgboost
import pandas as pd

def train_all_models():
    """Train all 6 classification models"""
    
    print("\n" + "="*80)
    print(" "*20 + "ML ASSIGNMENT 2 - MODEL TRAINING")
    print(" "*15 + "Credit Card Classification - BITS Pilani")
    print("="*80)
    
    results = []
    
    # 1. Logistic Regression
    print("\n[1/6] Training Logistic Regression...")
    X, y, _ = logistic_regression.load_and_preprocess_data('credit_card_data.csv')
    model, scaler, metrics, _, _ = logistic_regression.train_logistic_regression_model(X, y)
    logistic_regression.save_model(model, scaler)
    results.append({'Model': 'Logistic Regression', **metrics})
    
    # 2. Decision Tree
    print("\n[2/6] Training Decision Tree...")
    X, y, _ = decision_tree.load_and_preprocess_data('credit_card_data.csv')
    model, scaler, metrics, _, _ = decision_tree.train_decision_tree_model(X, y)
    decision_tree.save_model(model, scaler)
    results.append({'Model': 'Decision Tree', **metrics})
    
    # 3. K-Nearest Neighbors
    print("\n[3/6] Training K-Nearest Neighbors...")
    X, y, _ = knn.load_and_preprocess_data('credit_card_data.csv')
    model, scaler, metrics, _, _ = knn.train_knn_model(X, y)
    knn.save_model(model, scaler)
    results.append({'Model': 'K-Nearest Neighbors', **metrics})
    
    # 4. Naive Bayes
    print("\n[4/6] Training Naive Bayes...")
    X, y, _ = naive_bayes.load_and_preprocess_data('credit_card_data.csv')
    model, scaler, metrics, _, _ = naive_bayes.train_naive_bayes_model(X, y)
    naive_bayes.save_model(model, scaler)
    results.append({'Model': 'Naive Bayes', **metrics})
    
    # 5. Random Forest
    print("\n[5/6] Training Random Forest...")
    X, y, _ = random_forest.load_and_preprocess_data('credit_card_data.csv')
    model, scaler, metrics, _, _ = random_forest.train_random_forest_model(X, y)
    random_forest.save_model(model, scaler)
    results.append({'Model': 'Random Forest', **metrics})
    
    # 6. XGBoost
    print("\n[6/6] Training XGBoost...")
    X, y, _ = xgboost.load_and_preprocess_data('credit_card_data.csv')
    model, scaler, metrics, _, _ = xgboost.train_xgboost_model(X, y)
    xgboost.save_model(model, scaler)
    results.append({'Model': 'XGBoost', **metrics})
    
    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['Model', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']]
    comparison_df.columns = ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    # Round values
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        comparison_df[col] = comparison_df[col].round(4)
    
    # Print comparison table
    print("\n" + "="*80)
    print(" "*25 + "FINAL COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\n✓ Comparison table saved to 'model_comparison.csv'")
    
    # Save test data for Streamlit app
    print("\n✓ Generating test data for Streamlit app...")
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    test_data = pd.DataFrame(X_test)
    test_data['Target'] = y_test
    test_data.to_csv('test_data.csv', index=False)
    print(f"✓ Test data saved to 'test_data.csv' ({len(test_data)} samples)")
    
    print("\n" + "="*80)
    print(" "*25 + "✓ ALL MODELS TRAINED!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update README.md with the metrics from 'model_comparison.csv'")
    print("2. Run the Streamlit app: streamlit run app.py")
    print("3. Test the app by uploading 'test_data.csv'")
    print("="*80)

if __name__ == "__main__":
    train_all_models()
