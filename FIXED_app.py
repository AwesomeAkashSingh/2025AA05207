"""
Streamlit Web Application for Credit Card Classification
ML Assignment 2 - BITS Pilani

Features:
- Download test data option
- Upload CSV/Excel file for testing
- Model selection dropdown
- Display all evaluation metrics
- Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Page configuration
st.set_page_config(
    page_title="Credit Card Classification",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name):
    """Load a trained model from pickle file"""
    model_files = {
        'Logistic Regression': 'model/logistic_regression_model.pkl',
        'Decision Tree': 'model/decision_tree_model.pkl',
        'K-Nearest Neighbors': 'model/knn_model.pkl',
        'Naive Bayes': 'model/naive_bayes_model.pkl',
        'Random Forest': 'model/random_forest_model.pkl',
        'XGBoost': 'model/xgboost_model.pkl'
    }
    
    try:
        with open(model_files[model_name], 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['scaler']
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_files[model_name]}")
        st.info("Please train the models first by running: python train_all_models.py")
        return None, None


def preprocess_uploaded_data(df):
    """Preprocess uploaded data - encode categorical variables"""
    df_processed = df.copy()
    
    # Encode categorical variables
    for col in df_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    return df_processed


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {}
    
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    # AUC score
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics


def plot_confusion_matrix(cm, title):
    """Create confusion matrix plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    return fig


def main():
    # Header
    st.title("💳 Credit Card Classification System")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h3 style='margin: 0; color: #1f77b4;'>ML Assignment 2 - BITS Pilani</h3>
        <p style='margin: 0.5rem 0 0 0;'>
            Predicting credit card approval using 6 different classification models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    st.sidebar.markdown("### 🤖 Select Model")
    model_choice = st.sidebar.selectbox(
        "Choose a classification model:",
        [
            'Logistic Regression',
            'Decision Tree',
            'K-Nearest Neighbors',
            'Naive Bayes',
            'Random Forest',
            'XGBoost'
        ],
        help="Select the machine learning model for predictions"
    )
    
    st.sidebar.markdown("---")
    
    # Data section
    st.sidebar.markdown("### 📊 Data Options")
    
    # Option 1: Download test data
    st.sidebar.markdown("#### Download Test Data")
    try:
        test_data = pd.read_csv('test_data.csv')
        
        # Convert to Excel for download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            test_data.to_excel(writer, index=False, sheet_name='Test Data')
        
        st.sidebar.download_button(
            label="📥 Download Test Data (Excel)",
            data=buffer.getvalue(),
            file_name="credit_card_test_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download sample test data with target column"
        )
        
        # Also provide CSV download
        csv = test_data.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download Test Data (CSV)",
            data=csv,
            file_name="credit_card_test_data.csv",
            mime="text/csv",
            help="Download sample test data in CSV format"
        )
        
        st.sidebar.info(f"✓ Test data ready ({len(test_data)} samples)")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Test data not found. Train models first.")
    
    st.sidebar.markdown("---")
    
    # Option 2: Upload file
    st.sidebar.markdown("#### Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your test dataset (must include target column)"
    )
    
    # Main content area
    if uploaded_file is not None:
        # Load uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Rows", df.shape[0])
            col2.metric("📋 Columns", df.shape[1])
            col3.metric("❓ Missing Values", df.isnull().sum().sum())
            
            # Show preview
            with st.expander("👀 Preview Data (First 10 rows)", expanded=False):
                st.dataframe(df.head(10))
            
            # Auto-detect target column (last column)
            target_col = df.columns[-1]
            st.info(f"📌 **Target column (auto-detected):** `{target_col}` (last column)")
            
            # Prepare data
            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner("Loading model and making predictions..."):
                    
                    # Load model
                    model, scaler = load_model(model_choice)
                    
                    if model is None:
                        st.stop()
                    
                    # Prepare features and target
                    X = df.drop(columns=[target_col])
                    y_true = df[target_col]
                    
                    # Encode target if needed
                    if y_true.dtype == 'object':
                        le_target = LabelEncoder()
                        y_true = le_target.fit_transform(y_true)
                    
                    # Preprocess features
                    X_processed = preprocess_uploaded_data(X)
                    
                    # Scale features
                    X_scaled = scaler.transform(X_processed)
                    
                    # Make predictions
                    y_pred = model.predict(X_scaled)
                    
                    try:
                        y_pred_proba = model.predict_proba(X_scaled)
                    except:
                        y_pred_proba = None
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
                    
                    st.success("✅ Prediction completed successfully!")
                    
                    # Display metrics
                    st.markdown("---")
                    st.markdown(f"## 📈 Model Performance: **{model_choice}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #1f77b4; margin: 0;'>Accuracy</h3>
                            <h2 style='margin: 0.5rem 0;'>{:.4f}</h2>
                            <p style='margin: 0; font-size: 0.9rem;'>Overall correctness</p>
                        </div>
                        """.format(metrics['Accuracy']), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #ff7f0e; margin: 0;'>AUC Score</h3>
                            <h2 style='margin: 0.5rem 0;'>{:.4f}</h2>
                            <p style='margin: 0; font-size: 0.9rem;'>Class separation</p>
                        </div>
                        """.format(metrics['AUC']), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #2ca02c; margin: 0;'>F1 Score</h3>
                            <h2 style='margin: 0.5rem 0;'>{:.4f}</h2>
                            <p style='margin: 0; font-size: 0.9rem;'>Harmonic mean</p>
                        </div>
                        """.format(metrics['F1 Score']), unsafe_allow_html=True)
                    
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                    
                    with col5:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    
                    with col6:
                        st.metric("MCC", f"{metrics['MCC']:.4f}")
                    
                    st.markdown("---")
                    
                    # Confusion Matrix
                    st.markdown("## 🎯 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig = plot_confusion_matrix(cm, model_choice)
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    # Classification Report
                    st.markdown("## 📋 Classification Report")
                    report = classification_report(y_true, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    st.dataframe(
                        report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    # Predictions table
                    st.markdown("## 🔍 Prediction Results")
                    
                    results_df = pd.DataFrame({
                        'True Label': y_true,
                        'Predicted Label': y_pred,
                        'Match': ['✓' if t == p else '✗' for t, p in zip(y_true, y_pred)]
                    })
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Predictions", len(results_df))
                    col2.metric("Correct Predictions", (results_df['True Label'] == results_df['Predicted Label']).sum())
                    col3.metric("Incorrect Predictions", (results_df['True Label'] != results_df['Predicted Label']).sum())
                    
                    st.dataframe(results_df.head(50), use_container_width=True)
                    
                    # Download predictions
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name=f"predictions_{model_choice.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file has the correct format and includes a target column.")
    
    else:
        # Instructions when no file uploaded
        st.info("👈 **Please upload a file or download test data from the sidebar to begin**")
        
        st.markdown("---")
        st.markdown("## 📝 Instructions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔽 Option 1: Use Test Data
            1. Click **"Download Test Data"** in the sidebar
            2. You'll get a sample dataset with labels
            3. Upload it back for testing
            4. Select model and run prediction
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Option 2: Use Your Own Data
            1. Prepare your CSV or Excel file
            2. Ensure it includes a target column
            3. Upload using the file uploader
            4. Select target column and run prediction
            """)
        
        st.markdown("---")
        st.markdown("## 🤖 Available Models")
        
        models_info = {
            'Logistic Regression': 'Linear model for binary classification with interpretable coefficients',
            'Decision Tree': 'Tree-based model that creates decision rules from features',
            'K-Nearest Neighbors': 'Instance-based learning using distance metrics (k=5)',
            'Naive Bayes': 'Probabilistic classifier based on Bayes theorem (Gaussian)',
            'Random Forest': 'Ensemble of decision trees for robust predictions',
            'XGBoost': 'Gradient boosting ensemble for high performance'
        }
        
        for model, desc in models_info.items():
            st.markdown(f"**{model}**: {desc}")
        
        st.markdown("---")
        st.markdown("## 📊 Evaluation Metrics")
        
        metrics_info = {
            'Accuracy': 'Ratio of correct predictions to total predictions',
            'AUC': 'Area Under ROC Curve - measures class separation ability',
            'Precision': 'Ratio of true positives to predicted positives',
            'Recall': 'Ratio of true positives to actual positives',
            'F1 Score': 'Harmonic mean of precision and recall',
            'MCC': 'Matthews Correlation Coefficient - balanced measure'
        }
        
        for metric, desc in metrics_info.items():
            st.markdown(f"**{metric}**: {desc}")
        
        # Show pre-trained model comparison if available
        st.markdown("---")
        try:
            comparison_df = pd.read_csv('model_comparison.csv')
            st.markdown("## 📊 Pre-trained Models Comparison")
            st.dataframe(
                comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'F1']),
                use_container_width=True
            )
        except:
            pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 1rem; color: #666;'>
            <p><strong>ML Assignment 2</strong> | BITS Pilani M.Tech (AIML/DSE)</p>
            <p>Credit Card Classification System</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
