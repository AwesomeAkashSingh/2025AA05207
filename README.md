# Credit Card Classification - ML Assignment 2

**Author:** [SINGH AKASH ARVIND MAYA]  
**Roll Number:** [2025AA05207]  
**Course:** [M.Tech (AIML/DSE) - Machine Learning]  
**Institution:** [BITS Pilani Work Integrated Learning Programme]  

---

## 📋 Problem Statement

This project implements a **binary classification system** to predict credit card approval decisions. The goal is to build and compare multiple machine learning models to determine whether a credit card application should be approved or rejected based on various applicant features.

The problem is addressed using 6 different classification algorithms, and their performance is evaluated using comprehensive metrics to identify the most suitable model for this task.

---

## 📊 Dataset Description

**Dataset Source:** [Kaggle - Credit Card Details](https://www.kaggle.com/datasets/rohitudageri/credit-card-details)

### Dataset Characteristics:
- **Type:** Binary Classification
- **Number of Features:** [X features - update after loading your dataset]
- **Number of Instances:** [Y instances - update after loading your dataset]
- **Target Variable:** Credit card approval status (Approved/Rejected)

### Features:
The dataset contains various applicant information such as:
- Demographic details (age, gender, etc.)
- Financial information (income, employment details)
- Credit history indicators
- Application-specific features

**Note:** The exact feature names and descriptions depend on the specific Kaggle dataset downloaded. Please update this section after examining your downloaded dataset.

### Data Preprocessing:
1. Handled missing values through imputation/removal
2. Encoded categorical variables using Label Encoding
3. Scaled numerical features using StandardScaler
4. Split data into 80% training and 20% testing sets

---

## 🤖 Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| kNN | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

**Instructions:** Run the `credit_card_ml_models.py` script to generate actual metrics and update this table.

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Logistic Regression provides a good baseline with interpretable coefficients. It performs well when the relationship between features and target is approximately linear. The model shows [good/moderate/poor] performance with an accuracy of XX% and handles the binary classification task [effectively/adequately]. Its regularization helps prevent overfitting. |
| **Decision Tree** | Decision Tree captures non-linear relationships and provides clear decision rules. The model achieved an accuracy of XX% with [excellent/good/moderate] recall. However, it may be prone to overfitting on training data. The tree-based approach makes it easy to understand which features are most important for credit approval decisions. |
| **kNN** | K-Nearest Neighbors shows [strong/moderate/weak] performance with XX% accuracy. The model's performance is highly dependent on the choice of k and distance metric. It performs [well/poorly] on this dataset, suggesting that [similar applicants tend to have similar outcomes / the decision boundary is complex]. The model requires feature scaling for optimal performance. |
| **Naive Bayes** | Gaussian Naive Bayes assumes feature independence and shows [competitive/moderate/limited] performance with XX% accuracy. Despite the independence assumption being unrealistic for credit data, it provides [surprisingly good/reasonable/limited] results. The model is computationally efficient and works well with [limited/sufficient] training data. |
| **Random Forest (Ensemble)** | Random Forest ensemble method demonstrates [superior/strong/good] performance with XX% accuracy and XX AUC score. The bagging approach reduces overfitting compared to single decision trees. It handles feature interactions well and provides robust predictions. The model shows [excellent/good] generalization with high [precision/recall/F1], making it [one of the best performers/a strong candidate] for this task. |
| **XGBoost (Ensemble)** | XGBoost achieves [the highest/competitive/moderate] performance with XX% accuracy and XX AUC. The gradient boosting approach iteratively improves predictions, resulting in [superior/strong] performance across all metrics. It handles imbalanced data well and captures complex patterns. With an MCC of XX, it shows [excellent/good] correlation between predictions and actual outcomes, making it [the best choice/a top performer] for credit card approval prediction. |

**General Observations:**
- Ensemble methods (Random Forest and XGBoost) generally [outperform/match/underperform] individual classifiers
- The model with the highest accuracy is: [Model Name] with XX%
- The model with the best AUC score is: [Model Name] with XX
- For this credit card approval task, [model name] is recommended due to its [high accuracy/balanced precision-recall/robust performance]

---

## 🚀 Project Structure

```
credit-card-classification/
│
├── app.py                          # Streamlit web application
├── train_all_models.py             # Master training script (trains all models)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── credit_card_data.csv            # Dataset (download from Kaggle)
│
├── model/                          # Model training scripts and saved models
│   ├── __init__.py                 # Package initializer
│   ├── knn.py                      # KNN model training script
│   ├── logistic_regression.py      # Logistic Regression training script
│   ├── decision_tree.py            # Decision Tree training script
│   ├── naive_bayes.py              # Naive Bayes training script
│   ├── random_forest.py            # Random Forest training script
│   ├── xgboost.py                  # XGBoost training script
│   │
│   └── (After training, .pkl files will be saved here)
│       ├── logistic_regression_model.pkl
│       ├── decision_tree_model.pkl
│       ├── knn_model.pkl
│       ├── naive_bayes_model.pkl
│       ├── random_forest_model.pkl
│       └── xgboost_model.pkl
│
├── model_comparison.csv            # Results comparison table (auto-generated)
└── test_data.csv                   # Test dataset for Streamlit app (auto-generated)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Steps to Run Locally

1. **Clone the repository**
   ```bash
   git clone <your-github-repo-url>
   cd credit-card-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/rohitudageri/credit-card-details)
   - Place the CSV file in the project directory
   - Rename it to `credit_card_data.csv`

4. **Train all models (one command)**
   ```bash
   python train_all_models.py
   ```
   
   This will:
   - Train all 6 models
   - Save models to `model/` folder as .pkl files
   - Generate `model_comparison.csv`
   - Create `test_data.csv` for the Streamlit app

   **Alternative: Train models individually**
   ```bash
   cd model
   python knn.py                    # Train KNN
   python logistic_regression.py   # Train Logistic Regression
   python decision_tree.py         # Train Decision Tree
   python naive_bayes.py           # Train Naive Bayes
   python random_forest.py         # Train Random Forest
   python xgboost.py               # Train XGBoost
   cd ..
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - Navigate to: `http://localhost:8501`
   - Download test data or upload your own CSV/Excel file
   - Select a model and make predictions

---

## 🌐 Deployment

This application is deployed on **Streamlit Community Cloud**.

**Live App:** [Your Streamlit App URL]

### Deployment Steps:
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository, branch, and `app.py`
6. Click "Deploy"

---

## 📱 Streamlit App Features

The web application includes:

1. **📥 Download Test Data**
   - Download sample test data in Excel or CSV format
   - Includes target column for validation
   - Ready to upload and test immediately

2. **📤 Upload CSV/Excel Files**
   - Upload your own test data (CSV or Excel)
   - Automatic data preprocessing
   - Support for categorical and numerical features

3. **🔧 Model Selection Dropdown**
   - Choose from 6 trained models
   - Easy comparison between different algorithms
   - Models load instantly from saved .pkl files

4. **📊 Comprehensive Metrics Display**
   - Accuracy, AUC, Precision, Recall, F1, MCC
   - Visual metric cards for quick insights
   - Color-coded performance indicators

5. **🎯 Confusion Matrix Visualization**
   - Interactive heatmap
   - Easy interpretation of predictions
   - Shows true vs predicted labels

6. **📋 Detailed Classification Report**
   - Per-class metrics breakdown
   - Precision, recall, F1-score for each class
   - Support column showing sample sizes

7. **🔍 Prediction Results Table**
   - Individual prediction viewing
   - Comparison of true vs predicted labels
   - Download predictions as CSV
   - Shows correct/incorrect predictions

---

## 📈 Evaluation Metrics Explained

- **Accuracy:** Overall correctness of predictions
- **AUC (Area Under ROC Curve):** Model's ability to distinguish between classes
- **Precision:** Ratio of true positives to total positive predictions
- **Recall:** Ratio of true positives to actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure for binary classification

---

## 🔍 Key Insights

[Update this section after running your models]

1. **Best Performing Model:** [Model Name] with [XX%] accuracy
2. **Most Important Metrics:** [Based on your analysis]
3. **Recommendations:** [Your recommendations for deployment]

---

## 📚 Technologies Used

- **Python 3.8+**
- **Scikit-learn:** Machine learning algorithms
- **XGBoost:** Gradient boosting
- **Streamlit:** Web application framework
- **Pandas:** Data manipulation
- **NumPy:** Numerical computing
- **Matplotlib & Seaborn:** Data visualization

---

## 👨‍💻 Author

**[Your Name]**  
Roll Number: [Your Roll Number]  
Email: [Your Email]  
BITS Pilani - M.Tech (AIML/DSE)

---

## 📝 Assignment Details

- **Course:** Machine Learning
- **Assignment:** Assignment 2
- **Total Marks:** 15
- **Submission Deadline:** 15-Feb-2026, 11:59 PM
- **Platform:** BITS Virtual Lab

---

## 📄 License

This project is created for educational purposes as part of BITS Pilani coursework.

---

## 🙏 Acknowledgments

- BITS Pilani Work Integrated Learning Programmes
- Kaggle for providing the dataset
- Streamlit Community Cloud for free hosting

---

**Note:** Remember to update all placeholders (marked with [X], XX, etc.) with actual values after running your models and deploying the application.
