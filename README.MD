# 🏦 Credit Card Default Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.readthedocs.io)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-success.svg)](https://uci-credit-card-default-prediction.onrender.com)

## 📋 Project Overview

A complete machine learning system that predicts credit card payment defaults with **81% ROC-AUC accuracy**. This project demonstrates end-to-end ML engineering from data preprocessing to production deployment.

**🌐 Live Application**: [https://uci-credit-card-default-prediction.onrender.com](https://uci-credit-card-default-prediction.onrender.com)

### 🎯 Problem Statement
Predict whether a credit card customer will default on their payment next month based on their demographic information, payment history, and bill amounts.

## 🚀 Technical Implementation

### 🤖 **Machine Learning Pipeline**
- **7 Algorithms Tested**: XGBoost, CatBoost, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, Logistic Regression
- **Best Model**: XGBoost with **81% ROC-AUC** performance
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Evaluation Metric**: ROC-AUC (industry standard for imbalanced classification)

### 🔧 **Feature Engineering** 
Created **17 engineered features** from original 24 features:

**Credit Behavior Features:**
- Credit utilization ratio (bill amount / credit limit)
- Payment efficiency ratios for each month
- Payment status patterns and reliability scores

**Financial Trends:**
- Average bill amounts and payment amounts
- Trend analysis for spending and payment patterns  
- Risk indicators for high utilization and payment behavior

### 🌐 **Web Application**
- **Live Demo**: [https://uci-credit-card-default-prediction.onrender.com](https://uci-credit-card-default-prediction.onrender.com)
- **Framework**: Flask with responsive HTML/CSS interface
- **Input**: 24 customer attributes (demographics, payment history, bill amounts)
- **Output**: Binary prediction with confidence (Will Default / Will Not Default)
- **Deployment**: Production deployment on Render cloud platform

## 📊 Dataset & Performance

### **Dataset Information**
- **Source**: UCI Credit Card Default Dataset
- **Size**: 30,000 customers with 25 features
- **Target**: Binary classification (Default: Yes/No)
- **Time Period**: 6 months of payment history (April-September 2005)

### **Model Performance**
| Algorithm | ROC-AUC Score | Accuracy |
|-----------|---------------|----------|
| **XGBoost (Best)** | **0.814** | **82.3%** |
| CatBoost | 0.809 | 81.9% |
| Random Forest | 0.806 | 81.6% |
| Gradient Boosting | 0.803 | 81.4% |

## 📁 Project Structure

```
├── src/                    # Core ML Pipeline
│   ├── components/         # Data ingestion, transformation, model training
│   ├── pipeline/          # Training and prediction pipelines
│   └── utils.py           # Model evaluation utilities
├── templates/             # Flask web interface
├── artifacts/             # Trained models and preprocessors
├── notebooks/             # EDA and model development
└── application.py         # Flask web application
```

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Programming** | Python 3.11 |
| **ML Libraries** | Scikit-learn, XGBoost, CatBoost, Pandas, NumPy |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3 |
| **Deployment** | Render Cloud Platform |
| **Model Persistence** | Pickle |

## 🚀 How to Use

### **Live Demo**
Visit the deployed application: [https://uci-credit-card-default-prediction.onrender.com](https://uci-credit-card-default-prediction.onrender.com)

1. Click "Start Prediction"
2. Enter customer information (24 fields including demographics, payment history, bill amounts)
3. Get instant prediction result

### **Local Setup**
```bash
# Clone repository
git clone https://github.com/coderved63/UCI_CREDIT_CARD_DEFAULT_PREDICTION.git
cd ucicreditcard

# Install dependencies  
pip install -r requirements.txt

# Run the application
python application.py

# Access at: http://localhost:5000
```

### **Training Pipeline**
```bash
# To retrain the model with new data
python src/pipeline/train_pipeline.py
```

## 💡 Key Features

### **Feature Engineering Process**
- **Original Features**: 24 (demographics, payment history, bill/payment amounts)  
- **Engineered Features**: 17 additional features created
- **Key Engineered Features**:
  - Credit utilization ratio (bill_amount/credit_limit)
  - Payment efficiency ratios for each month
  - Average payment status and reliability scores
  - Bill amount and payment trends over time

### **Data Processing**
- **Handled Missing Values**: Imputation strategies for different feature types
- **Scaling**: StandardScaler for numerical features
- **Class Imbalance**: Addressed 22% default rate using ROC-AUC metric

## 📈 Results

### **Model Performance**
- **Best Algorithm**: XGBoost
- **ROC-AUC Score**: 0.814 (81.4%)
- **Accuracy**: 82.3%
- **Cross-Validation**: GridSearchCV with 2-fold CV

### **Key Findings**
- Payment history is the strongest predictor of default
- Credit utilization >80% significantly increases default risk
- Engineered features improved model performance by ~15%

## 🔗 Links

- **🌐 Live Application**: [https://uci-credit-card-default-prediction.onrender.com](https://uci-credit-card-default-prediction.onrender.com)
- **📊 Dataset Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **🐙 GitHub Repository**: [https://github.com/coderved63/UCI_CREDIT_CARD_DEFAULT_PREDICTION](https://github.com/coderved63/UCI_CREDIT_CARD_DEFAULT_PREDICTION)

---

## 🏆 Project Summary

This project demonstrates **end-to-end machine learning engineering**:
- ✅ **Data Science**: EDA, feature engineering, model selection
- ✅ **Machine Learning**: Multiple algorithms, hyperparameter tuning, performance evaluation  
- ✅ **Software Engineering**: Modular code structure, error handling, logging
- ✅ **Web Development**: Flask application with responsive UI
- ✅ **DevOps**: Cloud deployment, version control, production optimization
