# 🦴 Vertebral Column Condition Classification

This project explores various machine learning techniques to classify patients' spinal conditions based on six biomechanical measurements. The dataset is sourced from the UCI Machine Learning Repository and includes three diagnostic categories:

- `Normal`
- `Disk Hernia`
- `Spondylolisthesis`

The goal is to build a robust classification model capable of accurately predicting a patient's condition using these measurements. Multiple models are evaluated, tuned, and compared, with a final ensemble chosen for deployment in a Streamlit web app.



## 📌 Features

- End-to-end ML pipeline: EDA, preprocessing, training, evaluation
- Handling class imbalance with SMOTE
- Hyperparameter tuning via GridSearchCV
- Ensemble methods: Hard and Soft Voting
- Final model deployment via Streamlit


## 📊 Dataset

- **Source**: UCI Machine Learning Repository
- **Samples**: 310
- **Features**: 6 continuous biomechanical attributes
- **Target Classes**: Normal, Disk Hernia, Spondylolisthesis
- **No missing values**

[UCI Dataset Link](https://archive.ics.uci.edu/ml/datasets/Vertebral+Column)


## 🤖 Models Trained

- Logistic Regression
- Random Forest
- XGBoost (untuned and tuned)
- K-Nearest Neighbors (tuned)
- Support Vector Machine (with and without SMOTE)
- Hard Voting Ensemble ✅ (Final deployed model)
- Soft Voting Ensemble

## ✅ Final Results (Test Accuracy)

| Model                   | Accuracy |
|------------------------|----------|
| Logistic Regression     | 79.03%   |
| Random Forest           | 85.48%   |
| XGBoost (untuned)       | 82.26%   |
| XGBoost (tuned)         | 79.03%   |
| KNN (tuned)             | 79.03%   |
| SVM (tuned)             | 87.10%   |
| SVM + SMOTE             | 88.71%   |
| Soft Voting Ensemble    | 88.71%   |
| **Hard Voting Ensemble**| **90.32%** ✅


