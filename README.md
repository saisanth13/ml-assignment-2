# Breast Cancer Classification using Machine Learning

## Problem Statement
The objective of this project is to build and evaluate multiple machine learning
classification models to predict whether a breast tumor is **benign** or
**malignant** based on diagnostic features. The project compares the performance
of different classifiers using standard evaluation metrics and deploys the models
using a Streamlit application.

---

## Dataset Description
The **Breast Cancer Wisconsin (Diagnostic)** dataset is used for this project.
It is a public dataset widely used for binary classification tasks.

- Number of samples: 569
- Number of features: 30 (all numerical)
- Target variable: `diagnosis`
  - `B` → Benign
  - `M` → Malignant
- Source: UCI Machine Learning Repository

The dataset satisfies the assignment requirements of having at least 500 samples
and a minimum of 12 features.

---

## Preprocessing Steps
The following preprocessing steps were applied:

- Removed completely empty columns
- Encoded target labels (`B → 0`, `M → 1`)
- Handled missing values using **median imputation**
- Standardized features using **StandardScaler**

The same preprocessing pipeline is used during both training and inference.

---

## Models Implemented
The following classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

All trained models are saved in the `model/` directory.

---

## Evaluation Metrics
Each model was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC (ROC)
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC | MCC |
|------|----------|-----------|--------|----------|-----|-----|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 | 0.9245 |
| Decision Tree | 0.9123 | 1.0000 | 0.7619 | 0.8649 | 0.9688 | 0.8179 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9823 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9231 | 0.8571 | 0.8889 | 0.9891 | 0.8292 |
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9950 | 0.9442 |
| XGBoost | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9940 | 0.9442 |

---

## Observations
- Logistic Regression achieved strong performance with high accuracy (96.49%)
  and the highest AUC (0.996), indicating that the dataset is close to being
  linearly separable.

- KNN performed well after feature scaling, but its recall was slightly lower
  compared to Logistic Regression and ensemble models, showing sensitivity to
  local neighborhood structure.

- Decision Tree achieved perfect precision but relatively lower recall, which
  indicates that the constrained tree depth helped reduce overfitting but led
  to missed positive cases.

- Naive Bayes showed reasonable performance but lower MCC compared to other
  models, suggesting weaker balanced classification despite good AUC.

- Random Forest and XGBoost delivered the best overall performance, achieving
  the highest accuracy (97.37%) and MCC (0.944), demonstrating the effectiveness
  of ensemble methods in reducing variance and improving generalization.

- MCC proved to be a reliable metric for comparing models, as it balances
  precision and recall and provides a robust measure even when class
  distributions are not perfectly balanced.

---

## Streamlit Application
A Streamlit application was developed to:
- Upload CSV files
- Select a classification model
- View predictions, probabilities, evaluation metrics, and confusion matrix
- Download prediction results

### Running the App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
