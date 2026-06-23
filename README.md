# 🩺 Diabetes Risk Prediction Using Machine Learning

## 👨🏽‍💻 Author
**Olalemi Olaoluwakintan Emmanuel**  
**Role:** Data Scientist

---

# 📌 Project Overview

This project develops a machine learning solution for predicting diabetes risk using patient health records. The objective is to assist healthcare providers with early identification of high-risk individuals, enabling timely intervention and preventive care.

The project follows a complete data science workflow:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Baseline Model Development
- Model Optimization
- Model Evaluation
- Clinical Interpretation of Results

---

# 🎯 Business Problem

Diabetes is a chronic disease that can lead to severe health complications if not detected early.

Healthcare organizations require predictive systems capable of:

- Identifying patients at risk
- Supporting preventive healthcare strategies
- Reducing delayed diagnosis
- Assisting clinicians during screening

This project builds a predictive model that classifies patients as:

- **0 → No Diabetes**
- **1 → Has Diabetes**

---

# 📊 Dataset Information

### Dataset Size

| Attribute | Value |
|------------|--------|
| Total Records | 768 |
| Total Features | 8 |
| Target Column | 1 |
| Total Columns | 9 |

### Features

| Feature |
|----------|
| Pregnancies |
| Glucose |
| BloodPressure |
| SkinThickness |
| Insulin |
| BMI |
| DiabetesPedigreeFunction |
| Age |

### Target Variable

| Value | Meaning |
|---------|---------|
| 0 | No Diabetes |
| 1 | Has Diabetes |

---

# 📈 Target Distribution

| Class | Count | Percentage |
|---------|---------|---------|
| No Diabetes (0) | 500 | 65.10% |
| Diabetes (1) | 268 | 34.90% |

### Observation

The dataset is moderately imbalanced.

This means machine learning models may naturally favor predicting the majority class (healthy patients) unless corrective techniques are applied.

---

# 🧹 Data Cleaning & Preprocessing

## Missing Value Analysis

| Feature | Missing Values |
|-----------|---------------|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

### Missing Value Strategy

Median Imputation was applied to all columns containing missing values.

### Why Median?

- Robust against outliers
- Suitable for skewed medical variables
- Preserves dataset size
- Prevents loss of valuable patient records

### Result

✅ All missing values successfully removed.

---

# 📊 Exploratory Data Analysis (EDA)

## Figure 1: Diabetes Distribution

### Findings

- 500 healthy patients
- 268 diabetic patients

### Insight

The dataset contains more healthy individuals than diabetic individuals.

This imbalance can negatively affect model performance if not addressed.

---

## Figure 2: Glucose Distribution by Outcome

### Findings

| Group | Median Glucose |
|---------|---------------|
| Non-Diabetic | 107.5 mg/dL |
| Diabetic | 140.0 mg/dL |

### Insight

Glucose levels show strong separation between diabetic and non-diabetic patients, making it one of the strongest predictors of diabetes.

---

## Figure 3: Correlation Heatmap

### Strongest Relationships with Diabetes

| Feature | Correlation |
|------------|------------|
| Glucose | 0.49 |
| BMI | 0.31 |
| Age | 0.24 |

### Insight

Higher glucose levels, elevated BMI, and increasing age are associated with greater diabetes risk.

---

# 🤖 Machine Learning Development

## Data Splitting

Training/Test Split:

- Training Set: 80%
- Testing Set: 20%

### Stratification

Stratified sampling was applied to preserve the original class distribution in both training and testing datasets.

---

# 📍 Baseline Model

## Algorithm

Logistic Regression

### Performance

| Metric | Score |
|----------|---------|
| Accuracy | 0.7013 |
| Precision | 0.5870 |
| Recall | 0.5000 |
| F1 Score | 0.5400 |

---

## Baseline Confusion Matrix

| | Predicted Healthy | Predicted Diabetic |
|---|---|---|
| Actual Healthy | 81 | 19 |
| Actual Diabetic | 27 | 27 |

### Clinical Interpretation

- True Negatives: 81
- False Positives: 19
- False Negatives: 27
- True Positives: 27

### Medical Assessment

The baseline model correctly identified 70.13% of cases overall.

However, it detected only 50% of actual diabetic patients.

This means 27 diabetic individuals were missed and incorrectly classified as healthy.

Since healthcare screening prioritizes identifying as many sick patients as possible, the baseline model is not suitable for deployment without further optimization.

---

# 🚀 Model Optimization

To improve predictive performance, the following techniques were implemented:

## Feature Scaling

**StandardScaler**

Benefits:

- Normalizes feature ranges
- Improves coefficient learning
- Enhances model stability

---

## Class Balancing

**SMOTE (Synthetic Minority Oversampling Technique)**

Benefits:

- Generates synthetic diabetic samples
- Reduces class imbalance
- Improves minority class detection

---

## Hyperparameter Tuning

**GridSearchCV**

### Parameters Tuned

- Regularization Strength (C)
- Solver Type
- Class Weight

### Best Parameters

```python
{
 'model__C': 0.001,
 'model__class_weight': None,
 'model__solver': 'liblinear'
}
```

---

# 🏆 Optimized Model Results

| Metric | Score |
|----------|---------|
| Accuracy | 0.6883 |
| Precision | 0.5395 |
| Recall | 0.7593 |
| F1 Score | 0.6308 |
| ROC-AUC | 0.7869 |

---

# 📊 Optimized Confusion Matrix

| | Predicted Healthy | Predicted Diabetic |
|---|---|---|
| Actual Healthy | 65 | 35 |
| Actual Diabetic | 13 | 41 |

### Breakdown

| Metric | Value |
|-----------|---------|
| True Negative (TN) | 65 |
| False Positive (FP) | 35 |
| False Negative (FN) | 13 |
| True Positive (TP) | 41 |

---

# 🏥 Medical Context Interpretation

### The Optimized Model Shows:

- **Moderate Accuracy:** The model correctly classifies 68.83% of patients.

- **Clinical Advantage:** Recall improved significantly from **50.00%** to **75.93%**.

- **Reduced Diagnostic Risk:** Only **13 diabetic patients** were missed compared to **27** in the baseline model.

- **Acceptable Trade-Off:** False positives increased, meaning more healthy patients may be referred for additional testing.

- **Healthcare Perspective:** Additional testing is generally safer than failing to detect a diabetic patient.

### Verdict

The optimized model prioritizes patient safety by detecting substantially more diabetic patients.

Although overall accuracy decreased slightly, the large improvement in Recall makes the optimized model significantly more suitable for healthcare screening applications.

---

# 🔍 Feature Importance

## Top Predictors of Diabetes

| Rank | Feature | Importance |
|---------|---------|---------|
| 1 | Glucose | 0.156972 |
| 2 | BMI | 0.102375 |
| 3 | SkinThickness | 0.062338 |
| 4 | Insulin | 0.058959 |
| 5 | Age | 0.058537 |
| 6 | Pregnancies | 0.053139 |
| 7 | BloodPressure | 0.045498 |
| 8 | DiabetesPedigreeFunction | 0.043508 |

### Key Finding

Glucose emerged as the strongest predictor of diabetes risk within this dataset.

---

# 🔄 Cross-Validation Results

### 5-Fold Cross Validation F1 Scores

```text
0.6560
0.6250
0.6400
0.7258
0.6829
```

### Mean F1 Score

```text
0.6659
```

### Interpretation

The model demonstrates relatively stable performance across different data partitions, indicating reasonable generalization capability.

---

# 📌 Model Comparison

| Metric | Baseline | Optimized |
|----------|----------|----------|
| Accuracy | 0.7013 | 0.6883 |
| Precision | 0.5870 | 0.5395 |
| Recall | 0.5000 | 0.7593 |
| F1 Score | 0.5400 | 0.6308 |

### Improvement Summary

✅ Recall improved by **25.93 percentage points**

✅ F1 Score improved by **16.81 percentage points**

✅ False Negatives reduced from **27 → 13**

✅ Better suitability for healthcare screening

---

# 🏁 Conclusion

This project successfully developed a machine learning pipeline capable of predicting diabetes risk using patient health indicators.

The optimized model achieved substantial improvements in diabetic patient detection through:

- Data preprocessing
- Feature scaling
- SMOTE oversampling
- Hyperparameter optimization

While accuracy decreased slightly, the significant improvement in Recall and reduction in missed diabetic cases make the optimized model more appropriate for healthcare screening environments.

---

# 🚀 Future Improvements

- Random Forest Classifier
- XGBoost Classifier
- Ensemble Learning
- Feature Engineering
- Threshold Optimization
- Additional Clinical Variables
- Explainable AI (SHAP/LIME)

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Google Colab

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

---

⭐ If you found this project useful, consider giving it a star.
