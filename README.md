# Detecting Parkinson’s Disease using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Parkinson’s Disease (PD) is a progressive neurodegenerative disorder that affects movement, speech, and coordination due to the loss of dopamine-producing neurons in the brain. Early and accurate detection can significantly improve symptom management and quality of life.

This project applies **machine learning techniques** to detect Parkinson’s Disease using **biomedical voice measurements**, based on the **UCI Parkinson’s dataset**.

---

## 📖 Problem Statement

Parkinson’s Disease is chronic, progressive, and currently incurable. Traditional diagnosis relies heavily on clinical observation, which may be subjective and may fail to detect early-stage symptoms.

**Objective:**  
To build and evaluate machine learning models that can accurately classify whether an individual has Parkinson’s Disease based on voice-related biomedical features.

---

## 🧠 About Parkinson’s Disease

Parkinson’s Disease primarily affects dopamine-producing neurons in the **substantia nigra** region of the brain. Common symptoms include tremors, rigidity, bradykinesia, gait imbalance, and speech impairment.

Recent research emphasizes identifying **biomarkers** that enable early diagnosis using computational and machine learning approaches.

---

## 📊 Dataset Description

- **Source:** UCI Machine Learning Repository  
- **Total Records:** 195  
- **Features:** 23 biomedical voice measurements  
- **Target Variable:** `status`  
  - `1` → Parkinson’s Disease  
  - `0` → Healthy  

Key features include:
- Fundamental frequency measures (MDVP:Fo, MDVP:Fhi, MDVP:Flo)
- Jitter and shimmer parameters
- Harmonics-to-noise ratio (HNR)
- Nonlinear dynamical complexity measures

---

## 🔍 Exploratory Data Analysis (EDA)

### 🔹 Pairwise Feature Relationships
![Pairplot](images/pairplot.png)

### 🔹 Feature Distributions
![Feature Distribution](images/feature_distribution.png)

### 🔹 Correlation Heatmap
Strong correlations between several voice features indicate their relevance for classification.
![Correlation Heatmap](images/correlation_heatmap.png)

---

## ⚙️ Methodology

1. Loaded the dataset and removed non-informative identifiers  
2. Separated features and labels (`status`)  
3. Normalized features using **MinMaxScaler**  
4. Split data into training and testing sets (80/20)  
5. Trained and evaluated multiple machine learning models  

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Gradient Boosting  
- XGBoost  
- Voting Classifier (Ensemble)

---

## 📈 Model Performance Comparison

![Model Comparison](images/model_comparison.png)

| Model | Accuracy |
|------|----------|
| Logistic Regression | ~85% |
| Decision Tree | ~97% |
| Random Forest | ~99% |
| SVM | ~98% |
| KNN | ~99% |
| XGBoost | ~92% |
| Voting Classifier | ~90% |

---

## 🧪 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### 🔹 Confusion Matrix (Best Model)
![Confusion Matrix](images/confusion_matrix.png)

---

## 🏆 Key Results

- Achieved **up to 99% accuracy** using ensemble and tree-based models  
- Demonstrated strong predictive power of voice-based biomedical features  
- Validated the effectiveness of machine learning for early Parkinson’s detection  

---

## 🛠️ Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## 🚀 How to Run the Project

```bash
git clone https://github.com/paanchuk9080/Detecting-Parkinsons-Disease.git
cd Detecting-Parkinsons-Disease
pip install -r requirements.txt
jupyter notebook
