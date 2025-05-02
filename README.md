# 🫀 Heart Disease Prediction using Gaussian Naive Bayes and Logistic Regression

This repository contains the final project for **CS275P: Graphical Models & Statistical Learning** at UC Irvine (Spring 2024). The project addresses the challenge of predicting heart disease presence using machine learning techniques on clinical data from the UCI Heart Disease dataset.

## 📌 Project Overview

Heart disease remains one of the leading causes of mortality worldwide. Early detection is crucial for timely intervention. In this project, we develop two supervised learning models — **Gaussian Naive Bayes** and **Logistic Regression** — and evaluate them on their ability to predict heart disease based on patient clinical features.

Key objectives include:
- Achieving high **accuracy** and **true positive rate (TPR)** for early diagnosis.
- Understanding how **threshold adjustments** affect the balance between sensitivity and specificity.
- Interpreting **model behaviors** through probability distributions and optimization processes.

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Preprocessing**:
  - Removed rows with missing values
  - Standardized features using `StandardScaler`
  - Binarized target variable (0: no disease, 1–4: disease → converted to 1)
  - Split into training and test sets (80/20)

## 🧠 Models Implemented

### 1. Gaussian Naive Bayes
- Assumes features are conditionally independent and follow a normal distribution
- Computes class priors, means, and variances
- Classification is done via log-likelihood and posterior probability comparison

**Results**:
- Accuracy: **91.7%**
- True Positive Rate: **83.3%**
- ROC curve analysis reveals trade-offs when tuning class prior ratio

### 2. Logistic Regression
- Uses a linear model with sigmoid activation to compute probabilities
- Trained using **L-BFGS-B** optimization algorithm with different feature transformations:
  - Linear features
  - Diagonal quadratic features

**Results**:
- **Linear features**: Test Accuracy = **86.7%**, TPR = **83.3%**
- **Quadratic features**: Test Accuracy = **76.7%**, TPR = **70.8%**
- Lowering classification thresholds increases TPR at the cost of overall accuracy

## 📈 Performance Summary

| Model                  | Accuracy | TPR   |
|------------------------|----------|-------|
| Gaussian Naive Bayes   | 91.7%    | 83.3% |
| Logistic Regression (linear) | 86.7%    | 83.3% |
| Logistic Regression (quad.) | 76.7%    | 70.8% |
| Logistic (linear) w/ low threshold | 86.7% | 91.7% |

## 🛠 Tech Stack

- **Python 3.10+**
- `NumPy`, `Pandas` – data manipulation
- `Matplotlib`, `Seaborn` – visualization
- `SciPy` – optimization (L-BFGS-B)
- `Scikit-learn` – model evaluation and preprocessing
- `ucimlrepo` – dataset retrieval

## 👨‍👩‍👧‍👦 Contributors

- **Wenqian**, **Yufei**, **Yu-Han**

## 📄 References

1. Nadikatla, C. et al. *Processes*, 2023  
2. Hossain, S. et al. *BMC Cardiovascular Disorders*, 2024  
3. Majumder, A. B. et al. *Algorithms*, 2023  
4. Saraswathi, R. V. et al. *Advances in Computer Engineering*, 2022  
5. Sudderth, E. – Lecture notes, UC Irvine, 2024

---

## 📌 Notes

This project was developed for educational purposes. The models are not intended for real-world clinical deployment without further validation.
