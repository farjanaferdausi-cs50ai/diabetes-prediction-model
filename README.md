<div align="center">

# 🧬 Diabetes Prediction Model
### *AI-Powered Early Risk Detection using Machine Learning*

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge)

<img src="https://img.shields.io/badge/ROC--AUC-0.8428-00C9A7?style=flat-square&logo=target&logoColor=white"/>
<img src="https://img.shields.io/badge/Models%20Trained-4-FF6F61?style=flat-square"/>
<img src="https://img.shields.io/badge/Dataset-Pima%20Indians%20Diabetes-4B8BBE?style=flat-square"/>

*A clean, end-to-end supervised learning pipeline that predicts diabetes risk from routine medical attributes — built, tuned, and evaluated like a production ML workflow.*

</div>

---

## 🚀 Overview

I built this project to predict whether a patient is likely to have diabetes, using only eight simple, routinely-collected medical measurements. Rather than training a single model, I designed a **comparative ML pipeline** — training four different classifiers, evaluating them with multiple metrics, validating results with cross-validation, and scientifically selecting the best-performing model for deployment.

This isn't just a notebook that runs top-to-bottom — it reflects a real modeling workflow: **handle hidden missing data → engineer a fair train/test split → train multiple candidates → evaluate rigorously → validate → select → persist.**

---

## 🧠 Problem Statement

> **Objective:** Predict the presence of diabetes (`Outcome`: 0 = No, 1 = Yes) using 8 medical diagnostic measurements.

**Dataset:** [Pima Indians Diabetes Database](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
**Samples:** 768 patient records
**Features:** 8 numeric medical attributes
**Target:** Binary classification (`Outcome`)

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes hereditary risk score |
| `Age` | Age in years |

---

## ⚙️ Project Workflow

```
1️⃣  Import Libraries          →  pandas, numpy, scikit-learn, seaborn
2️⃣  Data Loading & EDA        →  Shape, distributions, correlation heatmap
3️⃣  Data Preprocessing        →  Zero-value imputation + stratified split + scaling
4️⃣  Model Training            →  4 classifiers trained in parallel
5️⃣  Model Evaluation          →  Accuracy, Precision, Recall, F1, ROC-AUC
6️⃣  Confusion Matrices        →  Visual error analysis per model
7️⃣  ROC Curve Comparison      →  Discriminative power across models
8️⃣  Feature Importance        →  Random Forest interpretability
9️⃣  Cross-Validation (5-fold) →  Robust, unbiased model comparison
🔟  Best Model Selection      →  Saved as production-ready .pkl artifact
```

### 🔍 Key Engineering Decision: Hidden Missing Data

A critical insight I identified during EDA: columns like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contained **biologically impossible zero values** — these aren't real measurements, they're disguised missing data. I treated these as `NaN` and imputed them using median values computed **only from the training set**, then applied to the test set — preventing data leakage and keeping the evaluation honest.

---

## 🏆 Model Performance

I trained and benchmarked four classification algorithms on identical preprocessed data:

| Model | Accuracy | Precision | Recall | F1-Score | Test ROC-AUC | **5-Fold CV ROC-AUC** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🔵 **Logistic Regression** | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | **0.8428 ± 0.0190** ⭐ |
| 🌲 Random Forest | 0.7727 | 0.7021 | 0.6111 | 0.6535 | 0.8181 | 0.8183 ± 0.0216 |
| 🟣 SVM (RBF Kernel) | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.7964 | 0.8337 ± 0.0220 |
| 🟠 KNN (k=5) | 0.7532 | 0.6600 | 0.6111 | 0.6346 | 0.7899 | 0.7830 ± 0.0383 |

### ✅ Final Model Selection

> **Winner: Logistic Regression** — selected using **5-fold cross-validated ROC-AUC** rather than a single test split, ensuring the choice generalizes and isn't a lucky artifact of one train/test partition.

The final model and its `StandardScaler` were serialized with `joblib` and verified by reloading them and re-running inference on an unseen sample — confirming the saved pipeline is deployment-ready.

---

## 📊 Visual Analysis

This project includes:
- 📈 **Class distribution plot** — visualizing the outcome imbalance (65% non-diabetic / 35% diabetic)
- 🔥 **Correlation heatmap** — feature relationships and multicollinearity check
- 🧩 **Confusion matrices** — per-model true/false positive & negative breakdown
- 📉 **ROC curve comparison** — visual AUC benchmarking across all 4 models
- 🌟 **Feature importance chart** — Random Forest ranking of the most predictive medical attributes

---

## 🛠️ Tech Stack

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-orange?style=flat-square)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

---

## 📁 Repository Structure

```
diabetes-prediction-model/
│
├── Diabetes_Prediction_Model.ipynb   # Full end-to-end notebook
├── diabetes_model.pkl                # Saved best model (Logistic Regression)
├── scaler.pkl                        # Saved StandardScaler
├── requirements.txt                  # Project dependencies
├── LICENSE
└── README.md
```

---

## ▶️ How to Run

```bash
git clone https://github.com/farjanaferdausi-cs50ai/diabetes-prediction-model.git
cd diabetes-prediction-model
pip install -r requirements.txt
jupyter notebook Diabetes_Prediction_Model.ipynb
```

---

## 💡 What I Learned

- Detecting and correctly handling **disguised missing data** (zeros as NaN) is a make-or-break preprocessing step in medical datasets.
- **Cross-validation beats a single test-split score** for trustworthy model selection.
- ROC-AUC is a more reliable comparison metric than accuracy alone on an imbalanced medical dataset.
- Building a reusable, verified inference pipeline (`model.pkl` + `scaler.pkl`) is what separates a notebook exercise from a deployable ML artifact.

---

<div align="center">

## 🖊️ Author

## **Farjana Ferdausi**

*AI/ML Engineering & Data Science* Fellow — Google Cloud Gen AI Academy APAC Edition (Cohort 3)
Agentic AI · RAG · Gemini · ADK · BigQuery MCP · Cloud Run
Former HR Professional (14+ years) at Radisson Blu Dhaka Water Garden, Bangladesh

[![LinkedIn](https://img.shields.io/badge/LinkedIn-farjana--ferdausi-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/farjana-ferdausi/)
[![Medium](https://img.shields.io/badge/Medium-@farjana.rafi1983-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@farjana.rafi1983)

</div>

---

<div align="center">

### ⭐ If this project helped you, consider starring the repository!

</div>
