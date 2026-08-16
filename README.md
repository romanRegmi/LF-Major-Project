# 🏥 Machine Learning in Healthcare: Multi-Disease Prediction System

An end-to-end Machine Learning pipeline and interactive web application to predict risk factors for major health conditions (Heart, Kidney, Liver, Diabetes, Breast Cancer, Pneumonia, and Malaria).

---

## 📌 Features & Supported Datasets

The models are trained using public datasets from Kaggle:

### Tabular Data
* 🫀 **Heart:** [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)
* 🩺 **Kidney:** [Chronic Kidney Disease](https://www.kaggle.com/mansoordaku/ckdisease)
* 🧪 **Liver:** [Indian Liver Patient Records](https://www.kaggle.com/uciml/indian-liver-patient-records)
* 🩸 **Diabetes:** [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
* 🎗️ **Breast Cancer:** [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

### Image Data
* 🫁 **Pneumonia:** [Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* 🔬 **Malaria:** [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

> 💡 **Note:** Detailed information on preprocessed attributes and dropped features after feature selection can be found in `aboutData.txt`.

---

## 🔄 Project Pipeline & Methodology

### 1. Feature Engineering

* **Data Rectification:** Handled non-logical values (e.g., zero-values in blood pressure or glucose readings) and fixed formatting issues (such as trailing tab characters in WBC counts).
  
  ![Faulty Data Example](images/image/faulty_data.png)  
  *Example of faulty zero-value readings in tabular data.*

  ![Typo Clean-up Example](images/image/typo.png)

* **Missing Value Imputation:** Missing fields were imputed using class-conditional medians (splitting by the target output class) to prevent data leakage from healthy vs. diseased distributions.
* **Class Imbalance Handling:** Applied resampling techniques via `imbalanced-learn` on the training set (after `train_test_split`).
* **Feature Encoding & Scaling:** 
  * Categorical/binary features were mapped using numeric replacements.
  * Standard scaling (`StandardScaler`) was applied for distance- and gradient-based models (Logistic Regression, K-NN). Tree-based ensembles (Random Forest) were left unscaled.

![Encoding Example](images/image/encoding.png)

### 2. Feature Selection
For high-dimensional datasets, features were pruned based on **Information Gain** to eliminate redundant predictors.

### 3. Model Training, Evaluation & MLOps
* **Tabular Models:** Evaluated algorithms including Logistic Regression, K-NN, and Random Forest.
* **Image Models:** Fine-tuned convolutional neural networks using **FastAI (ResNet)** architectures.
* **Metric Selection:** Optimized specifically for **Recall** to minimize **False Negatives** (the critical cost of failing to identify a diseased patient).
* **Tracking & Serialization:** Experiments were logged using `MLflow`, and finalized models were serialized with `pickle`.

---

## 🔍 Model Explainability (XAI)

In medical applications, black-box predictions are insufficient. To explain individual prediction drivers (e.g., determining whether glucose or insulin levels contributed more to a diabetic prediction), we integrated **Shapash**.

![Shapash Feature Importance](images/image/shapash.png)

*(Currently implemented for Chronic Kidney Disease and Breast Cancer modules).*

---

## 🛠️ Installation & Setup

### Prerequisites
* **OS:** macOS / Linux / Windows
* **Python:** 3.9+
* **Tools:** `pip`, `pipenv`, `git`

### Quickstart

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/Roman251/LF-Major-Project.git](https://github.com/Roman251/LF-Major-Project.git)
    cd LF-Major-Project
    ```

2. **Set up the virtual environment:**
    ```bash
    pip install --user pipenv
    pipenv shell
    pipenv install --skip-lock
    ```

3. **Launch the Streamlit App:**
    ```bash
    cd app
    streamlit run app.py
    ```
