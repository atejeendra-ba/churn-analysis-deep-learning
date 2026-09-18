# 📊 Financial Analytics & Churn Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An **end-to-end Deep Learning suite** designed to solve two core banking operations using **Artificial Neural Networks (ANN)** built with **TensorFlow / Keras**:

1. **Customer Churn Prediction (Binary Classification):** Classifies whether a customer is likely to leave the bank (`Exited = 1`) or remain active (`Exited = 0`).
2. **Customer Salary Estimation (Continuous Regression):** Predicts exact customer financial valuation (`EstimatedSalary`) to enable automated credit scoring and wealth management targeting.

The repository includes complete **data preprocessing**, **categorical encoding**, **feature scaling**, **model training with callbacks**, and interactive **Streamlit web applications** for both pipelines.

---

## 📌 Project Overview

### **Dual-Pipeline Scope**

* **Classification Engine:** Predicts retention risk using a `Sigmoid` output layer and `Binary Cross-Entropy` loss to allow proactive customer retention.
* **Regression Engine:** Solves the missing income problem by predicting exact numerical salaries using a linear output layer and `Mean Absolute Error (MAE)` loss.
* **Shared Preprocessing:** Enforces standardized categorical encoding (`LabelEncoder`, `OneHotEncoder`) and feature scaling (`StandardScaler`) serialized via `pickle` to prevent data drift during live inference.
* **Monitoring & Callbacks:** Tracks real-time training and validation metrics using **TensorBoard** and controls convergence via **EarlyStopping**.

---

## 🛠️ Key Features & Tech Stack

| Domain | Tools / Technologies |
| :--- | :--- |
| **Language** | **Python 3.10+** |
| **Deep Learning** | **TensorFlow 2.15.0**, **Keras** |
| **Data Processing & ML** | **Pandas**, **NumPy**, **Scikit-Learn** (`StandardScaler`, `OneHotEncoder`, `LabelEncoder`) |
| **Monitoring & Callbacks** | **TensorBoard**, **EarlyStopping** |
| **Deployment & UI** | **Streamlit** |
| **Development Environment** | **VS Code**, **Jupyter Notebooks (`.ipynb`)**, **Python `.venv`** |

---

## ⚙️ Project Architecture & Comparison

| Feature | Churn Prediction (`app.py`) | Salary Estimation (`streamlit_regression.py`) |
| :--- | :--- | :--- |
| **Task Type** | Binary Classification | Continuous Regression |
| **Target Variable** | `Exited` (0 or 1) | `EstimatedSalary` ($) |
| **Output Activation** | `Sigmoid` | Linear / None |
| **Loss Function** | `binary_crossentropy` | `mean_absolute_error` (MAE) |
| **Primary Metric** | Accuracy / Loss | Mean Absolute Error (MAE) |

---

## 📁 Repository Structure

```text
├── logs/                         # TensorBoard execution logs (Classification)
├── regressionlogs/               # TensorBoard execution logs (Regression)
├── .venv/                        # Virtual environment directory
├── Churn_Modelling.csv           # Primary bank dataset
├── experiments.ipynb             # Exploratory Data Analysis & Churn Model training
├── salaryregression.ipynb        # Data Preprocessing & Salary Model training
├── app.py                        # Streamlit web app (Churn Classification)
├── streamlit_regression.py       # Streamlit web app (Salary Regression)
├── model.h5                      # Trained TensorFlow model weights (Classification)
├── regression_model.h5           # Trained TensorFlow model weights (Regression)
├── scaler.pkl                    # Fitted StandardScaler object
├── label_encoder_gender.pkl     # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl       # Fitted OneHotEncoder for Geography
├── requirements.txt             # Environment dependency manifest
└── README.md                     # Project documentation
```
---

## 🚀 Getting Started

### **1. Clone the Repository**
```bash
git clone [https://github.com/atejeendra-ba/Churn-Analysis-Deep-Learning.git](https://github.com/atejeendra-ba/Churn-Analysis-Deep-Learning.git)
cd Churn-Analysis-Deep-Learning
```
### **2. Set Up Virtual Environment**

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```
#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **4. Launch the Churn Classification App**
```bash
streamlit run app.py
```
### **5. Launch the Salary Regression App**
```bash
streamlit run streamlit_regression.py
```
### **6. 📈 Monitoring with TensorBoard**
To view real-time training and validation metrics saved during model execution, run:
```bash
tensorboard --logdir logs/fit
```
---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/https://github.com/atejeendra-ba/Churn-Analysis-Deep-Learning/issues).

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👤 Author

**A Tejeendra**
* **GitHub:** [@ATejeendra](https://github.com/atejeendra-ba)
* **LinkedIn:** [A Tejeendra](https://www.linkedin.com/in/a-tejeendra/)

---

## 🙏 Acknowledgements

* [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs)
* [Streamlit Documentation](https://docs.streamlit.io/)
* Dataset sourced from [Kaggle](https://www.kaggle.com/) (`Churn_Modelling.csv`)
