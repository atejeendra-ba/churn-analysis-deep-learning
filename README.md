# 📊 Churn Analysis & Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An **end-to-end Machine Learning and Deep Learning pipeline** designed to predict bank customer churn. This project utilizes an **Artificial Neural Network (ANN)** built with **TensorFlow / Keras** to classify whether a customer is likely to leave the bank (**`Exited = 1`**) or remain active (**`Exited = 0`**).

The repository includes complete **data preprocessing**, **categorical encoding**, **feature scaling**, **model training with callbacks**, and an interactive **Streamlit web application** for real-time inference.

---

## 📌 Project Overview

Customer retention is critical for banking institutions. Predicting customer churn allows banks to take **proactive, data-driven measures** to retain high-risk accounts before they exit.

### **Lifecycle Coverage**

* **Data Ingestion & Cleaning:** Ingested and scrubbed customer demographic and financial indicators from `Churn_Modelling.csv`.
* **Preprocessing Pipeline:** Encoded categorical features and scaled numerical distributions using **Scikit-Learn**, serializing objects via **`pickle`** for production reuse.
* **Deep Learning Training:** Built, optimized, and monitored a multi-layer Neural Network using **Early Stopping** and **TensorBoard**.
* **Interactive Deployment:** Shipped an interactive web interface using **Streamlit** to serve real-time predictions.

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

## ⚙️ Project Architecture & Workflow

### 1️⃣ Data Preprocessing & Feature Engineering
* **Feature Selection:** Dropped non-informative columns (`RowNumber`, `CustomerId`, `Surname`).
* **Categorical Encoding:**
  * **Gender:** Transformed using **`LabelEncoder`** into binary indicators.
  * **Geography:** One-hot encoded using **`OneHotEncoder`** into country flags (*France, Germany, Spain*).
* **Feature Scaling:** Standardized numerical attributes using **`StandardScaler`** to improve convergence rates during backpropagation.
* **Artifact Serialization:** Saved **`label_encoder_gender.pkl`**, **`onehot_encoder_geo.pkl`**, and **`scaler.pkl`** to prevent data drift during live inference.

---

### 2️⃣ Neural Network Architecture (ANN)

* **Hidden Layers:** Configured with **`ReLU` activation functions** to capture non-linear relationships.
* **Output Layer:** Single neuron with a **`Sigmoid` activation function** outputting probabilities in the range `[0.0, 1.0]`.
* **Optimization & Loss:** Compiled with the **Adam Optimizer** (`learning_rate=0.01`) and **`binary_crossentropy`** loss.
* **Callback Integration:**
  * **`EarlyStopping`:** Monitored `val_loss` with patience to eliminate overfitting.
  * **`TensorBoard`:** Captured epoch-level training metrics and parameter histograms.

---

### 3️⃣ Model Serialization & Inference Engine
* Saved the best-performing model weights as **`model.h5`**.
* Implemented an isolated prediction pipeline that loads serialized preprocessing artifacts and processes new input feature vectors dynamically.

---

### 4️⃣ Interactive Web Application (`app.py`)
* Built an intuitive user interface featuring **interactive sliders** (*Age, Tenure, Products*) and **dropdowns** (*Geography, Gender, Membership Status*).
* Computes real-time **churn probability percentages** with visual status alerts (**`High Risk`** vs. **`Low Risk`**).

---

## 📁 Repository Structure

```text
├── logs/                      # TensorBoard execution logs
│   └── fit/                   # Epoch metrics & execution timestamps
├── .venv/                     # Virtual environment directory
├── Churn_Modelling.csv        # Primary bank churn dataset
├── experiments.ipynb          # Exploratory Data Analysis (EDA) & ANN training
├── prediction.ipynb           # Model verification & inference testing
├── app.py                     # Streamlit frontend web application
├── model.h5                   # Trained TensorFlow/Keras neural network
├── scaler.pkl                 # Fitted StandardScaler pickle
├── label_encoder_gender.pkl   # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl     # Fitted OneHotEncoder for Geography
├── requirements.txt           # Environment dependency manifest
└── README.md                  # Project documentation
```
---

## 🚀 Getting Started

### **1. Clone the Repository**
```bash
git clone [https://github.com/https://github.com/atejeendra-ba/Churn-Analysis-Deep-Learning.git](https://github.com/https://github.com/atejeendra-ba/Churn-Analysis-Deep-Learning.git)
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
### **4. Launch the Streamlit Web App**
```bash
streamlit run app.py
```
### **5. 📈 Monitoring with TensorBoard**
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
