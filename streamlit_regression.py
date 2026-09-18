import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Define base directory safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# Load trained model and artifacts
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'regression_model.h5'))
    
    with open(os.path.join(BASE_DIR, 'label_encoder_gender.pkl'), 'rb') as f:
        label_encoder_gender = pickle.load(f)
        
    with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
        
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# Page configuration
st.set_page_config(page_title="Salary Regression Predictor", page_icon="💰", layout="centered")

# Header Section
st.title("💰 Customer Salary Prediction")
st.caption("Predict estimated customer salary using an Artificial Neural Network (ANN) regression model.")
st.divider()

# Input Form Section inside a bordered container
with st.container(border=True):
    st.subheader("📋 Enter Customer Details")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demographics**")
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 40)
        tenure = st.slider('Tenure (Years)', 0, 10, 3)

    with col2:
        st.markdown("**Financial Profile**")
        credit_score = st.number_input('Credit Score', value=600, min_value=300, max_value=850)
        balance = st.number_input('Balance ($)', value=60000.0, step=5000.0)
        num_of_products = st.slider('Number of Products', 1, 4, 2)
        
        # Sub-columns for binary status choices
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            has_cr_card = st.selectbox('Has Card', [1, 0])
        with sub_c2:
            is_active_member = st.selectbox('Is Active', [1, 0])
        with sub_c3:
            exited = st.selectbox('Exited', [0, 1])

    st.write("")
    predict_btn = st.button('🚀 Predict Estimated Salary', type="primary", use_container_width=True)

# Output Section
if predict_btn:
    with st.spinner("Calculating neural network predictions..."):
        # 1. Create base DataFrame
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'Exited': [exited]
        })

        # 2. One-Hot Encode Geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded, 
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )

        # 3. Concatenate feature sets
        input_df = pd.concat([input_data, geo_encoded_df], axis=1)

        # 4. Scale inputs using saved StandardScaler
        input_scaled = scaler.transform(input_df)

        # 5. Model Inference
        prediction = model.predict(input_scaled)
        predicted_salary = max(0.0, float(prediction[0][0]))

    # Display stylized results card
    st.write("")
    with st.container(border=True):
        st.subheader("📊 Prediction Results")
        
        res_col1, res_col2 = st.columns([1.5, 1])
        with res_col1:
            st.metric(
                label="ESTIMATED ANNUAL SALARY", 
                value=f"${predicted_salary:,.2f}"
            )
        with res_col2:
            st.metric(
                label="ESTIMATED MONTHLY SALARY", 
                value=f"${(predicted_salary / 12):,.2f}"
            )