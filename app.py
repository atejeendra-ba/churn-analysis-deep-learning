import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Safe relative paths for model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.h5'))
    with open(os.path.join(BASE_DIR, 'label_encoder_gender.pkl'), 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# Streamlit UI Setup
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title('📊 Customer Churn Prediction')

# Input Fields
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 40)
    tenure = st.slider('Tenure', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 2)

with col2:
    credit_score = st.number_input('Credit Score', value=600)
    balance = st.number_input('Balance', value=60000.0)
    estimated_salary = st.number_input('Estimated Salary', value=50000.0)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prediction Action
if st.button('Predict Churn Probability', type="primary"):
    # 1. Prepare base DataFrame (Matches training structure)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],  # Fixed typo: transform
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # 2. One-Hot Encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, 
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # 3. Concatenate Features
    input_df = pd.concat([input_data, geo_encoded_df], axis=1)

    # 4. Scale Data & Predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    proba = float(prediction[0][0])

    st.divider()
    st.metric(label="Churn Probability", value=f"{proba:.1%}")

    if proba > 0.5:
        st.error("⚠️ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")
        