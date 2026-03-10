import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# -------------------- Page Config --------------------

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a bank customer is likely to churn based on their account information.")

# -------------------- Load Model --------------------

model = tf.keras.models.load_model('D:\Deep_Learning_project\ANN_Classification\models\model.h5')

with open(r'D:\Deep_Learning_project\ANN_Classification\models\Lable_encoder.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open(r'D:\Deep_Learning_project\ANN_Classification\models\One_Hot_Encoder.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(r'D:\Deep_Learning_project\ANN_Classification\models\standart_scalar.pkl','rb') as file:
    scaler = pickle.load(file)

# -------------------- Sidebar Inputs --------------------

st.sidebar.header("Customer Information")

geography = st.sidebar.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)

age = st.sidebar.slider("Age", 18, 92)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, step=1)
balance = st.sidebar.number_input("Balance", min_value=0.0)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0)

tenure = st.sidebar.slider("Tenure (Years)", 0, 10)
num_of_products = st.sidebar.slider("Number of Products", 1, 4)

has_cr_card = st.sidebar.selectbox("Has Credit Card", [0,1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0,1])

predict_btn = st.sidebar.button("Predict Churn")

# -------------------- Prediction --------------------

if predict_btn:

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prediction_proba:.2%}")

    with col2:
        if prediction_proba > 0.5:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")