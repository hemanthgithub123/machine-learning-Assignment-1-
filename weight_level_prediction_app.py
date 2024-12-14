from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
try:
    with open('gradient_boosting_model.pkl', 'rb') as model_file:
        gb_model = pickle.load(model_file)
    with open('label_encoder.pkl', 'rb') as label_file:
        label_encoder = pickle.load(label_file)
    with open('column_transformer.pkl', 'rb') as transformer_file:
        column_transformer = pickle.load(transformer_file)
except FileNotFoundError as e:
    st.error("Required files for the app to work are missing: " + str(e))
    st.stop()

# Set the page configuration
st.set_page_config(
    page_title="Body-Weight Level Prediction App",
    page_icon="🩺",
    layout="wide"
)

# App Title
st.title("Body-Weight Level Prediction App")
st.markdown("""
    **Provide us your data to predict your Body-Weight level.**
    Taking a moment to know your weight level is a positive step toward a healthier, more informed lifestyle.
""")

# Layout: Input columns
left_col, right_col = st.columns([2, 1])

# User Input: Numerical Features
with left_col:
    age = st.slider("Age", min_value=10, max_value=90, value=25, step=1)
    height = st.slider("Height (meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
    weight = st.slider("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
    fcvc = st.slider("Frequency of Vegetable Consumption (0-3)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    ncp = st.slider("Number of Main Meals (1-4)", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
    ch2o = st.slider("Daily Water Consumption (liters)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    faf = st.slider("Physical Activity Frequency (0-5)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    tue = st.slider("Time Using Technology (hours/day)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

# User Input: Categorical Features
with right_col:
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
    favc = st.selectbox("Frequent High-Calorie Food Consumption", ["yes", "no"])
    caec = st.selectbox("Eating Between Meals", ["no", "Sometimes", "Frequently", "Always"])
    scc = st.selectbox("Monitor Calorie Intake", ["yes", "no"])
    mtrans = st.selectbox("Mode of Transportation", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
    smoke = st.selectbox("Do You Smoke?", ["no", "yes"])
    calc = st.selectbox("Alcohol Consumption Frequency", ["no", "Sometimes", "Frequently", "Always"])

# Prepare the input data
input_data = [[gender, family_history, favc, caec, smoke, calc, scc, mtrans,
               age, height, weight, fcvc, ncp, ch2o, faf, tue]]

input_df = pd.DataFrame(input_data, columns=[
    'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 
    'CALC', 'SCC', 'MTRANS', 'Age', 'Height', 'Weight', 'FCVC', 
    'NCP', 'CH2O', 'FAF', 'TUE'])

# Transform data
try:
    transformed_data = column_transformer.transform(input_df)
except Exception as e:
    st.error("Error transforming the input data: " + str(e))
    st.stop()

# Prediction
if st.button("Predict"):
    if transformed_data.shape[1] == 31:  # Ensure transformed data has the right number of features
        prediction = gb_model.predict(transformed_data)
        obesity_class = label_encoder.inverse_transform(prediction)
        st.success(f"Predicted Body-Weight Level: {obesity_class[0]}")
    else:
        st.error("Error: Transformed data does not match expected dimensions.")
