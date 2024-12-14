import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('gradient_boosting_model.pkl', 'rb') as model_file:
    gb_model = pickle.load(model_file)

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as label_file:
    label_encoder = pickle.load(label_file)

# Load the ColumnTransformer
with open('column_transformer.pkl', 'rb') as transformer_file:
    column_transformer = pickle.load(transformer_file)

# Set the page configuration
st.set_page_config(
    page_title="Body-Weight Level Prediction App",
    page_icon="🩺",
    layout="wide"
)

st.title("Body-Weight Level Prediction App")
st.subheader("Provide us your data to predict your Body-Weight level :)")
st.subheader("Taking a moment to know your obesity level is a positive step toward a healthier, more informed lifestyle.")

# Defining layouts for app with columns
left_col, right_col = st.columns([2, 1])

# Get user input
# Left column for inputs
with left_col:
    col1, col2 = st.columns([1, 1])
    
    # User inputs via sliders for numerical features
    with col1:
        age = st.slider("Age", min_value=10, max_value=90, value=25, step=1)
        height = st.slider("Height (in meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
        weight = st.slider("Weight (in kg)", min_value=30, max_value=200, value=70, step=1)
        fcvc = st.slider("Frequency of consumption of vegetables (0.0-3.0)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    with col2:
        ncp = st.slider("Number of main meals (1.0-4.0)", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
        ch2o = st.slider("Daily water consumption (1.0-3.0 liters)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        faf = st.slider("Physical activity frequency (0.0-5.0)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        tue = st.slider("Time spent on technology devices (0.0-2.0)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

# Right column for categorical features
with right_col:
    col3, col4 = st.columns([1, 1])
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_history_with_overweight = st.selectbox("Family history with overweight", ["yes", "no"])
        favc = st.selectbox("Frequent consumption of high caloric food", ["yes", "no"])
        caec = st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"])
    with col4:
        scc = st.selectbox("Do you monitor your calorie intake", ["yes", "no"])
        mtrans = st.selectbox("Transportation", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
        smoke = st.selectbox("Do you Smoke?", ["no", "yes"])
        calc = st.selectbox("How often do you drink alcohol?", ["no", "Sometimes", "Frequently", "Always"])

nl, center, nr = st.columns([2, 1, 2])
with center:
    st.subheader("Prediction")
    # Create a dataframe from the user inputs
    input_data = [[gender, family_history_with_overweight, favc, caec, smoke, calc, scc, mtrans,
                   age, height, weight, fcvc, ncp, ch2o, faf, tue]]
    # Converting into dataframe because in the model we did column transform for dataframe
    input_df = pd.DataFrame(input_data, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
                                                 'CALC', 'SCC', 'MTRANS', 'Age', 'Height', 'Weight', 'FCVC',
                                                 'NCP', 'CH2O', 'FAF', 'TUE'])
    # Apply the ColumnTransformer
    transformed_data = column_transformer.transform(input_df)
    # Prediction
    if st.button("Predict"):
        if transformed_data.shape[1] == 31:  # To check if the input values got encoded correctly or not
            prediction = gb_model.predict(transformed_data)
            obesity_class = label_encoder.inverse_transform(prediction)
            st.write(f"Predicted Body-Weight Level: {obesity_class[0]}")
        else:  # Returning the error message to user
            st.write("Error: Transformed data does not have 31 features.")
            st.write(f"Transformed data shape: {transformed_data.shape}")
    # Adding some spacing for better visual alignment
    st.markdown("<br><br>", unsafe_allow_html=True)
