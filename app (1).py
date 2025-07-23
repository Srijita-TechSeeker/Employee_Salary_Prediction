
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model, scaler = joblib.load("salary_predictor.pkl")

st.title("💼 Employee Salary Predictor")
st.markdown("Enter employee details to predict their salary")

# Input fields
age = st.number_input("Age", min_value=18, max_value=70)
gender = st.selectbox("Gender", ['Male', 'Female'])
education = st.selectbox("Education Level", ['Bachelor', 'Master', 'PhD'])
job_title = st.selectbox("Job Title", ['Engineer', 'Manager', 'Analyst', 'Consultant'])
experience = st.number_input("Years of Experience", min_value=0, max_value=40)
country = st.selectbox("Country", ['United States', 'India', 'Germany', 'Canada'])
race = st.selectbox("Race", ['White', 'Asian', 'Black', 'Hispanic'])

# Label Encoding mappings (must match training)
gender_map = {'Male': 1, 'Female': 0}
education_map = {'Bachelor': 0, 'Master': 1, 'PhD': 2}
job_map = {'Engineer': 0, 'Manager': 1, 'Analyst': 2, 'Consultant': 3}
country_map = {'United States': 3, 'India': 1, 'Germany': 0, 'Canada': 2}
race_map = {'White': 3, 'Asian': 0, 'Black': 1, 'Hispanic': 2}

input_df = pd.DataFrame([[
    age,
    gender_map[gender],
    education_map[education],
    job_map[job_title],
    experience,
    country_map[country],
    race_map[race]
]], columns=['Age', 'Gender', 'Education Level', 'Job Title',
             'Years of Experience', 'Country', 'Race'])

# Predict and display result
if st.button("Predict Salary"):
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)
    st.success(f"💰 Predicted Salary: ₹{int(prediction[0]):,}")
