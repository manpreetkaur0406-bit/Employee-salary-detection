import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Salary Prediction App")

st.divider


st.write("With this app , you can get estimation for the salaries of the company employee")
years = st.number_input("Enter years of experience", value=1, step=1, min_value=0)
jobrate = st.number_input("Enter job rate", value=3.5, step=0.5, min_value=0.0)

# Combine inputs
X = [years, jobrate]

# Load trained model
model = joblib.load("linearmodel.pkl")

st.divider()

# Button
predict = st.button("Press the button for salary prediction")

st.divider()

if predict:
    st.balloons()

    # Model expects 2D array
    X1 = np.array([X])

    # Prediction
    prediction = model.predict(X1)

    st.write(f"Salary prediction is {prediction[0]}")
