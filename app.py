import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("iris_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Prediction")

st.write("""
Enter the features of the Iris flower to predict its species.
""")

# User input
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width  = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Predict button
if st.button("Predict"):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(sample)
    st.success(f"Predicted species: {prediction[0]}")
