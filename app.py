# app.py

import streamlit as st
import numpy as np
from joblib import load  # ✅ Changed from pickle to joblib
import json

# Load model and columns
model = load('bangalore_home_prices_model.joblib')
# columns = load('columns.joblib')
with open("columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

# Prepare list of locations (skip first 3: sqft, bath, bhk)
locations = columns[3:]

# Streamlit App UI
st.title("🏡 Bangalore Home Price Prediction")

sqft = st.number_input("Total Square Feet", min_value=300, step=10)
bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5])
bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
location = st.selectbox("Location", sorted(locations))

# Predict button
if st.button("Predict Price"):
    try:
        loc_index = columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    prediction = model.predict([x])[0]
    st.success(f"Estimated Price: ₹ {round(prediction, 2)} Lakhs")
