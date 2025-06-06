# bhp_model.py
import json
import numpy as np
from joblib import load

# Load trained model and columns
model = load('bangalore_home_prices_model.joblib')
with open("columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)
