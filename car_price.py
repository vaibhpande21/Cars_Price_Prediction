import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn

# Load dataset (used for ranges & dropdowns)
cars_df = pd.read_csv("cars24-car-price.csv")

st.title("Car Resale Price Prediction")
st.write("This app predicts the resale price of a car based on its features.")
st.dataframe(cars_df.head())

# Load the pre-trained model
with open("car_pred_model", "rb") as f:
    model = pickle.load(f)

# --- Helpers from dataset ---
yr_min, yr_max = int(cars_df["year"].min()), int(cars_df["year"].max())
km_min, km_max = int(cars_df["km_driven"].min()), int(cars_df["km_driven"].max())
mil_min, mil_max = float(cars_df["mileage"].min()), float(cars_df["mileage"].max())
eng_min, eng_max = int(cars_df["engine"].min()), int(cars_df["engine"].max())
pwr_min, pwr_max = float(cars_df["max_power"].min()), float(cars_df["max_power"].max())
seat_opts = sorted(cars_df["seats"].dropna().unique().astype(int))

fuel_opts = list(cars_df["fuel_type"].dropna().unique())
seller_opts = list(cars_df["seller_type"].dropna().unique())
trans_opts = list(cars_df["transmission_type"].dropna().unique())

# --- UI inputs ---
st.subheader("Enter Car Details")


year = st.number_input(
    "Year of Manufacture",
    min_value=yr_min,
    max_value=yr_max,
    step=1,
    value=min(2015, yr_max),
)

km_driven = st.slider(
    "Kilometers Driven",
    min_value=km_min,
    max_value=km_max,
    step=1000,
    value=min(50000, km_max),
)

fuel_type = st.selectbox("Fuel Type", fuel_opts)
seller_type = st.selectbox("Seller Type", seller_opts)
transmission_type = st.selectbox("Transmission Type", trans_opts)

mileage = st.slider(
    "Mileage (kmpl)",
    min_value=mil_min,
    max_value=mil_max,
    step=0.5,
    value=min(18.0, mil_max),
)

engine = st.slider(
    "Engine (CC)",
    min_value=eng_min,
    max_value=eng_max,
    step=50,
    value=min(1200, eng_max),
)

max_power = st.slider(
    "Max Power (bhp)",
    min_value=pwr_min,
    max_value=pwr_max,
    step=1.0,
    value=min(80.0, pwr_max),
)

seats = st.selectbox("Seats", seat_opts)

# --- Encoding (must match training) ---
encode_dict = {
    "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5},
    "seller_type": {"Dealer": 1, "Individual": 2, "Trustmark Dealer": 3},
    "transmission_type": {"Manual": 1, "Automatic": 2},
}

if st.button("Get Price"):
    try:
        ef = encode_dict["fuel_type"][fuel_type]
        es = encode_dict["seller_type"][seller_type]
        et = encode_dict["transmission_type"][transmission_type]

        features = [
            year,  # year
            es,  # seller_type (encoded)
            km_driven,  # km_driven
            ef,  # fuel_type (encoded)
            et,  # transmission_type (encoded)
            mileage,  # mileage
            engine,  # engine
            max_power,  # max_power
            seats,  # seats
        ]

        pred = model.predict([features])[0]
        st.header(f"Predicted Price(in Lakh): ₹ {round(float(pred), 2)}")
    except KeyError as e:
        st.error(
            f"Value not found in encoding map: {e}. "
            "Make sure encodings match the ones used during training."
        )
