import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: white !important;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        font-weight: 600;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .info-box h4, .info-box p {
        color: #262730 !important;
        margin: 0.5rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 2rem 0;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load dataset (used for ranges & dropdowns)
@st.cache_data
def load_data():
    return pd.read_csv("cars24-car-price.csv")


@st.cache_resource
def load_model():
    with open("car_pred_model", "rb") as f:
        return pickle.load(f)


cars_df = load_data()

# Header
st.markdown(
    '<div class="main-header">🚗 Car Resale Price Predictor</div>',
    unsafe_allow_html=True,
)

# Introduction with info box
st.markdown(
    """
<div class="info-box">
    <h4>📊 How it works</h4>
    <p>This intelligent app analyzes various car features to predict the resale price using machine learning. 
    Simply enter your car's details below and get an instant price estimate!</p>
</div>
""",
    unsafe_allow_html=True,
)

# Load the pre-trained model
model = load_model()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown('<div class="sub-header">📈 Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(cars_df.head(), use_container_width=True)

    # Add some statistics
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Cars", f"{len(cars_df):,}")
    st.metric("Avg Price", f"₹{cars_df['selling_price'].mean():.2f}L")
    st.metric(
        "Price Range",
        f"₹{cars_df['selling_price'].min():.1f}L - ₹{cars_df['selling_price'].max():.1f}L",
    )

with col1:
    # --- Helpers from dataset ---
    yr_min, yr_max = int(cars_df["year"].min()), int(cars_df["year"].max())
    km_min, km_max = int(cars_df["km_driven"].min()), int(cars_df["km_driven"].max())
    mil_min, mil_max = float(cars_df["mileage"].min()), float(cars_df["mileage"].max())
    eng_min, eng_max = int(cars_df["engine"].min()), int(cars_df["engine"].max())
    pwr_min, pwr_max = (
        float(cars_df["max_power"].min()),
        float(cars_df["max_power"].max()),
    )
    seat_opts = sorted(cars_df["seats"].dropna().unique().astype(int))

    fuel_opts = list(cars_df["fuel_type"].dropna().unique())
    seller_opts = list(cars_df["seller_type"].dropna().unique())
    trans_opts = list(cars_df["transmission_type"].dropna().unique())

    # --- UI inputs ---
    st.markdown(
        '<div class="sub-header">🔧 Enter Car Details</div>', unsafe_allow_html=True
    )

    # Create input sections with better organization
    with st.expander("🏭 Basic Information", expanded=True):
        col_a, col_b = st.columns(2)

        with col_a:
            year = st.number_input(
                "📅 Year of Manufacture",
                min_value=yr_min,
                max_value=yr_max,
                step=1,
                value=min(2015, yr_max),
                help="The year your car was manufactured",
            )

            fuel_type = st.selectbox(
                "⛽ Fuel Type", fuel_opts, help="Select the fuel type of your car"
            )

            seller_type = st.selectbox(
                "👤 Seller Type", seller_opts, help="Who is selling the car?"
            )

        with col_b:
            km_driven = st.slider(
                "🛣️ Kilometers Driven",
                min_value=km_min,
                max_value=km_max,
                step=1000,
                value=min(50000, km_max),
                help="Total distance driven by the car",
            )

            transmission_type = st.selectbox(
                "⚙️ Transmission Type",
                trans_opts,
                help="Manual or Automatic transmission",
            )

            seats = st.selectbox(
                "💺 Number of Seats", seat_opts, help="Total seating capacity"
            )

    with st.expander("🔧 Technical Specifications", expanded=True):
        col_c, col_d, col_e = st.columns(3)

        with col_c:
            mileage = st.slider(
                "⛽ Mileage (kmpl)",
                min_value=mil_min,
                max_value=mil_max,
                step=0.5,
                value=min(18.0, mil_max),
                help="Fuel efficiency in kilometers per liter",
            )

        with col_d:
            engine = st.slider(
                "🔧 Engine (CC)",
                min_value=eng_min,
                max_value=eng_max,
                step=50,
                value=min(1200, eng_max),
                help="Engine displacement in cubic centimeters",
            )

        with col_e:
            max_power = st.slider(
                "⚡ Max Power (bhp)",
                min_value=pwr_min,
                max_value=pwr_max,
                step=1.0,
                value=min(80.0, pwr_max),
                help="Maximum power output in brake horsepower",
            )

    # --- Encoding (must match training) ---
    encode_dict = {
        "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5},
        "seller_type": {"Dealer": 1, "Individual": 2, "Trustmark Dealer": 3},
        "transmission_type": {"Manual": 1, "Automatic": 2},
    }

    # Prediction button and results
    st.markdown("---")

    if st.button("🎯 Get Price Prediction"):
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

            # Display prediction in a styled box
            st.markdown(
                f"""
            <div class="prediction-box">
                <h2>💰 Predicted Car Price</h2>
                <h1>₹ {round(float(pred), 2)} Lakhs</h1>
                <p>Based on the provided specifications</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Additional insights
            st.success("✅ Price prediction completed successfully!")

            # Show feature summary
            with st.expander("📋 Feature Summary"):
                feature_df = pd.DataFrame(
                    {
                        "Feature": [
                            "Year",
                            "Seller Type",
                            "KM Driven",
                            "Fuel Type",
                            "Transmission",
                            "Mileage",
                            "Engine",
                            "Max Power",
                            "Seats",
                        ],
                        "Value": [
                            year,
                            seller_type,
                            f"{km_driven:,} km",
                            fuel_type,
                            transmission_type,
                            f"{mileage} kmpl",
                            f"{engine} CC",
                            f"{max_power} bhp",
                            f"{seats} seats",
                        ],
                    }
                )
                st.dataframe(feature_df, use_container_width=True)

        except KeyError as e:
            st.error(
                f"❌ Value not found in encoding map: {e}. "
                "Make sure encodings match the ones used during training."
            )
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🚗 Car Price Predictor | Built with Streamlit & Machine Learning</p>
    <p><small>Predictions are estimates based on historical data and should be used as a reference only.</small></p>
</div>
""",
    unsafe_allow_html=True,
)
