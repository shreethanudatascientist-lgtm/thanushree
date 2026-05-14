import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from streamlit.components.v1 import html
from functions import *

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Carbon Footprint Calculator",
    page_icon="🌍",
    layout="wide"
)

# ---------------- HELPER FUNCTIONS ---------------- #

def get_base64(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

def load_model():
    try:
        with open("./models/model.sav", "rb") as f:
            model = pickle.load(f)

        with open("./models/scale.sav", "rb") as f:
            scaler = pickle.load(f)

        return model, scaler

    except FileNotFoundError:
        st.error("Model files not found inside models folder.")
        st.stop()

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ---------------- LOAD CSS ---------------- #

background = get_base64("./media/background_min.jpg")

try:
    with open("./style/style.css", "r") as style:
        css = f"""
        <style>
        {style.read().format(background=background)}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
except:
    pass

# ---------------- LOAD MODEL ---------------- #

model, scaler = load_model()

# ---------------- TITLE ---------------- #

st.title("🌍 Carbon Footprint Calculator")
st.write("Calculate your monthly carbon footprint")

# ---------------- USER INPUTS ---------------- #

st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    height = st.number_input(
        "Height (cm)",
        min_value=100,
        max_value=250,
        value=170
    )

with col2:
    weight = st.number_input(
        "Weight (kg)",
        min_value=30,
        max_value=250,
        value=70
    )

# BMI Calculation

bmi = weight / ((height / 100) ** 2)

if bmi < 18.5:
    body_type = "underweight"
elif bmi < 25:
    body_type = "normal"
elif bmi < 30:
    body_type = "overweight"
else:
    body_type = "obese"

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

diet = st.selectbox(
    "Diet",
    ["omnivore", "vegetarian", "vegan", "pescatarian"]
)

social = st.selectbox(
    "Social Activity",
    ["never", "sometimes", "often"]
)

# ---------------- TRAVEL ---------------- #

st.header("🚗 Travel")

transport = st.selectbox(
    "Transportation",
    ["public", "private", "walk/bicycle"]
)

vehicle_type = "None"

if transport == "private":
    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["petrol", "diesel", "hybrid", "electric"]
    )

vehicle_km = st.slider(
    "Monthly Distance Travelled (km)",
    0,
    5000,
    100
)

air_travel = st.selectbox(
    "Air Travel Frequency",
    ["never", "rarely", "frequently", "very frequently"]
)

# ---------------- WASTE ---------------- #

st.header("🗑️ Waste")

waste_bag = st.selectbox(
    "Waste Bag Size",
    ["small", "medium", "large", "extra large"]
)

waste_count = st.slider(
    "Waste Bags Per Week",
    0,
    10,
    2
)

recycle = st.multiselect(
    "Recycled Materials",
    ["Plastic", "Paper", "Metal", "Glass"]
)

# ---------------- ENERGY ---------------- #

st.header("⚡ Energy")

heating_energy = st.selectbox(
    "Heating Energy Source",
    ["electricity", "natural gas", "wood", "coal"]
)

daily_tv_pc = st.slider(
    "Daily PC/TV Usage (hours)",
    0,
    24,
    5
)

internet_daily = st.slider(
    "Daily Internet Usage (hours)",
    0,
    24,
    6
)

energy_efficiency = st.selectbox(
    "Use Energy Efficient Devices?",
    ["Yes", "No", "Sometimes"]
)

# ---------------- CONSUMPTION ---------------- #

st.header("💸 Consumption")

shower = st.selectbox(
    "Shower Frequency",
    ["daily", "twice a day", "more frequently", "less frequently"]
)

grocery_bill = st.slider(
    "Monthly Grocery Bill ($)",
    0,
    1000,
    200
)

clothes_monthly = st.slider(
    "Clothes Purchased Monthly",
    0,
    30,
    2
)

# ---------------- DATAFRAME ---------------- #

data = {
    "Body Type": body_type,
    "Sex": gender,
    "Diet": diet,
    "How Often Shower": shower,
    "Heating Energy Source": heating_energy,
    "Transport": transport,
    "Social Activity": social,
    "Monthly Grocery Bill": grocery_bill,
    "Frequency of Traveling by Air": air_travel,
    "Vehicle Monthly Distance Km": vehicle_km,
    "Waste Bag Size": waste_bag,
    "Waste Bag Weekly Count": waste_count,
    "How Long TV PC Daily Hour": daily_tv_pc,
    "Vehicle Type": vehicle_type,
    "How Many New Clothes Monthly": clothes_monthly,
    "How Long Internet Daily Hour": internet_daily,
    "Energy efficiency": energy_efficiency
}

# Add recycle columns

for item in recycle:
    data[f"Do You Recyle_{item}"] = 1

# ---------------- PREPROCESS ---------------- #

try:
    df = pd.DataFrame(data, index=[0])

    processed_data = input_preprocessing(df)

    sample_df = pd.DataFrame(
        data=np.zeros((1, len(sample))),
        columns=sample
    )

    sample_df[processed_data.columns] = processed_data.values

except Exception as e:
    st.error(f"Preprocessing Error: {e}")
    st.stop()

# ---------------- PREDICTION ---------------- #

if st.button("Calculate Carbon Footprint"):

    try:
        scaled_data = scaler.transform(sample_df)

        prediction = model.predict(scaled_data)[0]

        prediction = round(np.exp(prediction))

        st.success(
            f"Your Monthly Carbon Footprint: {prediction} kg CO₂"
        )

        tree_count = round(prediction / 411.4)

        st.info(
            f"You owe nature approximately {tree_count} trees 🌳"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ---------------- FOOTER ---------------- #

st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit"
)
