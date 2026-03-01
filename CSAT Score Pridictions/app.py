import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="DeepCSAT Predictor",
    page_icon="📊",
    layout="centered"
)

# -------------------------
# Custom CSS Styling
# -------------------------

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.main {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}
h1 {
    text-align: center;
    color: #1f4037;
}
.stButton>button {
    background-color: #1f4037;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #14532d;
    color: white;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "csat_model.pkl")
encoder_path = os.path.join(BASE_DIR, "label_encoders.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    label_encoders = pickle.load(f)

# -------------------------
# App UI
# -------------------------

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("📊 DeepCSAT – Ecommerce Satisfaction Predictor")
st.write("### Enter Customer Details Below")

input_data = {}

# Two Column Layout
col1, col2 = st.columns(2)

# Categorical Inputs
for i, col in enumerate(label_encoders.keys()):
    if i % 2 == 0:
        with col1:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_)
    else:
        with col2:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_)

st.markdown("### 📈 Transaction Details")

col3, col4 = st.columns(2)

with col3:
    item_price = st.number_input("Item Price", min_value=0.0)

with col4:
    handling_time = st.number_input("Handling Time (minutes)", min_value=0.0)

st.markdown("")

# -------------------------
# Prediction Button
# -------------------------

if st.button("🚀 Predict CSAT Score"):

    df_input = pd.DataFrame([input_data])

    for col in label_encoders:
        df_input[col] = label_encoders[col].transform(df_input[col])

    df_input["Item_price"] = item_price
    df_input["connected_handling_time"] = handling_time

    df_input = df_input[model.feature_names_in_]

    prediction = model.predict(df_input)[0]

    if prediction >= 4:
        color = "#16a34a"
        message = "😊 Highly Satisfied Customer"
    elif prediction >= 3:
        color = "#f59e0b"
        message = "😐 Moderately Satisfied Customer"
    else:
        color = "#dc2626"
        message = "😞 Low Customer Satisfaction"

    st.markdown(f"""
        <div class="result-box" style="background-color:{color}; color:white;">
            Predicted CSAT Score: {round(float(prediction),2)} / 5 <br><br>
            {message}
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
