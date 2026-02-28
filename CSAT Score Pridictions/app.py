import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open("csat_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("DeepCSAT – Ecommerce Satisfaction Predictor")

st.write("Enter customer support details to predict CSAT Score")

input_data = {}

for col in label_encoders.keys():
    options = label_encoders[col].classes_
    input_data[col] = st.selectbox(col, options)

# For numeric columns (you may customize based on dataset)
order_value = st.number_input("Order Value", min_value=0.0)
response_time = st.number_input("Response Time (minutes)", min_value=0.0)
resolution_time = st.number_input("Resolution Time (minutes)", min_value=0.0)

if st.button("Predict CSAT Score"):
    df_input = pd.DataFrame([input_data])

    for col in label_encoders:
        df_input[col] = label_encoders[col].transform(df_input[col])

    df_input["Order Value"] = order_value
    df_input["Response Time"] = response_time
    df_input["Resolution Time"] = resolution_time

    prediction = model.predict(df_input)[0]

    st.success(f"Predicted CSAT Score: {round(prediction,2)} / 5")

    if prediction >= 4:
        st.info("Customer is likely highly satisfied.")
    elif prediction >= 3:
        st.warning("Customer satisfaction is moderate.")
    else:
        st.error("Customer satisfaction is low.")