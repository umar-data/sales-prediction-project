import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Sales Prediction App")

# Load data
df = pd.read_csv("data.csv")

st.subheader("📂 Dataset")
st.write(df)

# Train model
X = df[["Month"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

# User input
st.subheader("🔮 Predict Future Sales")
month = st.number_input("Enter future month", min_value=1, step=1)

if st.button("Predict"):
    prediction = model.predict([[month]])
    st.success(f"Predicted Sales for Month {month}: {int(prediction[0])}")