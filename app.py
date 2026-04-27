import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Sales Predictor", layout="centered")

# Title
st.title("📊 AI-Based Sales Prediction System")
st.write("This app predicts future sales using Linear Regression.")


# Load data
df = pd.read_csv("data.csv")

# Show dataset
with st.expander("📂 View Dataset"):
    st.write(df)

# Prepare data
X = df[["Month"]]
y = df["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
st.subheader("🔮 Predict Future Sales")
month = st.number_input("Enter Month", min_value=1, step=1)

#warning
if month > 24:
    st.warning("⚠️ Prediction may be less accurate for large months")

# Prediction
if st.button("Predict"):
    prediction = model.predict([[month]])
    
    st.success(f"💰 Expected Sales for Month {month}: ₹ {int(prediction[0])}")
    st.write("Made by Umar Shaikh 🚀")

    # Plot graph
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Actual Data")
    ax.plot(X, model.predict(X), label="Trend Line")
    ax.scatter(month, prediction, color="red", label="Prediction")

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)
st.markdown("---")
