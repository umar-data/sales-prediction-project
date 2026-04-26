import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CSV se data load
df = pd.read_csv("data.csv")

X = df[["Month"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

future_months = np.array([[7], [8], [9]])
predictions = model.predict(future_months)

print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Month {7+i}: {int(pred)}")

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.scatter(future_months, predictions)

plt.show()