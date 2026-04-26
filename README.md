import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear\_model import LinearRegression



\# CSV se data load

df = pd.read\_csv("data.csv")



X = df\[\["Month"]]

y = df\["Sales"]



model = LinearRegression()

model.fit(X, y)



future\_months = np.array(\[\[7], \[8], \[9]])

predictions = model.predict(future\_months)



print("Predictions:")

for i, pred in enumerate(predictions):

&#x20;   print(f"Month {7+i}: {int(pred)}")



plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.scatter(future\_months, predictions)



plt.show()

