import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
df = pd.read_csv('data/energy_consumption.csv')

# Prepare the features (X) and the target (y)
X = df[['Size', 'Occupants', 'Computers']]
y = df['EnergyConsumption']

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Perform predictions on the same dataset to evaluate the model
predictions = model.predict(X)
predictions = np.round(predictions, 2)  # Round predictions to 2 decimal places

# Calculate evaluation metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Print the results
print(f"Coefficients: {np.round(model.coef_, 2)}")
print(f"Intercept: {round(model.intercept_, 2)}")
print(f"Predictions: {predictions}")
print(f"MSE: {round(mse, 2)}")
print(f"R-squared: {round(r2, 2)}")

# Predict the energy consumption for a new building
new_building = pd.DataFrame([[3500, 60, 70]], columns=['Size', 'Occupants', 'Computers'])
new_prediction = model.predict(new_building)
new_prediction = round(new_prediction[0], 2)  # Round the prediction to 2 decimal places

print(f"Predicted energy consumption for the new building: {new_prediction} kWh")
