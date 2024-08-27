import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from CSV file
csv_file_path = "data/car_prices.csv"
data = pd.read_csv(csv_file_path)

# Extract feature and target columns
x = data[['Age', 'Mileage', 'Horsepower']].values
y = data['Price'].values

# Initialize and fit the model
model = LinearRegression()
model.fit(x, y)

# Make predictions on the same data
predictions = model.predict(x)
predictions = np.round(predictions, 2)  # Limit to two decimal places

# Calculate evaluation metrics
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Print the results
print(f"Coefficients: {np.round(model.coef_, 2)}")
print(f"Intercept: {round(model.intercept_, 2)}")
print(f"Predictions: {predictions}")
print(f"MSE: {round(mse, 2)}")
print(f"R-squared: {round(r2, 2)}")

# Predict on new data
new_data = np.array([[4, 55000, 140]])
new_prediction = model.predict(new_data)
new_prediction = np.round(new_prediction, 2)  # Limit to two decimal places

print(f"Predicted Price for the car: ${new_prediction[0]}")
