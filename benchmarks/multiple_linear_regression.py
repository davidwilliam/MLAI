import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data from CSV file
csv_file_path = "data/multiple_linear_regression_data.csv"
data = pd.read_csv(csv_file_path)

# Extract feature and target columns
x_csv = data[['Feature1', 'Feature2']].values
y_csv = data['Target'].values

# Initialize and fit the model using the CSV data
model_csv = LinearRegression()
model_csv.fit(x_csv, y_csv)

# Make predictions on the same data
predictions_csv = model_csv.predict(x_csv)
predictions_csv = np.round(predictions_csv, 2)  # Limit to two decimal places

# Calculate evaluation metrics
mse_csv = mean_squared_error(y_csv, predictions_csv)
r2_csv = r2_score(y_csv, predictions_csv)

# Print the results
print(f"CSV Coefficients: {np.round(model_csv.coef_, 2)}")
print(f"CSV Intercept: {round(model_csv.intercept_, 2)}")
print(f"CSV Predictions: {predictions_csv}")
print(f"CSV MSE: {round(mse_csv, 2)}")
print(f"CSV R-squared: {round(r2_csv, 2)}")

# Predict on new data using the model trained on CSV data
new_data = np.array([
    [6, 7],
    [7, 8]
])
new_predictions_csv = model_csv.predict(new_data)
new_predictions_csv = np.round(new_predictions_csv, 2)  # Limit to two decimal places

print(f"New Predictions from CSV: {new_predictions_csv}")

# Original example for comparison

# Define the dataset with multiple features
x_values = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])
y_values = np.array([5, 7, 9, 11, 13])

# Initialize and fit the model
model = LinearRegression()
model.fit(x_values, y_values)

# Make predictions on the same data
predictions = model.predict(x_values)
predictions = np.round(predictions, 2)  # Limit to two decimal places

# Calculate evaluation metrics
mse = mean_squared_error(y_values, predictions)
r2 = r2_score(y_values, predictions)

# Print the results
print(f"Coefficients: {np.round(model.coef_, 2)}")
print(f"Intercept: {round(model.intercept_, 2)}")
print(f"Predictions: {predictions}")
print(f"MSE: {round(mse, 2)}")
print(f"R-squared: {round(r2, 2)}")

# Predict on new data
new_predictions = model.predict(new_data)
new_predictions = np.round(new_predictions, 2)  # Limit to two decimal places

print(f"New Predictions: {new_predictions}")
