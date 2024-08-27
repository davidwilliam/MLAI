from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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
new_data = np.array([
    [6, 7],
    [7, 8]
])
new_predictions = model.predict(new_data)
new_predictions = np.round(new_predictions, 2)  # Limit to two decimal places

print(f"New Predictions: {new_predictions}")
