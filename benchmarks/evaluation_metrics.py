from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Define the dataset
x_values = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_values = np.array([2, 4, 5, 4, 5])

# Initialize and fit the model
model = LinearRegression()
model.fit(x_values, y_values)

# Make predictions
predictions = model.predict(x_values)

# Limit predictions to two decimal digits
predictions = np.round(predictions, 2)

# Calculate evaluation metrics
mse = mean_squared_error(y_values, predictions)
r2 = r2_score(y_values, predictions)

# Limit MSE and R-squared to two decimal digits
mse = round(mse, 2)
r2 = round(r2, 2)

# Print results
print(f"Slope: {round(model.coef_[0], 2)}")
print(f"Intercept: {round(model.intercept_, 2)}")
print(f"Predictions: {predictions}")
print(f"MSE: {mse}")
print(f"R-squared: {r2}")
