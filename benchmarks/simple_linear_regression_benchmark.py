from sklearn.linear_model import LinearRegression
import numpy as np

# Example 1: Basic Usage
x_basic = np.array([1, 2, 3]).reshape(-1, 1)
y_basic = np.array([2, 4, 6])

# Initialize the model and fit
model_basic = LinearRegression().fit(x_basic, y_basic)

# Make a prediction
prediction_basic = model_basic.predict([[4]])
print(f"Basic Example Prediction: {prediction_basic}")  # Output: [8.]

# Example 2: Larger Dataset
x_large = np.array(range(1, 101)).reshape(-1, 1)
y_large = np.array([3 * x + 5 for x in range(1, 101)])

# Initialize the model and fit
model_large = LinearRegression().fit(x_large, y_large)

# Make a prediction
prediction_large = model_large.predict([[150]])
print(f"Larger Dataset Prediction: {prediction_large}")  # Output: [455.]

# Example 3: Handling Negative and Positive Values
x_mixed = np.array([-10, -5, 0, 5, 10]).reshape(-1, 1)
y_mixed = np.array([2 * x - 3 for x in [-10, -5, 0, 5, 10]])

# Initialize the model and fit
model_mixed = LinearRegression().fit(x_mixed, y_mixed)

# Make a prediction
prediction_mixed = model_mixed.predict([[15]])
print(f"Negative and Positive Values Prediction: {prediction_mixed}")  # Output: [27.]
