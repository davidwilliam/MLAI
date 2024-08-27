import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from CSV file
csv_file_path = "data/simple_linear_regression_data.csv"
data = pd.read_csv(csv_file_path)

# Extract feature and target columns
x_csv = data[['Feature']].values
y_csv = data['Target'].values

# Initialize the model and fit using the CSV data
model_csv = LinearRegression().fit(x_csv, y_csv)

# Make predictions on the same data
predictions_csv = model_csv.predict(x_csv)
print(f"CSV Dataset Predictions: {predictions_csv}")  # Output will match the Target column

# Make a prediction on new data
new_data_prediction = model_csv.predict([[6]])
print(f"New Data Prediction from CSV: {new_data_prediction}")  # Example: [13.]

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
