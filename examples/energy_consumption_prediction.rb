# frozen_string_literal: true

require_relative '../lib/ml_ai'

# Load the dataset
dataset = MLAI::Dataset.new('data/energy_consumption.csv')

# Initialize the MultipleLinearRegression model with slight regularization to avoid singular matrix
model = MLAI::MultipleLinearRegression.new(regularization: 1e-9)

# Perform 5-fold cross-validation
average_mse = model.cross_validate(dataset: dataset, feature_columns: ["Size", "Occupants", "Computers"], target_column: "EnergyConsumption", k: 5)

# Fit the model on the entire dataset
model.fit(dataset: dataset, feature_columns: ["Size", "Occupants", "Computers"], target_column: "EnergyConsumption")

# Get the coefficients and intercept
coefficients = model.coefficients.map { |coef| coef.round(2) }
intercept = model.intercept.round(2)

# Perform predictions on the same dataset to evaluate the model
predictions = model.predict(dataset.data.map { |row| row[0..2] })
predictions = predictions.map { |pred| pred.round(2) }  # Round predictions to 2 decimal places

# Calculate evaluation metrics
mse = model.mean_squared_error(dataset.data.map { |row| row[3] }, predictions)
r_squared = model.r_squared(dataset.data.map { |row| row[3] }, predictions)

# Print the results in the desired format
puts "Coefficients: #{coefficients.inspect}"
puts "Intercept: #{intercept}"
puts "Predictions: #{predictions.inspect}"
puts "MSE: #{mse.round(2)}"
puts "R-squared: #{r_squared.round(2)}"

# Predict the energy consumption for a new building
new_building = [[3500, 60, 70]]  # A building with 3500 sq ft, 60 occupants, and 70 computers
predicted_energy_consumption = model.predict(new_building).first
puts "Predicted energy consumption for the new building: #{predicted_energy_consumption.round(2)} kWh"
