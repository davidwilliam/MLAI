# frozen_string_literal: true

# Example: Predicting Car Prices Based on Multiple Features

require_relative '../lib/ml_ai'

# Initialize the model
model = MLAI::MultipleLinearRegression.new

# Create a Dataset from the CSV file
dataset = MLAI::Dataset.new('data/car_prices.csv')

# Fit the model using the Dataset
model.fit(dataset: dataset, feature_columns: ['Age', 'Mileage', 'Horsepower'], target_column: 'Price')

# Predict the price of a new car
new_car_features = [[4, 55000, 140]] # Age: 4 years, Mileage: 55,000 miles, Horsepower: 140
predicted_price = model.predict(new_car_features).first.round(2)
puts "Predicted Price for the car: $#{predicted_price}"

# Evaluate the model using the original dataset
original_features = dataset.data.map { |row| row[0..2] } # Assuming 'Age', 'Mileage', 'Horsepower' are the first three columns
original_prices = dataset.data.map { |row| row[3] } # Assuming 'Price' is the fourth column
predictions = model.predict(original_features).map { |p| p.round(2) }

mse = model.mean_squared_error(original_prices, predictions).round(2)
r2 = model.r_squared(original_prices, predictions).round(2)

puts "Mean Squared Error: #{mse}"
puts "R-squared: #{r2}"
