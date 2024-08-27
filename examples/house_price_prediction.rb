# frozen_string_literal: true

# Example: Predicting House Prices Based on Size

require_relative '../lib/ml_ai'

# Initialize the model
model = MLAI::SimpleLinearRegression.new

# Create a Dataset from the CSV file
dataset = MLAI::Dataset.new('data/house_prices.csv')

# Fit the model using the Dataset
model.fit(dataset: dataset, feature_column: 'Size', target_column: 'Price')

# Predict the price of a new house
new_house_size = [1600] # Size of the new house in square feet
predicted_price = model.predict(new_house_size).first.round(2)
puts "Predicted Price for a 1600 sq ft house: #{predicted_price} thousand dollars"

# Evaluate the model using the original dataset
original_sizes = dataset.data.map { |row| row[0] } # Assuming 'Size' is the first column
original_prices = dataset.data.map { |row| row[1] } # Assuming 'Price' is the second column
predictions = model.predict(original_sizes).map { |p| p.round(2) }

mse = model.mean_squared_error(original_prices, predictions).round(2)
r2 = model.r_squared(original_prices, predictions).round(2)

puts "Mean Squared Error: #{mse}"
puts "R-squared: #{r2}"
