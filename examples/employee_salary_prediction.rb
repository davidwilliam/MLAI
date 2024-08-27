# frozen_string_literal: true

# Example: Predicting Employee Salaries Based on Experience, Education, and Skills

require_relative '../lib/ml_ai'

# Initialize the model with a smaller regularization parameter
model = MLAI::MultipleLinearRegression.new(regularization: 0.00083)

# Create a Dataset from the CSV file
dataset = MLAI::Dataset.new('data/employee_salaries.csv')

# Fit the model using the Dataset
model.fit(dataset: dataset, feature_columns: ['Experience', 'Education', 'Skills'], target_column: 'Salary')

# Print coefficients and intercept for comparison
puts "Coefficients: #{model.coefficients.map { |coef| coef.round(2) }}"
puts "Intercept: #{model.intercept.round(2)}"

# Predict the salary of a new employee
new_employee_features = [[6, 3, 6]] # Experience: 6 years, Education: Master's, Skills: 6
predicted_salary = model.predict(new_employee_features).first.round(2)
puts "Predicted Salary for the new employee: $#{predicted_salary}"

# Evaluate the model using the original dataset
original_features = dataset.data.map { |row| row[0..2] } # Assuming 'Experience', 'Education', 'Skills' are the first three columns
original_salaries = dataset.data.map { |row| row[3] } # Assuming 'Salary' is the fourth column
predictions = model.predict(original_features).map { |p| p.round(2) }

mse = model.mean_squared_error(original_salaries, predictions).round(2)
r2 = model.r_squared(original_salaries, predictions).round(2)

puts "Mean Squared Error: #{mse}"
puts "R-squared: #{r2}"
