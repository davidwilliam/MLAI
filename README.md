# MLAI

Algorithms for machine learning and artificial intelligence in [Ruby](https://www.ruby-lang.org/en/).

To check this Ruby implementation, all features are implemented and benchmarked against [scikit-learn](https://scikit-learn.org/), a well-known library in [Python](https://www.python.org/) for machine learning.

## Why?

[Python](https://www.python.org/) is the go-to language for machine learning. But even if you don't plan to use it in production, if you are working with Ruby, you deserve to have ML/AI algorithms available in [Ruby](https://www.ruby-lang.org/en/). :) 

I also know that many libraries are available in Ruby for machine learning and related areas. Here is a [curated list](https://github.com/arbox/machine-learning-with-ruby).

Then why? I could name several reasons, but it will all come down to my interest in the computational challenges associated with machine learning and AI. There is no better way to identify, isolate, and inspect bottlenecks and opportunities for improvement and even the proposal of new ideas than to implement core functionalities from scratch. This will be true for most things in software: if you really want to have a deeper, more precise, and more comprehensive view of some resource, implement that resource yourself. 

Plus, it is a lot of fun! 

## Observations on Benchmarking

Using the `scikit-learn` library in Python as a reference (and also to keep this Ruby implementation in check), allowed me to arrive at some interesting observations.

### Intermediary Results

Throughout the development process, a series of cases were encountered where intermediary computations—such as matrix operations and inversions—differed between the two implementations, sometimes significantly. However, these differences often had no impact on the final results, such as predictions, mean squared error (MSE), or R-squared values. The underlying reason for this lies in how numerical computations are handled in each language.

### Language-Specific Matrix Computations and Precision Matters

First, the way matrices are computed and inverted can vary between Ruby and Python, especially considering the libraries and methods used. Differences in how these operations are optimized and executed can lead to slight variations in intermediate results. Additionally, floating-point arithmetic is handled differently in each language, influenced by factors such as precision, rounding methods, and how numbers are represented internally. Ruby and Python might employ distinct strategies to manage these computations, leading to the observed differences.

### Theory vs Practice

These differences highlight the importance of understanding that while numerical algorithms are deterministic in theory, their practical implementations can introduce variability due to the intricacies of the programming languages and their respective libraries. Despite these variations, the ability of both implementations to yield virtually identical final results speaks the robustness of the underlying mathematical principles. This also emphasizes the value of cross-referencing implementations in different environments to ensure the reliability and accuracy of the models developed. By the way, one more reason for writing a new ML/AI llibrary in Ruby. :) 

## Installation

Run:

    $ gem install ml_ai

If you use Bundler, add the following to your Gemfile:

    gem 'ml_ai'

## Usage

### Simple Linear Regression

The `SimpleLinearRegression` class in the `MLAI` gem allows you to fit a linear model with a single feature, make predictions, and evaluate the model using metrics like Mean Squared Error (MSE) and R-squared. Additionally, it now supports loading data directly from a CSV file using the `Dataset` class.

### Using Raw Arrays

```ruby
require 'ml_ai'

# Initialize the model
model = MLAI::SimpleLinearRegression.new

# Define the dataset with a single feature
x_values = [1, 2, 3, 4, 5]
y_values = [3, 5, 7, 9, 11]

# Fit the model to the data
model.fit(x_values: x_values, y_values: y_values)

# Make predictions on the original data
predictions = model.predict(x_values)
puts "Predictions: #{predictions}"
# Output: Predictions: [3.0, 5.0, 7.0, 9.0, 11.0]

# Make predictions on new data
new_data = [6, 7]
new_predictions = model.predict(new_data)
puts "New Predictions: #{new_predictions}"
# Output: New Predictions: [13.0, 15.0]

# Calculate evaluation metrics
mse = model.mean_squared_error(y_values, predictions)
r2 = model.r_squared(y_values, predictions)

puts "Mean Squared Error: #{mse}"
# Output: Mean Squared Error: 0.0

puts "R-squared: #{r2}"
# Output: R-squared: 1.0
```

#### Using a CSV Dataset

```ruby 
require 'ml_ai'

# Initialize the model
model = MLAI::SimpleLinearRegression.new

# Create a Dataset from a CSV file
dataset = MLAI::Dataset.new('data/simple_linear_regression_data.csv')

# Fit the model using the Dataset
model.fit(dataset: dataset, feature_column: 'Feature', target_column: 'Target')

# Make predictions on new feature values
new_features = [6, 7]
new_predictions = model.predict(new_features)
puts "New Predictions: #{new_predictions}"
# Output: New Predictions: [13.0, 15.0]

# Evaluate the model using the original dataset
original_features = dataset.data.map { |row| row[0] } # Assuming 'Feature' is the first column
original_targets = dataset.data.map { |row| row[1] } # Assuming 'Target' is the second column
original_predictions = model.predict(original_features)

mse = model.mean_squared_error(original_targets, original_predictions)
r2 = model.r_squared(original_targets, original_predictions)

puts "Mean Squared Error: #{mse}"
# Output: Mean Squared Error: 0.0

puts "R-squared: #{r2}"
# Output: R-squared: 1.0
```

#### Benchmark

To check the Ruby implementation, run the Python benchmark using the same data:

```
$ python3 benchmarks/simple_linear_regression_benchmark.py
```

### Evaluation Metrics

Fit a linear model, make predictions, and evaluate the model using common metrics like Mean Squared Error (MSE) and R-squared.

```ruby
require 'ml_ai'

# Initialize the model
model = MLAI::SimpleLinearRegression.new

# Define the dataset
x_values = [1, 2, 3, 4, 5]
y_values = [2, 4, 5, 4, 5]

# Fit the model to the data
model.fit(x_values, y_values)

# Make predictions
predictions = model.predict(x_values)

# Calculate evaluation metrics
mse = model.mean_squared_error(y_values, predictions)
r2 = model.r_squared(y_values, predictions)

# Output the results
puts "Predictions: #{predictions}"
puts "Mean Squared Error: #{mse}"
puts "R-squared: #{r2}"

# Example output:
# Predictions: [2.8, 3.4, 4.0, 4.6, 5.2]
# Mean Squared Error: 0.48
# R-squared: 0.6
```

#### Benchmark

To check the Ruby implementation, run the Python benchmark using the same data:

```
$ python3 benchmarks/evaluation_metrics.py
```

#### Example

Here is an example closer to the real world. Imagine you're working in real estate and want to predict the price of a house based on its size. You've gathered data from various houses, including their sizes in square feet and their prices in thousands of dollars. You want to use this data to predict the price of new houses based on their size.

You can run this example here:

```
ruby examples/house_price_prediction.rb
```

and check the results with the following benchmark in Python:

```
python3 benchmarks/house_price_prediction_benchmark.py
```

### Multiple Linear Regression

The `MultipleLinearRegression` class in the `MLAI` gem allows you to fit a linear model with multiple features, make predictions, and evaluate the model using metrics like Mean Squared Error (MSE) and R-squared. The class supports loading data directly from a CSV file using the `Dataset` class, as well as passing raw arrays directly.

### Using Raw Arrays

```ruby
require 'ml_ai'

# Initialize the model
model = MLAI::MultipleLinearRegression.new

# Define the dataset with multiple features
x_values = [
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6]
]
y_values = [5, 7, 9, 11, 13]

# Fit the model to the data
model.fit(x_values: x_values, y_values: y_values)

# Make predictions on the original data
predictions = model.predict(x_values)
puts "Predictions: #{predictions}"
# Output: Predictions: [5.0, 7.0, 9.0, 11.0, 13.0]

# Make predictions on new data
new_data = [
  [6, 7],
  [7, 8]
]
new_predictions = model.predict(new_data)
puts "New Predictions: #{new_predictions}"
# Output: New Predictions: [15.0, 17.0]

# Calculate evaluation metrics
mse = model.mean_squared_error(y_values, predictions)
r2 = model.r_squared(y_values, predictions)

puts "Mean Squared Error: #{mse}"
# Output: Mean Squared Error: 0.0

puts "R-squared: #{r2}"
# Output: R-squared: 1.0
```

#### Using a CSV Dataset

```ruby
require 'ml_ai'

# Initialize the model
model = MLAI::MultipleLinearRegression.new

# Create a Dataset from a CSV file
dataset = MLAI::Dataset.new('data/multiple_linear_regression_data.csv')

# Fit the model using the Dataset
model.fit(dataset: dataset, feature_columns: ['Feature1', 'Feature2'], target_column: 'Target')

# Make predictions on new feature values
new_features = [
  [6, 7],
  [7, 8]
]
new_predictions = model.predict(new_features)
puts "New Predictions: #{new_predictions}"
# Output: New Predictions: [15.0, 17.0]

# Evaluate the model using the original dataset
original_features = dataset.data.map { |row| row[0..1] } # Assuming 'Feature1' and 'Feature2' are the first two columns
original_targets = dataset.data.map { |row| row[2] } # Assuming 'Target' is the third column
original_predictions = model.predict(original_features)

mse = model.mean_squared_error(original_targets, original_predictions)
r2 = model.r_squared(original_targets, original_predictions)

puts "Mean Squared Error: #{mse}"
# Output: Mean Squared Error: 0.0

puts "R-squared: #{r2}"
# Output: R-squared: 1.0
```

#### Benchmark

To check the Ruby implementation, run the Python benchmark using the same data:

```
$ python3 benchmarks/multiple_linear_regression.py
```

#### Example

Here is an example closer to the real world. Imagine you work at a car dealership, and you want to predict the price of used cars based on various features such as the car's age, mileage, and horsepower. You have collected data from previous sales and want to use this data to predict the price of new cars based on these features.

You can run this example here:

```
ruby examples/car_price_prediction.rb
```

and check the results with the following benchmark in Python:

```
python3 benchmarks/car_price_prediction_benchmark.py
```

### Regularization

Regularization helps prevent overfitting by adding a penalty to large coefficients, making the model more generalizable. In this example, we'll predict advertising revenue based on the amount spent on TV, radio, and newspaper advertisements. Regularization is applied to prevent overfitting and ensure the model generalizes well to new data.

```ruby
# frozen_string_literal: true

# Example: Predicting Advertising Revenue with Regularization

require_relative '../lib/ml_ai'

# Initialize the model with a regularization parameter
model = MLAI::MultipleLinearRegression.new(regularization: 0.00083)

# Create a Dataset from a CSV file
dataset = MLAI::Dataset.new('data/advertising_revenue.csv')

# Fit the model using the Dataset
model.fit(dataset: dataset, feature_columns: ['TV', 'Radio', 'Newspaper'], target_column: 'Revenue')

# Print coefficients and intercept
puts "Coefficients: #{model.coefficients.map { |coef| coef.round(2) }}"
puts "Intercept: #{model.intercept.round(2)}"

# Predict the revenue based on new advertising spends
new_ad_spend = [[230, 37, 69]] # TV: $230, Radio: $37, Newspaper: $69
predicted_revenue = model.predict(new_ad_spend).first.round(2)
puts "Predicted Advertising Revenue: $#{predicted_revenue}"

# Evaluate the model using the original dataset
original_features = dataset.data.map { |row| row[0..2] } # Extracting 'TV', 'Radio', 'Newspaper'
original_revenue = dataset.data.map { |row| row[3] } # Extracting 'Revenue'
predictions = model.predict(original_features).map { |p| p.round(2) }

mse = model.mean_squared_error(original_revenue, predictions).round(2)
r2 = model.r_squared(original_revenue, predictions).round(2)

puts "Mean Squared Error: #{mse}"
puts "R-squared: #{r2}"
```

#### Example

Here is an example closer to the real world. Imagine you're working for a company that wants to predict the salary of employees based on several factors, including their years of experience, level of education, and number of relevant skills. You have collected data from existing employees and want to use this data to predict salaries for new hires.

You can run this example here:

```
ruby examples/employee_salary_prediction.rb
```

and check the results with the following benchmark in Python:

```
python3 benchmarks/employee_salary_prediction_benchmark.py
```

### Cross-Validation

Cross-validation is a powerful technique to evaluate the performance of your model by splitting your dataset into multiple folds. The model is trained on a subset of the data and tested on the remaining data, and this process is repeated multiple times. The final evaluation metric is the average of all the individual metrics across the folds.

#### Example Usage with Raw Arrays

```ruby
# Require the necessary files
require 'ml_ai'

# Initialize the model with regularization
model = MLAI::MultipleLinearRegression.new(regularization: 0.01)

# Define the input data
x_values = [
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [5, 6]
]
y_values = [5, 7, 9, 11, 13]

# Perform 3-fold cross-validation
average_mse = model.cross_validate(x_values: x_values, y_values: y_values, k: 3)
puts "Average Mean Squared Error across 3 folds: #{average_mse.round(4)}"
```

#### Example Using a CSV Dataset

```ruby
# Require the necessary files
require 'ml_ai'

# Create a Dataset from a CSV file
dataset = MLAI::Dataset.new('path_to_your_dataset.csv')

# Initialize the model with regularization
model = MLAI::MultipleLinearRegression.new(regularization: 0.01)

# Perform 3-fold cross-validation
average_mse = model.cross_validate(dataset: dataset, feature_columns: ["Feature1", "Feature2"], target_column: "Target", k: 3)
puts "Average Mean Squared Error across 3 folds: #{average_mse.round(4)}"
```

#### Example

Here is an example closer to the real world. Imagine that we want to predict the energy consumption of a building based on its size, the number of occupants, and the number of computers it houses. We'll use the `MultipleLinearRegression` class with cross-validation.

You can run this example here:

```
ruby examples/energy_consumption_prediction.rb
```

and check the results with the following benchmark in Python:

```
python3 benchmarks/energy_consumption_prediction_benchmark.py
```

### Dataset


## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/davidwilliam/MLAI. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/davidwilliam/MLAI/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the MlAi project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/[USERNAME]/ml_ai/blob/main/CODE_OF_CONDUCT.md).