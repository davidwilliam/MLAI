# MLAI

Algorithms for machine learning and artificial intelligence in Ruby.

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

### Benchmark

To check the Ruby implementation, run the Python benchmark using the same data:

```
$ python3 benchmarks/evaluation_metrics.py
```

Example 5: Multiple Linear Regression

The `MultipleLinearRegression` class in the `MLAI` gem allows you to fit a linear model with multiple features, make predictions, and evaluate the model using metrics like Mean Squared Error (MSE) and R-squared.

### Fitting a Model and Making Predictions

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
model.fit(x_values, y_values)

# Make predictions on the original data
predictions = model.predict(x_values)
puts "Predictions: #{predictions}"
# Output: Predictions: [5, 7, 9, 11, 13]

# Make predictions on new data
new_data = [
  [6, 7],
  [7, 8]
]
new_predictions = model.predict(new_data)
puts "New Predictions: #{new_predictions}"
# Output: New Predictions: [15, 17]
```

Evaluating the model:

```ruby
# Calculate evaluation metrics
mse = model.mean_squared_error(y_values, predictions)
r2 = model.r_squared(y_values, predictions)

puts "Mean Squared Error: #{mse}"
# Output: Mean Squared Error: 0.0

puts "R-squared: #{r2}"
# Output: R-squared: 1.0
```

### Benchmark

To check the Ruby implementation, run the Python benchmark using the same data:

```
$ python3 benchmarks/multiple_linear_regression.py
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/[USERNAME]/ml_ai. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/[USERNAME]/ml_ai/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the MlAi project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/[USERNAME]/ml_ai/blob/main/CODE_OF_CONDUCT.md).