# frozen_string_literal: true

require "test_helper"

class TestMultipleLinearRegression < Minitest::Test
  def setup
    @model = MLAI::MultipleLinearRegression.new
  end

  def test_multiple_linear_regression_basic
    x_values = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ]
    y_values = [5, 7, 9, 11, 13]

    @model.fit(x_values, y_values)
    predictions = @model.predict(x_values)

    # Expected predictions based on y = 1.0 * x1 + 1.0 * x2 + 2.0
    expected_predictions = [5, 7, 9, 11, 13]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction should be close to #{expected_predictions[index]}"
    end
  end

  def test_multiple_linear_regression_with_new_data
    x_values = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ]
    y_values = [5, 7, 9, 11, 13]

    @model.fit(x_values, y_values)
    new_data = [
      [6, 7],
      [7, 8]
    ]
    predictions = @model.predict(new_data)

    # Expected predictions for new data based on y = 1.0 * x1 + 1.0 * x2 + 2.0
    expected_predictions = [15, 17]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction for new data should be close to #{expected_predictions[index]}"
    end
  end

  def test_evaluation_metrics
    x_values = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ]
    y_values = [5, 7, 9, 11, 13]

    @model.fit(x_values, y_values)
    predictions = @model.predict(x_values)

    mse = @model.mean_squared_error(y_values, predictions)
    r2 = @model.r_squared(y_values, predictions)

    assert_in_delta 0.0, mse, 0.01, "MSE should be close to 0.0"
    assert_in_delta 1.0, r2, 0.01, "R-squared should be close to 1.0"
  end
end
