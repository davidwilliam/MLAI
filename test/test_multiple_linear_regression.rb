# frozen_string_literal: true

require "test_helper"

class TestMultipleLinearRegression < Minitest::Test
  # In your test file, when initializing the model:

  def setup
    @model = MLAI::MultipleLinearRegression.new(regularization: 0.01) # Increase regularization parameter

    # Prepare a CSV file for Dataset
    @csv_content = <<~CSV
      Feature1,Feature2,Target
      1,2,5
      2,3,7
      3,4,9
      4,5,11
      5,6,13
    CSV
    @csv_filename = "test_multiple_linear_regression.csv"
    File.write(@csv_filename, @csv_content)
    @dataset = MLAI::Dataset.new(@csv_filename)
  end


  def teardown
    File.delete(@csv_filename) if File.exist?(@csv_filename)
  end

  # Test fitting with raw arrays
  def test_multiple_linear_regression_with_arrays
    x_values = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ]
    y_values = [5, 7, 9, 11, 13]

    @model.fit(x_values: x_values, y_values: y_values)
    predictions = @model.predict(x_values)

    # Expected predictions based on y = 1.0 * x1 + 1.0 * x2 + 2.0
    expected_predictions = [5, 7, 9, 11, 13]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction should be close to #{expected_predictions[index]}"
    end
  end

  # Test fitting with a Dataset object
  def test_multiple_linear_regression_with_dataset
    # Fit using Dataset
    @model.fit(dataset: @dataset, feature_columns: ["Feature1", "Feature2"], target_column: "Target")
    # Predict using feature values
    predictions = @model.predict([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ])

    # Expected predictions based on y = 1.0 * x1 + 1.0 * x2 + 2.0
    expected_predictions = [5, 7, 9, 11, 13]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction should be close to #{expected_predictions[index]}"
    end
  end

  # Test predictions on new data with raw arrays
  def test_multiple_linear_regression_with_new_data_with_arrays
    x_values = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ]
    y_values = [5, 7, 9, 11, 13]

    @model.fit(x_values: x_values, y_values: y_values)
    new_data = [
      [6, 7],
      [7, 8]
    ]
    predictions = @model.predict(new_data)

    # Expected predictions for new data based on y = 1.0 * x1 + 1.0 * x2 + 2.0
    expected_predictions = [15, 17]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.02, "Prediction for new data should be close to #{expected_predictions[index]}"
    end
  end

  # Test predictions on new data with a Dataset object
  def test_multiple_linear_regression_with_new_data_with_dataset
    # Fit using Dataset
    @model.fit(dataset: @dataset, feature_columns: ["Feature1", "Feature2"], target_column: "Target")

    # Prepare new data as Dataset
    new_csv_content = <<~CSV
      Feature1,Feature2,Target
      6,7,15
      7,8,17
    CSV
    new_csv_filename = "test_new_multiple_linear_regression.csv"
    File.write(new_csv_filename, new_csv_content)
    new_dataset = MLAI::Dataset.new(new_csv_filename)
    new_feature_values = new_dataset.data.map { |row| row[0..1] } # Assuming "Feature1" and "Feature2" are the first two columns

    predictions = @model.predict(new_feature_values)
    expected_predictions = [15, 17]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.02, "Prediction for new data should be close to #{expected_predictions[index]}"
    end

    File.delete(new_csv_filename) if File.exist?(new_csv_filename)
  end

  # Test evaluation metrics with raw arrays
  def test_evaluation_metrics_with_arrays
    x_values = [
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ]
    y_values = [5, 7, 9, 11, 13]

    @model.fit(x_values: x_values, y_values: y_values)
    predictions = @model.predict(x_values)

    mse = @model.mean_squared_error(y_values, predictions)
    r2 = @model.r_squared(y_values, predictions)

    assert_in_delta 0.0, mse, 0.01, "MSE should be close to 0.0"
    assert_in_delta 1.0, r2, 0.01, "R-squared should be close to 1.0"
  end

  # Test evaluation metrics with a Dataset object
  def test_evaluation_metrics_with_dataset
    # Fit using Dataset
    @model.fit(dataset: @dataset, feature_columns: ["Feature1", "Feature2"], target_column: "Target")
    # Predict using feature values
    predictions = @model.predict([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6]
    ])
    y_values = [5, 7, 9, 11, 13]

    mse = @model.mean_squared_error(y_values, predictions)
    r2 = @model.r_squared(y_values, predictions)

    assert_in_delta 0.0, mse, 0.01, "MSE should be close to 0.0"
    assert_in_delta 1.0, r2, 0.01, "R-squared should be close to 1.0"
  end

  # Test fitting with regularization
  def test_multiple_linear_regression_with_regularization
    model_with_regularization = MLAI::MultipleLinearRegression.new(regularization: 0.1)

    # Fit using the existing dataset with regularization
    model_with_regularization.fit(dataset: @dataset, feature_columns: ["Feature1", "Feature2"], target_column: "Target")

    # Check that the coefficients are affected by regularization
    coefficients = model_with_regularization.coefficients
    assert coefficients.all? { |coef| coef.abs < 2.0 }, "Coefficients should be reduced due to regularization"

    # Predict using the fitted model
    predictions = model_with_regularization.predict([
      [4, 5],
      [6, 7]
    ])

    expected_predictions = predictions.map { |pred| pred.round(2) }
    assert predictions.all? { |pred| pred.is_a?(Float) }, "Predictions should be valid floating-point numbers"
  end

end