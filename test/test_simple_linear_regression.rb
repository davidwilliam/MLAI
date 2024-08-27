# frozen_string_literal: true

require "test_helper"

class TestSimpleLinearRegression < Minitest::Test
  def setup
    @model = MLAI::SimpleLinearRegression.new

    # Prepare a CSV file for Dataset
    @csv_content = <<~CSV
      Feature,Target
      1,3
      2,5
      3,7
      4,9
      5,11
    CSV
    @csv_filename = "test_simple_linear_regression.csv"
    File.write(@csv_filename, @csv_content)
    @dataset = MLAI::Dataset.new(@csv_filename)
  end

  def teardown
    File.delete(@csv_filename) if File.exist?(@csv_filename)
  end

  # Test fitting with raw arrays
  def test_simple_linear_regression_basic_with_arrays
    x_values = [1, 2, 3, 4, 5]
    y_values = [3, 5, 7, 9, 11]

    @model.fit(x_values: x_values, y_values: y_values)
    predictions = @model.predict(x_values)

    # Expected predictions based on y = 2x + 1
    expected_predictions = [3, 5, 7, 9, 11]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction should be close to #{expected_predictions[index]}"
    end
  end

  # Test fitting with a Dataset object
  def test_simple_linear_regression_basic_with_dataset
    # Fit using Dataset
    @model.fit(dataset: @dataset, feature_column: "Feature", target_column: "Target")
    # Predict using feature values
    predictions = @model.predict([1, 2, 3, 4, 5])

    # Expected predictions based on y = 2x + 1
    expected_predictions = [3, 5, 7, 9, 11]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction should be close to #{expected_predictions[index]}"
    end
  end

  # Test predictions on new data with raw arrays
  def test_simple_linear_regression_with_new_data_with_arrays
    x_values = [1, 2, 3, 4, 5]
    y_values = [3, 5, 7, 9, 11]

    @model.fit(x_values: x_values, y_values: y_values)
    new_data = [6, 7]
    predictions = @model.predict(new_data)

    # Expected predictions for new data based on y = 2x + 1
    expected_predictions = [13, 15]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction for new data should be close to #{expected_predictions[index]}"
    end
  end

  # Test predictions on new data with a Dataset object
  def test_simple_linear_regression_with_new_data_with_dataset
    # Fit using Dataset
    @model.fit(dataset: @dataset, feature_column: "Feature", target_column: "Target")
    # Prepare new data as Dataset
    new_csv_content = <<~CSV
      Feature,Target
      6,13
      7,15
    CSV
    new_csv_filename = "test_new_simple_linear_regression.csv"
    File.write(new_csv_filename, new_csv_content)
    new_dataset = MLAI::Dataset.new(new_csv_filename)
    new_feature_values = new_dataset.data.map { |row| row[0] } # Assuming "Feature" is the first column

    predictions = @model.predict(new_feature_values)
    expected_predictions = [13, 15]

    predictions.each_with_index do |pred, index|
      assert_in_delta expected_predictions[index], pred, 0.01, "Prediction for new data should be close to #{expected_predictions[index]}"
    end

    File.delete(new_csv_filename) if File.exist?(new_csv_filename)
  end

  # Test evaluation metrics with raw arrays
  def test_evaluation_metrics_with_arrays
    x_values = [1, 2, 3, 4, 5]
    y_values = [3, 5, 7, 9, 11]

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
    @model.fit(dataset: @dataset, feature_column: "Feature", target_column: "Target")
    # Predict using feature values
    predictions = @model.predict([1, 2, 3, 4, 5])
    y_values = [3, 5, 7, 9, 11]

    mse = @model.mean_squared_error(y_values, predictions)
    r2 = @model.r_squared(y_values, predictions)

    assert_in_delta 0.0, mse, 0.01, "MSE should be close to 0.0"
    assert_in_delta 1.0, r2, 0.01, "R-squared should be close to 1.0"
  end
end
