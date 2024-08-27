# frozen_string_literal: true

require "test_helper"

class TestSimpleLinearRegression < Minitest::Test
    def test_simple_linear_regression
        model = MLAI::SimpleLinearRegression.new
        model.fit([1, 2, 3], [2, 4, 6])
        prediction = model.predict([4])
        assert_equal [8], prediction
    end

    def test_simple_linear_regression_large_dataset
        x_values = (1..100).to_a
        y_values = x_values.map { |x| 3 * x + 5 }
        model = MLAI::SimpleLinearRegression.new
        model.fit(x_values, y_values)
        prediction = model.predict([150])
        assert_equal [455], prediction
    end

    def test_simple_linear_regression_with_negative_values
        x_values = [-10, -5, 0, 5, 10]
        y_values = x_values.map { |x| 2 * x - 3 }
        model = MLAI::SimpleLinearRegression.new
        model.fit(x_values, y_values)
        prediction = model.predict([15])
        assert_equal [27], prediction
    end

    def test_simple_linear_regression_with_evaluation_metrics
        model = MLAI::SimpleLinearRegression.new
      
        # Fit the model to the data
        x_values = [1, 2, 3, 4, 5]
        y_values = [2, 4, 5, 4, 5]
        model.fit(x_values, y_values)
      
        # Predictions
        predictions = model.predict(x_values)
      
        # Expected predictions based on the fit
        expected_predictions = [2.8, 3.4, 4.0, 4.6, 5.2]
      
        # Assertions for predictions
        predictions.each_with_index do |pred, index|
          assert_in_delta expected_predictions[index], pred, 0.01, "Prediction for x = #{x_values[index]} should be close to #{expected_predictions[index]}"
        end
      
        # Calculate evaluation metrics
        mse = model.mean_squared_error(y_values, predictions)
        r2 = model.r_squared(y_values, predictions)
      
        # Updated expected values based on correct calculations
        expected_mse = 0.48
        expected_r2 = 0.6
      
        # Assertions for metrics
        assert_in_delta expected_mse, mse, 0.01, "MSE should be close to #{expected_mse}"
        assert_in_delta expected_r2, r2, 0.01, "R-squared should be close to #{expected_r2}"
      end      
        
end