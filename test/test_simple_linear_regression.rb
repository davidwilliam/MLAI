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
end