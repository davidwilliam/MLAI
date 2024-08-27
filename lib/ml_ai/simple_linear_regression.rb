# frozen_string_literal: true

require 'matrix'
require_relative 'dataset'

module MLAI
  class SimpleLinearRegression
    attr_reader :slope, :intercept

    def initialize
      @slope = nil
      @intercept = nil
    end

    # Fit method accepts either x_values and y_values or a Dataset object with specified columns
    def fit(x_values: nil, y_values: nil, dataset: nil, feature_column: nil, target_column: nil)
      if dataset
        unless feature_column && target_column
          raise ArgumentError, "When using a Dataset, you must specify feature_column and target_column"
        end

        # Extract indices of the specified columns
        feature_index = dataset.headers.index(feature_column)
        target_index = dataset.headers.index(target_column)

        unless feature_index && target_index
          raise ArgumentError, "Specified feature or target column does not exist in the dataset"
        end

        # Extract x and y values from the dataset
        x_values = dataset.data.map { |row| row[feature_index] }
        y_values = dataset.data.map { |row| row[target_index] }
      elsif x_values && y_values
        # Use x_values and y_values directly
      else
        raise ArgumentError, "You must provide either x_values and y_values or a dataset with feature_column and target_column"
      end

      # Ensure that x and y have the same number of observations
      unless x_values.length == y_values.length
        raise ArgumentError, "Input arrays must have the same length"
      end

      # Calculate means
      mean_x = x_values.sum / x_values.length.to_f
      mean_y = y_values.sum / y_values.length.to_f

      # Calculate the numerator and denominator for the slope
      numerator = 0.0
      denominator = 0.0

      x_values.each_with_index do |x, i|
        numerator += (x - mean_x) * (y_values[i] - mean_y)
        denominator += (x - mean_x) ** 2
      end

      @slope = numerator / denominator
      @intercept = mean_y - @slope * mean_x
    end

    # Predict method remains unchanged, accepts an array of x_values
    def predict(x_values)
      raise "Model has not been fitted yet" if @slope.nil? || @intercept.nil?

      x_values.map { |x| @slope * x + @intercept }
    end

    # Evaluation Metrics
    def mean_squared_error(y_true, y_pred)
      raise "Input arrays must have the same length" unless y_true.length == y_pred.length

      n = y_true.length
      sum_squared_errors = y_true.each_with_index.map { |y, i| (y - y_pred[i]) ** 2 }.sum
      sum_squared_errors / n.to_f
    end

    def r_squared(y_true, y_pred)
      raise "Input arrays must have the same length" unless y_true.length == y_pred.length

      mean_y = y_true.sum / y_true.length.to_f
      ss_total = y_true.map { |y| (y - mean_y) ** 2 }.sum
      ss_residual = y_true.each_with_index.map { |y, i| (y - y_pred[i]) ** 2 }.sum

      1 - (ss_residual / ss_total.to_f)
    end
  end
end
