# frozen_string_literal: true

require 'matrix'
require_relative 'dataset'

module MLAI
  class MultipleLinearRegression
    attr_reader :coefficients, :intercept, :regularization

    def initialize(alpha = 1e-8, regularization: 0.0)
      @coefficients = nil
      @intercept = nil
      @alpha = alpha # Small value to avoid singular matrix in inversion
      @regularization = regularization # Regularization strength for Ridge Regression
    end

    # Fit method accepts either x_values and y_values or a Dataset object with specified columns
    def fit(x_values: nil, y_values: nil, dataset: nil, feature_columns: nil, target_column: nil)
        if dataset
          # Extract feature and target columns from the dataset
          feature_indices = feature_columns.map { |col| dataset.headers.index(col) }
          target_index = dataset.headers.index(target_column)
      
          x_values = dataset.data.map { |row| feature_indices.map { |i| row[i] } }
          y_values = dataset.data.map { |row| row[target_index] }
        end
      
        raise "Input arrays must have the same length" unless x_values.length == y_values.length
      
        # Convert x_values to a matrix and add a column of ones for the intercept
        x_matrix = Matrix[*x_values.map { |x| [1] + x }]
        y_vector = Vector.elements(y_values)
      
        # Calculate coefficients using the normal equation with regularization: (X^T * X + λI)^-1 * X^T * Y
        x_transpose = x_matrix.transpose
        regularization_matrix = Matrix.build(x_matrix.column_count) { |i, j| i == j ? @regularization : 0 }
        
        xtx = x_transpose * x_matrix + regularization_matrix
      
        begin
          theta = xtx.inverse * x_transpose * y_vector
        rescue ExceptionForMatrix::ErrNotRegular
          raise "Matrix is singular or nearly singular, consider increasing regularization"
        end
      
        @intercept = theta[0]
        @coefficients = theta.to_a[1..-1]
    end
      
    def predict(x_values)
      raise "Model has not been fitted yet" if @coefficients.nil? || @intercept.nil?

      x_values.map do |x|
        @coefficients.each_with_index.reduce(@intercept) do |sum, (coef, i)|
          sum + coef * x[i]
        end
      end
    end

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
