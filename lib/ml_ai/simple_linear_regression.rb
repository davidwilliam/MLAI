# frozen_string_literal: true

module MLAI
  class SimpleLinearRegression
    attr_reader :slope, :intercept

    def initialize
      @slope = nil
      @intercept = nil
    end

    def fit(x_values, y_values)
      raise "Input arrays must have the same length" unless x_values.length == y_values.length

      n = x_values.length
      sum_x = x_values.sum
      sum_y = y_values.sum
      sum_x_squared = x_values.map { |x| x**2 }.sum
      sum_xy = x_values.each_with_index.map { |x, i| x * y_values[i] }.sum

      @slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
      @intercept = (sum_y - @slope * sum_x) / n
    end

    def predict(x_values)
      raise "Model has not been fitted yet" if @slope.nil? || @intercept.nil?

      x_values.map { |x| @slope * x + @intercept }
    end
  end
end
  