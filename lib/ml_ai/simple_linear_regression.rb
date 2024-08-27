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
      sum_x_squared = x_values.map { |x| x ** 2 }.sum
      sum_xy = x_values.each_with_index.map { |x, i| x * y_values[i] }.sum
    
      @slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2).to_f
      @intercept = (sum_y - @slope * sum_x) / n.to_f
    end    
  
    def predict(x_values)
      raise "Model has not been fitted yet" if @slope.nil? || @intercept.nil?
  
      x_values.map { |x| @slope * x + @intercept }
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
  