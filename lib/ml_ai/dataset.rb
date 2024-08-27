# frozen_string_literal: true

require 'csv'

module MLAI
  class Dataset
    attr_reader :data, :headers

    def initialize(filename)
      @filename = filename
      @data = []
      @headers = []

      load_csv
    end

    private

    def load_csv
      csv_data = CSV.read(@filename, headers: true)
      @headers = csv_data.headers
      @data = csv_data.map { |row| row.fields.map(&:to_f) } # Convert all fields to floats
    end
  end
end
