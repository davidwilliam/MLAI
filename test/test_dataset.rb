# frozen_string_literal: true

require "test_helper"

class TestDataset < Minitest::Test
  def setup
    @csv_content = <<~CSV
      Feature1,Feature2,Target
      1,2,3
      4,5,6
      7,8,9
    CSV
    @csv_filename = "test_dataset.csv"
    File.write(@csv_filename, @csv_content)
  end

  def teardown
    File.delete(@csv_filename) if File.exist?(@csv_filename)
  end

  def test_dataset_loading
    dataset = MLAI::Dataset.new(@csv_filename)

    # Check headers
    assert_equal ["Feature1", "Feature2", "Target"], dataset.headers

    # Check data
    expected_data = [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]
    ]
    assert_equal expected_data, dataset.data
  end
end
