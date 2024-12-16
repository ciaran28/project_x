import os
import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
from deltalake import DeltaTable
from data.transform import (
    data_cleaning,
    file_content_to_array,
    flag_identical_files,
    create_delta_table,
    read_delta_table,
    process_transcripts
)


class TestTransform(unittest.TestCase):

    def test_data_cleaning(self):
        sample_sentence = "Check out http://example.com #AI @user123!"
        expected_output = "check example ai"
        self.assertEqual(data_cleaning(sample_sentence), expected_output)

    @patch('os.listdir', return_value=['file1.txt', 'file2.txt'])
    @patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}")
    @patch('builtins.open', new_callable=mock_open, read_data='Sample content')
    def test_flag_identical_files(self, mock_open, mock_join, mock_listdir):
        # Test whether identical files are correctly identified
        with patch('builtins.print') as mock_print:
            flag_identical_files('mock_data_folder')
            mock_print.assert_called_with("No identical files found.")

    @patch('os.listdir', return_value=['file1.txt'])
    @patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}")
    @patch('builtins.open', new_callable=mock_open, read_data='File content')
    @patch('deltalake.writer.write_deltalake')
    def test_create_delta_table(self, mock_write_deltalake, mock_open, mock_join, mock_listdir):
        create_delta_table('mock_data_folder', 'mock_output_folder')
        mock_write_deltalake.assert_called_once()

    @patch.object(DeltaTable, 'to_pandas', return_value=pd.DataFrame({'file_content': ['Sample content']}))
    def test_read_delta_table(self, mock_to_pandas):
        df = read_delta_table('mock_output_folder')
        self.assertIn('file_content', df.columns)

    @patch('src.data.transform.flag_identical_files')
    @patch('src.data.transform.create_delta_table')
    def test_process_transcripts(self, mock_create_delta_table, mock_flag_identical_files):
        process_transcripts('mock_data_folder')
        mock_flag_identical_files.assert_called_once_with('mock_data_folder')
        mock_create_delta_table.assert_called_once_with('mock_data_folder', output_folder='data/sentiment_analysis/delta')


if __name__ == '__main__':
    unittest.main()