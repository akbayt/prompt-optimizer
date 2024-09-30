import unittest
import json
import os
import tempfile
from utils.data_loader import DataLoader, InvalidDataPointError


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test_dataset.json")

        # Sample data for testing
        self.test_data = [
            {
                "variables": {"text": "This is a test sentence.", "language": "english"},
                "expected_output": "A test sentence."
            },
            {
                "variables": {"text": "Another example for testing.", "language": "english"},
                "expected_output": "An example for testing."
            }
        ]

        # Write test data to the temporary file
        with open(self.temp_file, 'w') as f:
            json.dump(self.test_data, f)

        self.current_prompt = {
            "text": "Summarize the following {language} text in one sentence: {text}",
            "variables": ["text", "language"]
        }

    def tearDown(self):
        # Remove the temporary directory and its contents
        os.remove(self.temp_file)
        os.rmdir(self.temp_dir)

    def test_load_data_all_valid(self):
        loader = DataLoader(self.temp_dir, os.path.basename(self.temp_file), self.current_prompt)
        loaded_data = loader.load_data()
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data, self.test_data)

    def test_load_data_invalid_data_point(self):
        invalid_data = self.test_data + [{
            "variables": {"text": "Missing language key."},
            "expected_output": "This should cause an error."
        }]
        with open(self.temp_file, 'w') as f:
            json.dump(invalid_data, f)

        loader = DataLoader(self.temp_dir, os.path.basename(self.temp_file), self.current_prompt)
        with self.assertRaises(InvalidDataPointError) as context:
            loader.load_data()
        self.assertIn("Missing required variables", str(context.exception))

    def test_load_data_missing_expected_output(self):
        invalid_data = self.test_data + [{
            "variables": {"text": "Valid text", "language": "english"}
        }]
        with open(self.temp_file, 'w') as f:
            json.dump(invalid_data, f)

        loader = DataLoader(self.temp_dir, os.path.basename(self.temp_file), self.current_prompt)
        with self.assertRaises(InvalidDataPointError) as context:
            loader.load_data()
        self.assertIn("Missing expected output", str(context.exception))

    def test_missing_file(self):
        loader = DataLoader(self.temp_dir, "non_existent_file.json", self.current_prompt)
        with self.assertRaises(FileNotFoundError):
            loader.load_data()

    def test_missing_folder(self):
        non_existent_dir = os.path.join(self.temp_dir, "non_existent_folder")
        loader = DataLoader(non_existent_dir, "test_dataset.json", self.current_prompt)
        with self.assertRaises(FileNotFoundError):
            loader.load_data()

    def test_changed_current_prompt(self):
        new_prompt = {
            "text": "Translate the following text to {language}: {text}",
            "variables": ["text", "language"]
        }
        loader = DataLoader(self.temp_dir, os.path.basename(self.temp_file), new_prompt)
        loaded_data = loader.load_data()
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data, self.test_data)


if __name__ == '__main__':
    unittest.main()
