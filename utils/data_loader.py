import json
import os
from typing import List, Dict, Any
from config import DATASET_DIR, DATASET_FILE, ORIGINAL_PROMPT

class InvalidDataPointError(Exception):
    """Custom exception for invalid data points."""
    pass

class DataLoader:
    def __init__(self, dataset_dir=DATASET_DIR, dataset_file=DATASET_FILE, current_prompt=ORIGINAL_PROMPT):
        self.dataset_path = os.path.join(dataset_dir, dataset_file)
        self.current_prompt = current_prompt

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from the dataset file.

        Returns:
            List[Dict[str, Any]]: A list of valid data points, each containing variables and expected output.

        Raises:
            FileNotFoundError: If the dataset file is not found.
            InvalidDataPointError: If any data point is invalid.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        with open(self.dataset_path, 'r') as f:
            loaded_data = json.load(f)

        valid_data = []
        for index, data_point in enumerate(loaded_data):
            try:
                self._validate_data_point(data_point, index)
                valid_data.append(data_point)
            except InvalidDataPointError as e:
                raise InvalidDataPointError(f"Invalid data point at index {index}: {str(e)}") from e

        return valid_data

    def _validate_data_point(self, data_point: Dict[str, Any], index: int) -> None:
        """
        Validate that a data point has all required variables and an expected output.

        Args:
            data_point (Dict[str, Any]): A single data point.
            index (int): The index of the data point in the dataset.

        Raises:
            InvalidDataPointError: If the data point is invalid.
        """
        required_variables = set(self.current_prompt['variables'])
        data_point_variables = set(data_point.get('variables', {}).keys())

        if not required_variables.issubset(data_point_variables):
            missing_vars = required_variables - data_point_variables
            raise InvalidDataPointError(f"Missing required variables: {missing_vars}")

        if 'expected_output' not in data_point:
            raise InvalidDataPointError("Missing expected output")