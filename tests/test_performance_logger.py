import unittest
import os
import json
import tempfile
from utils.performance_logger import PerformanceLogger, create_logger


class TestPerformanceLogger(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test log files
        self.test_dir = tempfile.mkdtemp()
        self.original_prompt = "This is a test prompt: {text}"
        self.logger = PerformanceLogger(self.test_dir, self.original_prompt)

    def tearDown(self):
        # Clean up the temporary directory after tests
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_initialization(self):
        self.assertEqual(self.logger.log_data["original_prompt"], self.original_prompt)
        self.assertIsNone(self.logger.log_data["optimized_prompt"])
        self.assertEqual(self.logger.log_data["optimization_logs"], [])
        self.assertTrue(os.path.exists(self.logger.log_file))

    def test_log_iteration(self):
        prompts = ["Test prompt 1", "Test prompt 2"]
        evaluations = [
            {"prompt": "Test prompt 1", "score": 0.8},
            {"prompt": "Test prompt 2", "score": 0.9}
        ]
        self.logger.log_iteration(1, prompts, evaluations)

        with open(self.logger.log_file, 'r') as f:
            log_data = json.load(f)

        self.assertEqual(len(log_data["optimization_logs"]), 1)
        self.assertEqual(log_data["optimization_logs"][0]["iteration"], 1)
        self.assertEqual(log_data["optimization_logs"][0]["prompts"], prompts)
        self.assertEqual(log_data["optimization_logs"][0]["evaluations"], evaluations)

    def test_log_optimized_prompt(self):
        optimized_prompt = "This is an optimized test prompt: {text}"
        self.logger.log_optimized_prompt(optimized_prompt)

        with open(self.logger.log_file, 'r') as f:
            log_data = json.load(f)

        self.assertEqual(log_data["optimized_prompt"], optimized_prompt)

    def test_multiple_iterations(self):
        for i in range(3):
            prompts = [f"Test prompt {j} for iteration {i + 1}" for j in range(2)]
            evaluations = [
                {"prompt": prompts[0], "score": 0.8 + i * 0.05},
                {"prompt": prompts[1], "score": 0.9 + i * 0.05}
            ]
            self.logger.log_iteration(i + 1, prompts, evaluations)

        with open(self.logger.log_file, 'r') as f:
            log_data = json.load(f)

        self.assertEqual(len(log_data["optimization_logs"]), 3)
        self.assertEqual(log_data["optimization_logs"][2]["iteration"], 3)

    def test_file_creation(self):
        self.assertTrue(os.path.exists(self.logger.log_file))

        # Check if the file is created with correct initial content
        with open(self.logger.log_file, 'r') as f:
            log_data = json.load(f)

        self.assertEqual(log_data["original_prompt"], self.original_prompt)
        self.assertIsNone(log_data["optimized_prompt"])
        self.assertEqual(log_data["optimization_logs"], [])

    def test_create_logger_function(self):
        logger = create_logger(self.test_dir, self.original_prompt)
        self.assertIsInstance(logger, PerformanceLogger)
        self.assertEqual(logger.log_data["original_prompt"], self.original_prompt)
        self.assertTrue(os.path.exists(logger.log_file))

    def test_get_log_file_path(self):
        path = self.logger.get_log_file_path()
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.startswith(self.test_dir))
        self.assertTrue(path.endswith('.json'))


if __name__ == '__main__':
    unittest.main()
