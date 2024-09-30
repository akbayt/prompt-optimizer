import unittest
from unittest.mock import Mock, AsyncMock, patch
import os
import tempfile
import json
from optimizer.prompt_evaluator import Evaluator
from optimizer.model_interface import Model


class TestEvaluator(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Change to the project root directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def setUp(self):
        # Create a temporary file for logging
        self.temp_log_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.jsonl')
        self.temp_log_file.close()

        self.evaluator = Evaluator(log_file=self.temp_log_file.name)
        self.evaluator.data_loader = Mock()
        self.evaluator.data_loader.load_data.return_value = [
            {
                'variables': {'text': 'Sample text'},
                'expected_output': 'Expected output'
            }
        ]
        self.evaluator.dataset = self.evaluator.data_loader.load_data()

    def tearDown(self):
        # Remove the temporary log file
        os.unlink(self.temp_log_file.name)

    async def test_evaluate_prompts(self):
        prompts = ['Summarize: {text}', 'Paraphrase: {text}']
        generator_model = Model('test-model')

        # Mock evaluate_output to always return False for this test
        self.evaluator.evaluate_output = AsyncMock(return_value=False)

        result = await self.evaluator.evaluate_prompts(prompts, generator_model, iteration=1)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['prompt'], 'Summarize: {text}')
        self.assertEqual(result[0]['total_cases'], 1)
        self.assertEqual(result[0]['correct_answers'], 0)
        self.assertEqual(result[0]['score'], 0.0)

        # Check if log file was created and contains the correct data
        with open(self.temp_log_file.name, 'r') as f:
            log_content = f.read().strip()
            log_entry = json.loads(log_content)

            self.assertEqual(log_entry['iteration'], 1)
            self.assertEqual(len(log_entry['evaluations']), 2)

            # Check the first prompt's evaluation
            self.assertEqual(log_entry['evaluations'][0]['prompt'], 'Summarize: {text}')
            self.assertEqual(log_entry['evaluations'][0]['total_cases'], 1)
            self.assertEqual(log_entry['evaluations'][0]['correct_answers'], 0)
            self.assertEqual(log_entry['evaluations'][0]['score'], 0.0)

            # Check the second prompt's evaluation
            self.assertEqual(log_entry['evaluations'][1]['prompt'], 'Paraphrase: {text}')
            self.assertEqual(log_entry['evaluations'][1]['total_cases'], 1)
            self.assertEqual(log_entry['evaluations'][1]['correct_answers'], 0)
            self.assertEqual(log_entry['evaluations'][1]['score'], 0.0)

    async def test_evaluate_prompt(self):
        prompt = 'Summarize: {text}'
        generator_model = Model('test-model')
        result = await self.evaluator.evaluate_prompt(prompt, generator_model)

        self.assertEqual(result['prompt'], 'Summarize: {text}')
        self.assertEqual(result['total_cases'], 1)
        self.assertEqual(result['correct_answers'], 0)
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(len(result['results']), 1)

    async def test_generate_model_output(self):
        generator_model = Model('test-model')
        result = await self.evaluator.generate_model_output('Test prompt', generator_model)

        self.assertEqual(result, 'This is a dummy response for testing purposes.')

    async def test_evaluate_output(self):
        evaluation_model = Model('test-model')
        result = await self.evaluator.evaluate_output('Model output', 'Expected output')

        self.assertIsInstance(result, bool)

    def test_save_log(self):
        self.evaluator.log = [
            [{'prompt': 'Test prompt', 'score': 0.5}]
        ]

        self.evaluator.save_log(1)

        with open(self.temp_log_file.name, 'r') as f:
            log_content = f.read().strip()
            # Parse the JSON content
            log_entry = json.loads(log_content)

            # Check the structure and content
            self.assertEqual(log_entry['iteration'], 1)
            self.assertEqual(len(log_entry['evaluations']), 1)
            self.assertEqual(log_entry['evaluations'][0]['prompt'], 'Test prompt')
            self.assertEqual(log_entry['evaluations'][0]['score'], 0.5)

            # Check the formatting
            expected_format = '''{
  "iteration": 1,
  "evaluations": [
    {
      "prompt": "Test prompt",
      "score": 0.5
    }
  ]
}'''
            self.assertEqual(log_content, expected_format)

    def test_get_log(self):
        self.evaluator.log = [
            [{'prompt': 'Test prompt 1', 'score': 0.5}],
            [{'prompt': 'Test prompt 2', 'score': 0.7}]
        ]

        result = self.evaluator.get_log()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]['prompt'], 'Test prompt 1')
        self.assertEqual(result[1][0]['prompt'], 'Test prompt 2')


if __name__ == '__main__':
    unittest.main()
