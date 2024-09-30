import unittest
from unittest.mock import Mock, AsyncMock, patch
import os
from optimizer.prompt_evaluator import Evaluator
from optimizer.model_interface import Model
from utils.performance_logger import PerformanceLogger


class TestEvaluator(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Change to the project root directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def setUp(self):
        # Create a mock PerformanceLogger
        self.mock_logger = Mock(spec=PerformanceLogger)

        self.evaluator = Evaluator(logger=self.mock_logger)
        self.evaluator.data_loader = Mock()
        self.evaluator.data_loader.load_data.return_value = [
            {
                'variables': {'text': 'Sample text'},
                'expected_output': 'Expected output'
            }
        ]
        self.evaluator.dataset = self.evaluator.data_loader.load_data()

    async def test_evaluate_prompts(self):
        prompts = ['Summarize: {text}', 'Paraphrase: {text}']
        generator_model = Model('test-model')

        # Mock evaluate_output to return a dictionary for this test
        self.evaluator.evaluate_output = AsyncMock(return_value={
            'is_correct': False,
            'explanation_success': '',
            'explanation_failure': 'Test failure explanation'
        })

        result = await self.evaluator.evaluate_prompts(prompts, generator_model, iteration=1)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['prompt'], 'Summarize: {text}')
        self.assertEqual(result[0]['total_cases'], 1)
        self.assertEqual(result[0]['correct_answers'], 0)
        self.assertEqual(result[0]['score'], 0.0)

    async def test_evaluate_prompt(self):
        prompt = 'Summarize: {text}'
        generator_model = Model('test-model')

        # Mock evaluate_output to return a dictionary
        self.evaluator.evaluate_output = AsyncMock(return_value={
            'is_correct': False,
            'explanation_success': '',
            'explanation_failure': 'Test failure explanation'
        })

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

        # Mock the generate method of the evaluation model
        evaluation_model.generate = AsyncMock(return_value={
            'function': {
                'name': 'evaluate_semantic_equivalence',
                'data': {
                    'is_equivalent': True,
                    'explanation_success': 'Test success explanation',
                    'explanation_failure': ''
                }
            }
        })

        self.evaluator.evaluation_model = evaluation_model

        result = await self.evaluator.evaluate_output('Model output', 'Expected output')

        self.assertIsInstance(result, dict)
        self.assertIn('is_correct', result)
        self.assertIn('explanation_success', result)
        self.assertIn('explanation_failure', result)
        self.assertTrue(result['is_correct'])
        self.assertEqual(result['explanation_success'], 'Test success explanation')
        self.assertEqual(result['explanation_failure'], '')


if __name__ == '__main__':
    unittest.main()
