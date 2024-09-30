import unittest
from unittest.mock import Mock, AsyncMock
from optimizer.prompt_generator import PromptGenerator


class TestPromptGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.generate = AsyncMock()
        self.generator = PromptGenerator()
        self.generator.model = self.mock_model

    async def test_generate_suggestions(self):
        # Arrange
        prompts_and_scores = [
            ("Summarize this text: {text}", "This prompt is concise and direct.", 0.7),
            ("Give a brief summary of: {text}", "This prompt is clear but could be more specific.", 0.8),
            ("Provide a concise summary for the following: {text}", "This prompt is slightly verbose.", 0.6)
        ]
        num_suggestions = 2

        mock_response = {
            'function': {
                'name': 'analyze_and_suggest_prompts',
                'data': {
                    "analysis": {
                        "high_performance_factors": [
                            "Clear and direct instructions",
                            "Use of action verbs"
                        ],
                        "low_performance_factors": [
                            "Overly verbose phrasing",
                            "Lack of specificity"
                        ]
                    },
                    "suggestions": [
                        {
                            "prompt": "Concisely summarize this text: {text}",
                            "explanation": "This prompt is clear and uses an action verb."
                        },
                        {
                            "prompt": "Provide a one-sentence summary of: {text}",
                            "explanation": "This prompt is specific and direct."
                        }
                    ]
                }
            }
        }
        self.mock_model.generate.return_value = mock_response

        # Act
        result = await self.generator.generate_suggestions(prompts_and_scores, num_suggestions)

        # Assert
        self.mock_model.generate.assert_called_once()
        self.assertIn("analysis", result)
        self.assertIn("suggestions", result)
        self.assertEqual(len(result["suggestions"]), num_suggestions)
        self.assertIn("high_performance_factors", result["analysis"])
        self.assertIn("low_performance_factors", result["analysis"])

    def test_prepare_prompt(self):
        # Arrange
        prompts_and_scores = [
            ("Summarize this text: {text}", "This prompt is concise and direct.", 0.7),
            ("Give a brief summary of: {text}", "This prompt is clear but could be more specific.", 0.8)
        ]
        num_suggestions = 2

        # Act
        prepared_prompt = self.generator._prepare_prompt(prompts_and_scores, num_suggestions)

        # Assert
        self.assertIn("Summarize this text: {text}", prepared_prompt)
        self.assertIn("Give a brief summary of: {text}", prepared_prompt)
        self.assertIn("This prompt is concise and direct.", prepared_prompt)
        self.assertIn("This prompt is clear but could be more specific.", prepared_prompt)
        self.assertIn("generate 2 new prompt variations", prepared_prompt)
        self.assertIn("Use the 'analyze_and_suggest_prompts' function", prepared_prompt)

    async def test_generate_suggestions_error_handling(self):
        # Arrange
        prompts_and_scores = [
            ("Summarize this text: {text}", "This prompt is concise and direct.", 0.7),
            ("Give a brief summary of: {text}", "This prompt is clear but could be more specific.", 0.8)
        ]
        num_suggestions = 2

        mock_response = {
            'message': 'Some unexpected response'
        }
        self.mock_model.generate.return_value = mock_response

        # Act
        result = await self.generator.generate_suggestions(prompts_and_scores, num_suggestions)

        # Assert
        self.assertEqual(result, {})

    async def test_non_mocked_generator(self):
        self.prompt_generator = PromptGenerator()
        prompts_and_scores = [
            ("Summarize this text: {text}", "This prompt is concise and direct.", 0.7),
            ("Give a brief summary of: {text}", "This prompt is clear but could be more specific.", 0.8),
            ("Provide a concise summary for the following: {text}", "This prompt is slightly verbose.", 0.6)
        ]
        num_suggestions = 2

        # Act
        result = await self.prompt_generator.generate_suggestions(prompts_and_scores, num_suggestions)

        # Assert
        self.assertIn("analysis", result)
        self.assertIn("high_performance_factors", result["analysis"])
        self.assertIn("low_performance_factors", result["analysis"])
        self.assertIn("suggestions", result)
        self.assertEqual(len(result["suggestions"]), num_suggestions)


if __name__ == '__main__':
    unittest.main()
