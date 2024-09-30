import unittest
from unittest.mock import Mock, AsyncMock
import json
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
            ("Summarize this text: {text}", 0.7),
            ("Give a brief summary of: {text}", 0.8),
            ("Provide a concise summary for the following: {text}", 0.6)
        ]
        num_suggestions = 2

        mock_response = json.dumps({
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
        })
        self.mock_model.generate.return_value = mock_response

        # Act
        result = await self.generator.generate_suggestions(prompts_and_scores, num_suggestions)

        # Assert
        self.mock_model.generate.assert_called_once()
        self.assertIn("analysis", result)
        self.assertIn("suggestions", result)
        self.assertIn("explanations", result)
        self.assertEqual(len(result["suggestions"]), num_suggestions)
        self.assertEqual(len(result["explanations"]), num_suggestions)

    def test_prepare_prompt(self):
        # Arrange
        prompts_and_scores = [
            ("Summarize this text: {text}", 0.7),
            ("Give a brief summary of: {text}", 0.8)
        ]
        num_suggestions = 2

        # Act
        prepared_prompt = self.generator._prepare_prompt(prompts_and_scores, num_suggestions)

        # Assert
        self.assertIn("Summarize this text: {text}", prepared_prompt)
        self.assertIn("Give a brief summary of: {text}", prepared_prompt)
        self.assertIn("generate 2 new prompt variations", prepared_prompt)

    def test_parse_response_valid(self):
        # Arrange
        mock_response = json.dumps({
            "analysis": {
                "high_performance_factors": ["Factor 1", "Factor 2"],
                "low_performance_factors": ["Factor 3", "Factor 4"]
            },
            "suggestions": [
                {"prompt": "Prompt 1", "explanation": "Explanation 1"},
                {"prompt": "Prompt 2", "explanation": "Explanation 2"}
            ]
        })

        # Act
        result = self.generator._parse_response(mock_response, 2)

        # Assert
        self.assertIn("analysis", result)
        self.assertIn("suggestions", result)
        self.assertIn("explanations", result)
        self.assertEqual(len(result["suggestions"]), 2)
        self.assertEqual(len(result["explanations"]), 2)

    def test_parse_response_invalid_json(self):
        # Arrange
        mock_response = "Invalid JSON"

        # Act
        result = self.generator._parse_response(mock_response, 2)

        # Assert
        self.assertEqual(result, {})

    def test_parse_response_missing_keys(self):
        # Arrange
        mock_response = json.dumps({"invalid": "structure"})

        # Act
        result = self.generator._parse_response(mock_response, 2)

        # Assert
        self.assertEqual(result, {})

    async def test_non_mocked_generator(self):
        self.prompt_generator = PromptGenerator()
        prompts_and_scores = [
            ("Summarize this text: {text}", 0.7),
            ("Give a brief summary of: {text}", 0.8),
            ("Provide a concise summary for the following: {text}", 0.6)
        ]
        num_suggestions = 2

        # Act
        result = await self.prompt_generator.generate_suggestions(prompts_and_scores, num_suggestions)

        # Assert
        self.assertIn("analysis", result)
        self.assertIn("high_performance_factors", result["analysis"])
        self.assertIn("low_performance_factors", result["analysis"])
        self.assertIn("suggestions", result)
        self.assertIn("explanations", result)
        self.assertEqual(len(result["suggestions"]), num_suggestions)
        self.assertEqual(len(result["explanations"]), num_suggestions)


if __name__ == '__main__':
    unittest.main()
