import unittest
import warnings
from optimizer.model_interface import Model

# Suppress the specific Pydantic warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic",
    message="The `dict` method is deprecated; use `model_dump` instead."
)


class TestModelInterface(unittest.IsolatedAsyncioTestCase):
    async def test_openai_model_generation_message(self):
        # Arrange
        model = Model("gpt-4o-mini")

        # Act
        result = await model.generate("say only 'hi'. No other word. only 'hi'")

        # Assert
        self.assertIsNotNone(result)
        self.assertIn('message', result)
        self.assertIsInstance(result['message'], str)
        self.assertTrue(len(result['message']) > 0)
        self.assertIsNone(result['error'])
        print(f"OpenAI message response: {result['message']}")

    async def test_openai_model_generation_function_calling(self):
        # Arrange
        model = Model("gpt-4o-mini")
        functions = [
            {
                "name": "sum",
                "description": "Sum two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "The first number"},
                        "b": {"type": "number", "description": "The second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        ]

        # Act
        result = await model.generate("sum 2 and 3", functions=functions, function_call={"name": "sum"})

        # Assert
        self.assertIsNotNone(result)
        self.assertIn('function', result)
        self.assertIsNotNone(result['function'])
        self.assertEqual(result['function']['name'], 'sum')
        self.assertIn('data', result['function'])
        self.assertIn('a', result['function']['data'])
        self.assertIn('b', result['function']['data'])
        self.assertEqual(result['function']['data']['a'], 2)
        self.assertEqual(result['function']['data']['b'], 3)
        self.assertIsNone(result['error'])
        print(f"OpenAI function calling response: {result['function']}")

    def test_model_info(self):
        # Arrange
        model = Model("gpt-4o-mini", temperature=0.5)

        # Act
        info = model.get_model_info()

        # Assert
        self.assertEqual(info['model_name'], "gpt-4o-mini")
        self.assertEqual(info['temperature'], 0.5)
        self.assertEqual(info['provider'], "OpenAIProvider")


if __name__ == '__main__':
    unittest.main()
