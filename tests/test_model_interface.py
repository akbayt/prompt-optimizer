import unittest
from optimizer.model_interface import Model


class TestModelInterface(unittest.IsolatedAsyncioTestCase):
    async def test_openai_model_generation(self):
        # Arrange
        model = Model("gpt-4o-mini")

        # Act
        result = await model.generate("say 'hi'")

        # Assert
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, str))
        self.assertTrue(len(result) > 0)
        print(f"OpenAI response: {result}")

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
