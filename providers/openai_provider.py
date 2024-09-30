import os
from openai import OpenAI
from dotenv import load_dotenv
from config import OPENAI_API_KEY

# Load environment variables
load_dotenv()


class OpenAIProvider:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)

    async def generate(self, model_name: str, prompt: str, temperature: float = 0.7):
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generating response with OpenAI: {str(e)}")
            return None
