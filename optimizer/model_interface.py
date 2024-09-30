from typing import Dict, Any
from providers.openai_provider import OpenAIProvider


class Model:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self._setup_provider()

    def _setup_provider(self):
        if self.model_name.startswith(("gpt", "text-davinci")):
            self.provider = OpenAIProvider()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    async def generate(self, prompt: str) -> str:
        return await self.provider.generate(self.model_name, prompt, self.temperature)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "provider": self.provider.__class__.__name__
        }
