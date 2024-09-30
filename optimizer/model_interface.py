import asyncio
from typing import Dict, Any, List, Optional
from providers.openai_provider import OpenAIProvider
from providers.base_provider import ProviderResponse
from config import MAX_RETRIES, RETRY_DELAY


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

    async def generate(self,
                       prompt: str,
                       functions: Optional[List[Dict[str, Any]]] = None,
                       function_call: Optional[Dict[str, str]] = None) -> ProviderResponse:
        for attempt in range(MAX_RETRIES):
            try:
                return await self.provider.generate(
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    functions=functions,
                    function_call=function_call
                )
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Error generating model output: {str(e)}. Retrying in {RETRY_DELAY} seconds...")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"Failed to generate model output after {MAX_RETRIES} attempts. Error: {str(e)}")
                    return self.provider.create_response(error=str(e))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "provider": self.provider.get_provider_name()
        }
