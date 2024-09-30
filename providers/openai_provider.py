from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import OPENAI_API_KEY
from .base_provider import BaseProvider, ProviderResponse, FunctionCall
import json


class OpenAIProvider(BaseProvider):
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)

    def get_provider_name(self) -> str:
        return self.__class__.__name__

    async def generate(self,
                       model_name: str,
                       prompt: str,
                       temperature: float = 0.7,
                       functions: Optional[List[Dict[str, Any]]] = None,
                       function_call: Optional[Dict[str, str]] = None) -> ProviderResponse:
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                functions=functions,
                function_call=function_call
            )

            # Extract the relevant information from the response
            message = response.choices[0].message

            if message.function_call:
                # If there's a function call, parse it
                function_data = {
                    "name": message.function_call.name,
                    "data": json.loads(message.function_call.arguments)
                }
                return self.create_response(
                    function=function_data,
                    raw_response=response.dict()
                )
            else:
                # If there's no function call, return the message content
                return self.create_response(
                    message=message.content,
                    raw_response=response.dict()
                )

        except Exception as e:
            # If there's an error, return it in the standardized format
            return self.create_response(
                error=str(e),
                raw_response=str(e)
            )
