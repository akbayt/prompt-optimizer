from abc import ABC, abstractmethod
from typing import Optional, Any, TypedDict, List, Dict


class FunctionCall(TypedDict):
    name: str
    data: Any


class ProviderResponse(TypedDict):
    function: Optional[FunctionCall]
    message: Optional[str]
    error: Optional[str]
    raw_response: Any
    provider: str


class BaseProvider(ABC):
    @abstractmethod
    async def generate(self,
                       model_name: str,
                       prompt: str,
                       temperature: float = 0.7,
                       functions: Optional[List[Dict[str, Any]]] = None,
                       function_call: Optional[Dict[str, str]] = None) -> ProviderResponse:
        """
        Generate a response from the AI model.

        Args:
            model_name (str): The name of the model to use.
            prompt (str): The input prompt for the model.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.
            functions (Optional[List[Dict[str, Any]]], optional): List of function specifications. Defaults to None.
            function_call (Optional[Dict[str, str]], optional): Function call specifications. Defaults to None.

        Returns:
            ProviderResponse: A dictionary containing the response details.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the provider.

        Returns:
            str: The name of the provider.
        """
        pass

    def create_response(self,
                        function: Optional[FunctionCall] = None,
                        message: Optional[str] = None,
                        error: Optional[str] = None,
                        raw_response: Any = None) -> ProviderResponse:
        """
        Create a standardized response dictionary.

        Args:
            function (Optional[FunctionCall], optional): Function call details. Defaults to None.
            message (Optional[str], optional): Response message. Defaults to None.
            error (Optional[str], optional): Error message. Defaults to None.
            raw_response (Any, optional): Raw response from the provider. Defaults to None.

        Returns:
            ProviderResponse: A dictionary containing the response details.
        """
        return {
            "function": function,
            "message": message,
            "error": error,
            "raw_response": raw_response,
            "provider": self.get_provider_name()
        }
