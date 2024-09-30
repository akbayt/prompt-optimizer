from typing import List, Tuple, Dict, Any
from config import PROMPT_GENERATOR_MODEL
from optimizer.model_interface import Model


class PromptGenerator:
    def __init__(self, prompt_generator_model=PROMPT_GENERATOR_MODEL):
        self.model = Model(prompt_generator_model)

    async def generate_suggestions(self, prompts_and_scores: List[Tuple[str, float]], num_suggestions: int = 3) -> Dict[
        str, Any]:
        # Prepare the input for the LLM
        prompt = self._prepare_prompt(prompts_and_scores, num_suggestions)

        # Define the function for the model to call
        functions = [
            {
                "name": "analyze_and_suggest_prompts",
                "description": "Analyze given prompts and suggest improvements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "object",
                            "properties": {
                                "high_performance_factors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Factors contributing to high performance in prompts"
                                },
                                "low_performance_factors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Factors contributing to low performance in prompts"
                                }
                            },
                            "required": ["high_performance_factors", "low_performance_factors"]
                        },
                        "suggestions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "prompt": {"type": "string"},
                                    "explanation": {"type": "string"}
                                },
                                "required": ["prompt", "explanation"]
                            },
                            "description": f"List of {num_suggestions} suggested improved prompts with explanations"
                        }
                    },
                    "required": ["analysis", "suggestions"]
                }
            }
        ]

        # Generate analysis and suggestions using the LLM with function calling
        response = await self.model.generate(prompt, functions=functions,
                                             function_call={"name": "analyze_and_suggest_prompts"})

        # Extract the function call result
        if response.get('function') and response['function']['name'] == "analyze_and_suggest_prompts":
            result = response['function']['data']
        else:
            print("Error: Unexpected response format from the model")
            return {}

        return result

    def _prepare_prompt(self, prompts_and_scores: List[Tuple[str, float]], num_suggestions: int) -> str:
        prompt_list = "\n".join(
            [f"<Item>\n\t<Prompt>\n\t\t{prompt}\n\t</Prompt>\n\t<Score>\n\t\t{score}\n\t</Score>\n</Item>" for
             prompt, score in prompts_and_scores])

        return f"""Analyze the following list of prompts and their performance scores, then generate new suggestions:

{prompt_list}

Identify factors contributing to high and low performance. Then, generate {num_suggestions} new prompt variations that aim to improve upon the highest-scoring prompts, incorporating insights from your analysis.

Use the 'analyze_and_suggest_prompts' function to structure your response.
"""
