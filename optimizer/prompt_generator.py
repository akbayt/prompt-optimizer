import json
from typing import List, Tuple, Dict
from config import PROMPT_GENERATOR_MODEL
from optimizer.model_interface import Model


class PromptGenerator:
    def __init__(self, prompt_generator_model=PROMPT_GENERATOR_MODEL):
        self.model = Model(prompt_generator_model)

    async def generate_suggestions(self, prompts_and_scores: List[Tuple[str, float]], num_suggestions: int = 3) -> Dict:
        # Prepare the input for the LLM
        prompt = self._prepare_prompt(prompts_and_scores, num_suggestions)

        # Generate analysis and suggestions using the LLM
        response = await self.model.generate(prompt)

        # Parse the response to extract analysis and suggestions
        result = self._parse_response(response, num_suggestions)

        return result

    def _prepare_prompt(self, prompts_and_scores: List[Tuple[str, float]], num_suggestions: int) -> str:
        prompt_list = "\n".join([f"<Item>\n\t<Prompt>\n\t\t{prompt}\n\t</Prompt>\n\t<Score>\n\t\t{score}\n\t</Score>\n</Item>\n" for prompt, score in prompts_and_scores])

        return f"""Given the following list of prompts and their performance scores:

{prompt_list}

Please provide an analysis of the prompts and generate new suggestions in the following JSON format:

{{
  "analysis": {{
    "high_performance_factors": [
      "Factor 1 contributing to high performance",
      "Factor 2 contributing to high performance",
      ...
    ],
    "low_performance_factors": [
      "Factor 1 contributing to low performance",
      "Factor 2 contributing to low performance",
      ...
    ]
  }},
  "suggestions": [
    {{
      "prompt": "Your first suggested prompt",
      "explanation": "Brief explanation of why this prompt might perform well"
    }},
    {{
      "prompt": "Your second suggested prompt",
      "explanation": "Brief explanation of why this prompt might perform well"
    }},
    ...
  ]
}}

In your analysis, identify factors for high and low performance. Then, generate {num_suggestions} new prompt variations that aim to improve upon the highest-scoring prompts, incorporating insights from your analysis.

Ensure that your response is a valid JSON object.
"""

    def _parse_response(self, response: str, num_suggestions: int) -> Dict:
        try:
            data = json.loads(response)
            result = {
                "analysis": {
                    "high_performance_factors": data["analysis"]["high_performance_factors"],
                    "low_performance_factors": data["analysis"]["low_performance_factors"]
                },
                "suggestions": [item["prompt"] for item in data["suggestions"][:num_suggestions]],
                "explanations": [item["explanation"] for item in data["suggestions"][:num_suggestions]]
            }
            return result
        except json.JSONDecodeError:
            print("Error: Failed to parse JSON response")
            return {}
        except KeyError:
            print("Error: JSON response does not have the expected structure")
            return {}