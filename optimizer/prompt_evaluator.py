from typing import List, Dict, Any
from optimizer.model_interface import Model
from config import EVALUATION_MODEL, PERFORMANCE_LOG
from utils.data_loader import DataLoader
import json
import os


class Evaluator:
    def __init__(self, log_file: str = PERFORMANCE_LOG):
        self.evaluation_model = Model(EVALUATION_MODEL)
        self.data_loader = DataLoader()
        self.dataset = self.data_loader.load_data()
        self.log_file = log_file
        self.log = []

    async def evaluate_prompts(self, prompts: List[str], generator_model: Model, iteration: int) -> List[
        Dict[str, Any]]:
        evaluations = []
        for prompt in prompts:
            evaluation = await self.evaluate_prompt(prompt, generator_model)
            evaluations.append(evaluation)

        self.log.append(evaluations)
        self.save_log(iteration)
        return evaluations

    async def evaluate_prompt(self, prompt: str, generator_model: Model) -> Dict[str, Any]:
        total_cases = len(self.dataset)
        correct_answers = 0
        results = []

        for test_case in self.dataset:
            variables = test_case['variables']
            expected_output = test_case['expected_output']

            formatted_prompt = prompt.format(**variables)
            model_output = await self.generate_model_output(formatted_prompt, generator_model)

            is_correct = await self.evaluate_output(model_output, expected_output)

            if is_correct:
                correct_answers += 1

            results.append({
                'prompt': formatted_prompt,
                'model_output': model_output,
                'expected_output': expected_output,
                'is_correct': is_correct
            })

        return {
            'prompt': prompt,
            'total_cases': total_cases,
            'correct_answers': correct_answers,
            'score': correct_answers / total_cases,
            'results': results
        }

    async def generate_model_output(self, prompt: str, generator_model: Model) -> str:
        response = await generator_model.generate(prompt)
        return response.get('message', '')

    async def evaluate_output(self, model_output: str, expected_output: str) -> bool:
        evaluation_prompt = f"""
        Model output: {model_output}
        Expected output: {expected_output}

        Evaluate if the model output is semantically equivalent to the expected output.
        """

        evaluation_function = {
            "name": "evaluate_semantic_equivalence",
            "description": "Evaluate if two outputs are semantically equivalent",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_equivalent": {
                        "type": "boolean",
                        "description": "True if the outputs are semantically equivalent, False otherwise"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "A brief explanation of the evaluation"
                    }
                },
                "required": ["is_equivalent", "explanation"]
            }
        }

        response = await self.evaluation_model.generate(
            evaluation_prompt,
            functions=[evaluation_function],
            function_call={"name": "evaluate_semantic_equivalence"}
        )

        if response.get('function'):
            function_data = response['function'].get('data', {})
            return function_data.get('is_equivalent', False)
        else:
            print("Unexpected response format from evaluation model")
            return False

    def save_log(self, iteration: int):
        log_entry = {
            'iteration': iteration,
            'evaluations': self.log[-1]
        }

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'a') as f:
            json_str = json.dumps(log_entry, indent=2)
            f.write(json_str)
            f.write('\n\n')  # Add an extra newline for separation between entries

    def get_log(self):
        return self.log
