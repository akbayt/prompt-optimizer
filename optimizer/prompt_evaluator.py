from typing import List, Dict, Any, Tuple
from optimizer.model_interface import Model
from config import EVALUATION_MODEL
from utils.data_loader import DataLoader
from utils.performance_logger import PerformanceLogger


class Evaluator:
    def __init__(self, logger: PerformanceLogger):
        self.evaluation_model = Model(EVALUATION_MODEL)
        self.data_loader = DataLoader()
        self.dataset = self.data_loader.load_data()
        self.logger = logger

    async def evaluate_prompts(self, prompts: List[str], generator_model: Model, iteration: int) -> List[
        Dict[str, Any]]:
        evaluations = []
        for prompt in prompts:
            evaluation = await self.evaluate_prompt(prompt, generator_model)
            evaluations.append(evaluation)

        # Log the iteration using the PerformanceLogger
        self.logger.log_iteration(iteration, prompts, evaluations)

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

            evaluation_result = await self.evaluate_output(model_output, expected_output)

            if evaluation_result['is_correct']:
                correct_answers += 1

            results.append({
                'prompt': formatted_prompt,
                'model_output': model_output,
                'expected_output': expected_output,
                'is_correct': evaluation_result['is_correct'],
                'explanation_success': evaluation_result['explanation_success'],
                'explanation_failure': evaluation_result['explanation_failure']
            })

        summary = await self.summarize_explanations(results)

        return {
            'prompt': prompt,
            'total_cases': total_cases,
            'correct_answers': correct_answers,
            'score': correct_answers / total_cases,
            'results': results,
            'summary': summary
        }

    async def summarize_explanations(self, results: List[Dict[str, Any]]) -> str:
        success_explanations = []
        failure_explanations = []

        for result in results:
            if result['is_correct']:
                success_explanations.append(result['explanation_success'])
            else:
                failure_explanations.append(result['explanation_failure'])

        success_list = '\n'.join(success_explanations)
        failure_list = '\n'.join(failure_explanations)

        prompt = f"""An AI model was provided with various prompts, and the results were compared to expected outputs. 
    Some prompts were successful, while others failed. Here is a summary of the evaluations:

    Explanations of successful evaluations:
    {success_list}

    Explanations of failed evaluations:
    {failure_list}

    Please provide a concise summary of these evaluations. Your summary should:
    1. Summarize the key strengths of the successful prompts.
    2. Explain the main weaknesses or issues with the failed prompts.
    3. Suggest potential areas for improvement in the prompt design with the insights into any pattern or trends you noticed.

    Your summary should be clear, brief, and focused on helping improve future prompt performance."""

        response = await self.evaluation_model.generate(prompt)

        summary = response.get('message', '')

        return summary

    async def generate_model_output(self, prompt: str, generator_model: Model) -> str:
        response = await generator_model.generate(prompt)
        return response.get('message', '')

    async def evaluate_output(self, model_output: str, expected_output: str) -> Dict[str, Any]:
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
                    "explanation_success": {
                        "type": "string",
                        "description": "A very brief explanation of why the outputs are equivalent"
                    },
                    "explanation_failure": {
                        "type": "string",
                        "description": "A very brief explanation of why the outputs are not equivalent"
                    }
                },
                "required": ["is_equivalent", "explanation_success", "explanation_failure"]
            }
        }

        try:
            response = await self.evaluation_model.generate(
                evaluation_prompt,
                functions=[evaluation_function],
                function_call={"name": "evaluate_semantic_equivalence"}
            )

            if response.get('function'):
                function_data = response['function'].get('data', {})
                is_correct = function_data.get('is_equivalent', False)

                explanation_success = function_data.get('explanation_success', '') if is_correct else ''
                explanation_failure = function_data.get('explanation_failure', '') if not is_correct else ''

                return {
                    'is_correct': is_correct,
                    'explanation_success': explanation_success,
                    'explanation_failure': explanation_failure
                }
            else:
                print("Unexpected response format from evaluation model")
                return {
                    'is_correct': False,
                    'explanation_success': '',
                    'explanation_failure': 'Evaluation failed due to unexpected response format'
                }

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {
                'is_correct': False,
                'explanation_success': '',
                'explanation_failure': f'Evaluation failed due to error: {str(e)}'
            }
