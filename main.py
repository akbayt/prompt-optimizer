import asyncio
from optimizer.prompt_generator import PromptGenerator
from optimizer.prompt_evaluator import Evaluator
from optimizer.model_interface import Model
from utils.performance_logger import create_logger
from config import (
    MAX_ITERATIONS,
    PARALLEL_VARIATIONS,
    PROMPT_GENERATOR_MODEL,
    GENERATOR_MODEL,
    ORIGINAL_PROMPT,
    ACCURACY_THRESHOLD,
    LOG_DIR
)


async def optimize_prompt() -> str:
    generator = PromptGenerator(PROMPT_GENERATOR_MODEL)
    logger = create_logger(LOG_DIR, ORIGINAL_PROMPT['text'])
    evaluator = Evaluator(logger)
    target_model = Model(GENERATOR_MODEL)

    current_prompts = [ORIGINAL_PROMPT['text']]
    best_prompt = ORIGINAL_PROMPT['text']
    best_score = 0

    print(f"Starting prompt optimization process...")
    print(f"Original prompt: {ORIGINAL_PROMPT['text']}")
    print(f"Target accuracy threshold: {ACCURACY_THRESHOLD}")

    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration + 1}/{MAX_ITERATIONS}")

        # Evaluate current prompts
        evaluations = await evaluator.evaluate_prompts(current_prompts, target_model, iteration)

        # Print evaluation results
        print("\nPrompt Evaluations:")
        for eval in evaluations:
            print(f"Prompt: {eval['prompt']}")
            print(f"Score: {eval['score']:.4f}")
            print("---------------")

        # Find the best performing prompt
        for eval in evaluations:
            if eval['score'] > best_score:
                best_score = eval['score']
                best_prompt = eval['prompt']

        print(f"\nBest prompt so far: {best_prompt}")
        print(f"Best score: {best_score:.4f}")

        # Check if we've reached our goal
        if best_score >= ACCURACY_THRESHOLD:
            print(f"\nAccuracy threshold reached! Optimization complete.")
            break

        # Get historical prompts
        historical_prompts = logger.get_historical_prompts()

        # Generate new prompt suggestions
        prompts_and_evals = [(eval['prompt'], eval['summary'], eval['score']) for eval in historical_prompts]
        suggestions = await generator.generate_suggestions(prompts_and_evals, PARALLEL_VARIATIONS)

        print("\nNew prompt suggestions:")
        for suggestion in suggestions['suggestions']:
            print(f"Prompt: {suggestion['prompt']}")
            print(f"Explanation: {suggestion['explanation']}")
            print("-----------------")

        # Prepare for next iteration
        current_prompts = [suggestion['prompt'] for suggestion in suggestions['suggestions']]

    # Log the optimized prompt
    logger.log_optimized_prompt(best_prompt)

    return best_prompt


async def main():
    optimized_prompt = await optimize_prompt()
    print(f"\nOptimization process completed.")
    print(f"Final optimized prompt: {optimized_prompt}")


if __name__ == "__main__":
    asyncio.run(main())
