import json
import os
from typing import Dict, Any, List
from datetime import datetime
from config import HISTORICAL_PROMPTS_COUNT
import matplotlib.pyplot as plt


class PerformanceLogger:
    def __init__(self, log_dir: str, original_prompt: str):
        self.log_dir = log_dir

        # if log_dir does not exist, create it
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"performance_log_{self.timestamp}.json")
        self.plot_file = os.path.join(self.log_dir, f"performance_plot_{self.timestamp}.png")
        self.log_data = {
            "original_prompt": original_prompt,
            "optimized_prompt": None,
            "optimization_logs": []
        }
        self._save_log()  # Save immediately upon initialization

        # Initialize plot
        plt.figure(figsize=(12, 6))
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Prompt Optimization Progress')
        plt.ylim(0, 1)  # Assuming scores are between 0 and 1

    def log_iteration(self, iteration: int, prompts: List[str], evaluations: List[Dict[str, Any]]):
        iteration_log = {
            "iteration": iteration,
            "prompts": prompts,
            "evaluations": evaluations
        }
        self.log_data["optimization_logs"].append(iteration_log)
        self._save_log()
        self._update_plot()

    def log_optimized_prompt(self, optimized_prompt: str):
        self.log_data["optimized_prompt"] = optimized_prompt
        self._save_log()
        self._update_plot()

    def _save_log(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def _update_plot(self):
        plt.clf()  # Clear the current figure
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Prompt Optimization Progress')
        plt.ylim(0, 1)

        iterations = []
        scores = []

        for log in self.log_data["optimization_logs"]:
            iteration = log["iteration"]
            for evaluations in log["evaluations"]:
                iterations.append(iteration)
                scores.append(evaluations["score"])

        plt.scatter(iterations, scores, alpha=0.5)
        plt.xticks(range(1, max(iterations) + 1))

        plt.savefig(self.plot_file)

    def get_log_file_path(self) -> str:
        return self.log_file

    def get_plot_file_path(self) -> str:
        return self.plot_file

    def get_historical_prompts(self) -> List[Dict[str, Any]]:
        all_prompts = []
        for log in self.log_data["optimization_logs"]:
            all_prompts.extend(log["evaluations"])

        # Sort prompts by score in descending order
        sorted_prompts = sorted(all_prompts, key=lambda x: x['score'], reverse=True)

        # if the number of prompts is less than the historical prompts count, return all prompts
        if len(sorted_prompts) <= HISTORICAL_PROMPTS_COUNT:
            return sorted_prompts

        # Calculate how many prompts to take from top and bottom
        half_count = HISTORICAL_PROMPTS_COUNT // 2
        remainder = HISTORICAL_PROMPTS_COUNT % 2

        # Get top performing prompts
        top_prompts = sorted_prompts[:half_count + remainder]

        # Get bottom performing prompts
        bottom_prompts = sorted_prompts[-half_count:]

        # Combine and return the results
        return top_prompts + bottom_prompts


def create_logger(log_dir: str, original_prompt: str) -> PerformanceLogger:
    return PerformanceLogger(log_dir, original_prompt)
