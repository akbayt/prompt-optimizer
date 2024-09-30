import json
import os
from typing import Dict, Any, List
from datetime import datetime


class PerformanceLogger:
    def __init__(self, log_dir: str, original_prompt: str):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"performance_log_{self.timestamp}.json")
        self.log_data = {
            "original_prompt": original_prompt,
            "optimized_prompt": None,
            "optimization_logs": []
        }
        self._save_log()  # Save immediately upon initialization

    def log_iteration(self, iteration: int, prompts: List[str], evaluations: List[Dict[str, Any]]):
        iteration_log = {
            "iteration": iteration,
            "prompts": prompts,
            "evaluations": evaluations
        }
        self.log_data["optimization_logs"].append(iteration_log)
        self._save_log()

    def log_optimized_prompt(self, optimized_prompt: str):
        self.log_data["optimized_prompt"] = optimized_prompt
        self._save_log()

    def _save_log(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def get_log_file_path(self) -> str:
        return self.log_file


def create_logger(log_dir: str, original_prompt: str) -> PerformanceLogger:
    return PerformanceLogger(log_dir, original_prompt)
