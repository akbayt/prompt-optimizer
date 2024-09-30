import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Original prompt to optimize
ORIGINAL_PROMPT = {
    "text": "Summarize the following text in one sentence: {text}",
    "variables": ["text"]
}

# Optimization settings
MAX_ITERATIONS = 10
PARALLEL_VARIATIONS = 3
HISTORICAL_PROMPTS_COUNT = 10 # the number of prompts to keep in memory for historical analysis

# Model settings
EVALUATION_MODEL = "gpt-4o-mini"  # Model used for evaluating outputs
PROMPT_GENERATOR_MODEL = "gpt-4o-mini"  # Model used for generating prompt variations
GENERATOR_MODEL = "gpt-4o-mini"  # Model used for generating responses, the prompt will be optimized for this model

# Provider settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data paths
DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DATASET_FILE = "dataset.json"
LOG_DIR = "logs"

# Prompt generation settings
MAX_PROMPT_LENGTH = 2000

# Error handling
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

ACCURACY_THRESHOLD = 0.95
