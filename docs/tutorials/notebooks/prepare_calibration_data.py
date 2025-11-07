import os
import gc
from itertools import islice
from datasets import Dataset, load_dataset, disable_caching
from nemo_skills.prompt.utils import get_prompt

disable_caching()
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define paths and parameters
LOCAL_DATASET_PATH = './calibration_dataset'
CALIB_DATASET_NAME = "nvidia/OpenMathReasoning"
CALIB_SPLIT = 'tir'
CALIB_SIZE = 4096

# Load samples, format them, and save as a Parquet file
print(f"Loading and formatting {CALIB_SIZE} samples for calibration...")
ds_samples = load_dataset(CALIB_DATASET_NAME, split=CALIB_SPLIT, streaming=True)

prompt_template = get_prompt('generic/math', tokenizer='nvidia/OpenMath-Nemotron-14B-kaggle')

# Process iteratively instead of loading all into memory
all_texts = []
for sample in islice(ds_samples, CALIB_SIZE):
    formatted_text = prompt_template.format_assistant_response(
        prompt_template.fill(
            {k: v for k, v in sample.items() if k in ['problem', 'generated_solution']},
            start_assistant_response_key='generated_solution',
            format_as_string=True,
        )
    )
    all_texts.append(formatted_text)

calibration_dataset = Dataset.from_dict({"text": all_texts})
os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
calibration_dataset.to_parquet(f"{LOCAL_DATASET_PATH}/data.parquet")

# Free memory before exit
del all_texts, calibration_dataset, prompt_template, ds_samples
gc.collect()
print(f"Calibration dataset saved to {LOCAL_DATASET_PATH}/data.parquet", flush=True)
os._exit(0)

