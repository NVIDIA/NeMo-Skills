import os
from itertools import islice
from datasets import Dataset, load_dataset
from nemo_skills.prompt.utils import get_prompt

# Define paths and parameters
LOCAL_DATASET_PATH = './calibration_dataset'
CALIB_DATASET_NAME = "nvidia/OpenMathReasoning"
CALIB_SPLIT = 'tir'
CALIB_SIZE = 4096

# Load samples, format them, and save as a Parquet file
print(f"Loading and formatting {CALIB_SIZE} samples for calibration...")
ds_samples = load_dataset(CALIB_DATASET_NAME, split=CALIB_SPLIT, streaming=True)
ds_samples = list(islice(ds_samples, CALIB_SIZE))

prompt_template = get_prompt('generic/math', tokenizer='nvidia/OpenMath-Nemotron-14B-kaggle')
calibration_dataset = Dataset.from_dict(
    {
        "text": [
            prompt_template.format_assistant_response(
                prompt_template.fill(
                    {k: v for k, v in sample.items() if k in ['problem', 'generated_solution']},
                    start_assistant_response_key='generated_solution',
                )
            )
            for sample in ds_samples
        ]
    }
)

os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
calibration_dataset.to_parquet(f"{LOCAL_DATASET_PATH}/data.parquet")
print(f"Calibration dataset saved to {LOCAL_DATASET_PATH}/data.parquet")
