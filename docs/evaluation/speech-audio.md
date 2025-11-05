# Speech and Audio Understanding

This section details how to evaluate speech and audio understanding benchmarks, testing models' ability to understand and reason about audio content including speech, music, and environmental sounds.

## Supported benchmarks

### MMAU-Pro

MMAU-Pro (Multimodal Audio Understanding - Pro) is a comprehensive benchmark for evaluating audio understanding capabilities across three different task categories:

- **Closed-form questions**: Questions with specific answers evaluated using NVEmbed similarity matching
- **Open-ended questions**: Questions requiring detailed responses, evaluated with LLM-as-a-judge (Qwen 2.5)
- **Instruction following**: Tasks that test the model's ability to follow audio-related instructions

#### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/mmau-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmau-pro/__init__.py)
- Original benchmark source is hosted on [HuggingFace](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro)

## Preparing MMAU-Pro Data

MMAU-Pro requires audio files for evaluation. The `--with-audio` parameter controls whether audio files are downloaded.

> **Note:** Meaningful evaluation requires audio files and audio-capable models. Text-only data preparation is possible but not recommended.

### Data Preparation with Audio Files

To prepare the dataset with audio files:

```bash
export HF_TOKEN=your_huggingface_token
ns prepare_data mmau-pro --with-audio
```

**What happens:**
1. Downloads audio archive (~50GB) from HuggingFace
2. Requires authentication (HuggingFace token via `HF_TOKEN` environment variable)
3. Extracts audio files to dataset directory


### Custom Data Directory

To store the dataset in a specific location:

```bash
ns prepare_data mmau-pro --with-audio --data-dir=/path/to/mmau-pro-data
```

Useful for cluster storage, sharing data across jobs, or persistent storage across runs.

## Running Evaluation

> **⚠️ Warning:** MMAU-Pro evaluation currently supports only the Megatron server type (`--server_type=megatron`) with audio-capable models. Support for additional server types is planned for future releases.

### Evaluation Example

Complete example using Megatron-based audio-visual language models:

```bash
# Set up environment variables - ADJUST THESE PATHS TO YOUR SETUP
export MEGATRON_PATH="/workspace/path/to/megatron-lm"
export CKPT_PATH=/workspace/path/to/checkpoint-tp1
export MODEL_CFG_PATH="${CKPT_PATH}/config.yaml"
export SERVER_ENTRYPOINT="$MEGATRON_PATH/path/to/server.py"
export SERVER_CONTAINER="/path/to/server_container.sqsh"
export OUTPUT_DIR="mmau-pro-eval"

# Set up keys
export HF_TOKEN='your_huggingface_token'
export NVIDIA_API_KEY='your_nvidia_api_key'
export WANDB='your_wandb_key'   # not neccessary

# Run evaluation with audio support
export HF_TOKEN=${HF_TOKEN} && \
export NVIDIA_API_KEY=${NVIDIA_API_KEY} && \
export MEGATRON_PATH="$MEGATRON_PATH" && \
ns eval \
    --cluster=oci_iad \
    --output_dir=/workspace/path/to/$OUTPUT_DIR \
    --benchmarks=mmau-pro \
    --server_type=megatron \
    --server_gpus=1 \
    --model=$CKPT_PATH \
    --server_entrypoint=$SERVER_ENTRYPOINT \
    --server_container=$SERVER_CONTAINER \
    --data_dir="/dataset" \
    --installation_command="pip install sacrebleu" \
    ++prompt_suffix='/no_think' \
    --server_args="--inference-max-requests 1 \
                   --model-config ${MODEL_CFG_PATH} \
                   --num-tokens-to-generate 256 \
                   --temperature 1.0 \
                   --top_p 1.0"
```


### Evaluating Individual Categories

You can evaluate specific MMAU-Pro categories independently by specifying the sub-benchmark:

```bash
# Evaluate only closed-form questions
ns eval --benchmarks=mmau-pro.closed_form [... other args ...]
```

## How Evaluation Works

Each category uses a different evaluation strategy:

| Category | Evaluation Method | How It Works |
|----------|-------------------|--------------|
| **Closed-Form** | NVEmbed similarity matching | Model generates short answer; compared to expected answer using embeddings |
| **Open-Ended** | LLM-as-a-judge (Qwen 2.5 7B) | Model generates detailed response; Qwen 2.5 judges quality and correctness |
| **Instruction Following** | Custom evaluation logic | Model follows instructions; evaluator checks adherence |

**Metrics tracked**: Success rate, average tokens, generation time, no-answer rate

## Understanding Results

After evaluation completes, results are saved in your output directory under `eval-results/`:

```
<output_dir>/
├── eval-results/
│   └── mmau-pro/
│       ├── metrics.json                              # Overall aggregate scores
│       ├── mmau-pro.instruction_following/
│       │   └── metrics.json
│       ├── mmau-pro.closed_form/
│       │   └── metrics.json
│       └── mmau-pro.open_ended/
│           └── metrics.json
```

### Evaluation Output Format

When evaluation completes, results are displayed in formatted tables in the logs:

**Open-Ended Questions:**

```
------------------------------- mmau-pro.open_ended -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 82         | 196         | 14.88%       | 0.00%     | 625
```

**Instruction Following:**

```
-------------------------- mmau-pro.instruction_following -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 102         | 21.84%       | 0.00%     | 87

```

**Closed-Form Questions (Main Category + Sub-categories):**

```
------------------------------- mmau-pro.closed_form ------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 2          | 6581        | 33.88%       | 0.00%     | 4593

---------------------------- mmau-pro.closed_form-sound ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 691         | 26.15%       | 0.00%     | 1048

---------------------------- mmau-pro.closed_form-multi ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6005        | 24.65%       | 0.00%     | 430

------------------------- mmau-pro.closed_form-sound_music ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 810         | 22.00%       | 0.00%     | 50

---------------------------- mmau-pro.closed_form-music ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 5          | 5467        | 42.81%       | 0.00%     | 1418

------------------------ mmau-pro.closed_form-spatial_audio -----------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5597        | 2.15%        | 0.00%     | 325

------------------------ mmau-pro.closed_form-music_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 5658        | 36.96%       | 0.00%     | 46

--------------------- mmau-pro.closed_form-sound_music_speech ---------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5664        | 14.29%       | 0.00%     | 7

------------------------ mmau-pro.closed_form-sound_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5713        | 36.36%       | 0.00%     | 88

--------------------------- mmau-pro.closed_form-speech ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6312        | 38.16%       | 0.00%     | 891

------------------------- mmau-pro.closed_form-voice_chat -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 6580        | 55.52%       | 0.00%     | 290
```

**Overall Aggregate Score:**

```
-------------------------------- mmau-pro -----------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 11         | 6879        | 31.44%       | 0.00%     | 5305
```

## Tips

1. **Audio files**: Large dataset (~50GB) - download with `--with-audio`
2. **Evaluation time**: Closed-form evaluation can take longer due to lengthy audio inputs (some music clips are 3+ minutes)
