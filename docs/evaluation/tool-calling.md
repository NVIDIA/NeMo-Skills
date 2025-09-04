# Tool-calling

## Supported benchmarks

## bfcl_v3
<<<<<<< HEAD
=======

BFCL v3 consists of seventeen distinct evaluation subsets that comprehensively test various aspects of function calling capabilities, from simple function calls to complex multi-turn interactions.
>>>>>>> bfd24f30 (BFCL Docs (#753))

- Benchmark is defined in [`nemo_skills/dataset/bfcl_v3/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/bfcl_v3/__init__.py)
- Original benchmark source is [here](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).

### Data Preparation

To prepare BFCL v3 data for evaluation:

```bash
ns prepare_data bfcl_v3
```

This command performs the following operations:
<<<<<<< HEAD
=======

>>>>>>> bfd24f30 (BFCL Docs (#753))
- Downloads the complete set of BFCL v3 evaluation files
- Processes and organizes data into seventeen separate subset folders
- Creates standardized test files in JSONL format

**Example output structure**:
```
nemo_skills/dataset/bfcl_v3/
├── simple/test.jsonl
├── parallel/test.jsonl
├── multiple/test.jsonl
└── ... (other subsets)
```

### Challenges of tool-calling tasks

There are three key steps in tool-calling which differentiate it from typical text-only tasks:
<<<<<<< HEAD
=======

>>>>>>> bfd24f30 (BFCL Docs (#753))
1. **Tool Presentation**: Presenting the available tools to the LLM
2. **Response Parsing**: Extracting and validating tool calls from model-generated text
3. **Tool Execution**: Executing the tool calls and communicating the results back to the model

For 1 and 3, we borrow the implementation choices from the [the BFCL repo](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard). For tool call parsing, we support both client side and server side implementations.


#### Client-Side Parsing (Default)

**When to use**: Standard models supported by the BFCL repository

**How it works**: Utilizes the parsing logic from [BFCL's local inference handlers](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference)

**Configuration Requirements**:
- Model name specification via `++model_name=<model_id>`

**Sample Command**:

<<<<<<< HEAD
```bash hl=9
=======
```bash hl_lines="9"
>>>>>>> bfd24f30 (BFCL Docs (#753))
ns eval \
  --benchmarks bfcl_v3 \
  --cluster dfw \
  --model /hf_models/Qwen3-4B \
  --server_gpus 2 \
  --server_type vllm \
  --output_dir /workspace/qwen3-4b-client-parsing/ \
  ++inference.tokens_to_generate=8192 \
  ++model_name=Qwen/Qwen3-4B-FC \
```

#### Server-Side Parsing

**When to use**:
- Models not supported by BFCL client-side parsing
- Custom tool-calling formats

**Configuration Requirements**:
- Set `++use_client_parsing=False` and
- Specify appropriate server arguments. For example, evaluating Qwen models with vllm server would require setting the server_args as follows:
```bash
--server_args="--enable-auto-tool-choice --tool-call-parser hermes"
```

<<<<<<< HEAD
**Sample Commands**:

The following command evaluates the `Qwen3-4B` model which uses a standard tool-calling format supported by vllm
```bash hl=9-10
=======
**Sample Command**:

The following command evaluates the `Qwen3-4B` model which uses a standard tool-calling format supported by vllm

```bash hl_lines="9-10"
>>>>>>> bfd24f30 (BFCL Docs (#753))
ns eval \
  --benchmarks bfcl_v3 \
  --cluster dfw \
  --model /hf_models/Qwen3-4B \
  --server_gpus 2 \
  --server_type vllm \
  --output_dir /workspace/qwen3-4b-server-parsing/ \
  ++inference.tokens_to_generate=8192 \
  ++use_client_parsing=False \
  --server_args="--enable-auto-tool-choice --tool-call-parser hermes"
```

<<<<<<< HEAD
Some models implement bespoke tool-calling formats that require specialized parsing logic. For example, the [Llama-3.3-Nemotron-Super-49B-v1.5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) model implements its tool calling logic which requires passing the server the model-specific parsing script, [llama_nemotron_toolcall_parser_no_streaming.py](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/main/llama_nemotron_toolcall_parser_no_streaming.py).

**Custom parsing example (NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5)**


<!-- ### Custom Tool Calling Formats

Some models implement proprietary tool-calling formats that require specialized parsing logic.

**Example**: The [NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) model uses a bespoke format requiring its dedicated [parsing script](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/main/llama_nemotron_toolcall_parser_no_streaming.py). -->


<!-- The default is client side parsing which relies on the parsing logic implemented in [the BFCL repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference). Note that for models not supported in the BFCL repo, the client side parsing won't work. The client side parsing logic requires invoking the huggingface tokenizer which requires the model name that can be specified as `++model_name=Qwen/Qwen3-4B`.

The server side parsing requires enabling tool calling on the server and the applicable parser. For example, for `Qwen/Qwen3-4B`, the equivalent command for server-side tool call parsing for vllm would require passing ```++use_client_parsing=False --server_args="--tool-parser-plugin --tool-call-parser hermes"```
For parsing bespoke tool calling formats, the server needs to be given the parsing script. For example, the [Llama-3_3-Nemotron-Super-49B-v1_5 model](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) by NVIDIA is trained with a bespoke tool calling format which requires using their custom parsing script [llama_nemotron_toolcall_parser_no_streaming.py](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/main/llama_nemotron_toolcall_parser_no_streaming.py). This can be accomplished via something like:

```bash
++use_client_parsing=False
--server_args="--tool-parser-plugin \"/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/llama_nemotron_toolcall_parser_no_streaming.py\" \
                    --tool-call-parser llama_nemotron_json \
                    --enable-auto-tool-choice"
```

For the full command, [see here](https://nvidia.github.io/NeMo-Skills/tutorials/2025/08/15/reproducing-llama-nemotron-super-49b-v15-evals/#command-for-bfcl-eval-reasoning-on).

#### Sample commands -->



<!-- # Tool-calling Support in NeMo-Skills

## Overview

NeMo-Skills provides comprehensive support for tool-calling evaluation through the Berkeley Function Call Leaderboard (BFCL) benchmarks. This document covers the supported benchmarks, data preparation, implementation details, and usage examples.

## Supported Benchmarks

### BFCL v3

**Implementation location**: [`nemo_skills/dataset/bfcl_v3/__init__.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/bfcl_v3/__init__.py)

**Original benchmark**: [Berkeley Function Call Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

BFCL v3 consists of seventeen distinct evaluation subsets that comprehensively test various aspects of function calling capabilities, from simple function calls to complex multi-turn interactions. -->

## Data Preparation

To prepare BFCL v3 data for evaluation:

```bash
ns prepare_data bfcl_v3
```

This command performs the following operations:
- Downloads the complete set of BFCL v3 evaluation files
- Processes and organizes data into seventeen separate subset folders
- Creates standardized test files in JSONL format

**Example output structure**:
```
nemo_skills/dataset/bfcl_v3/
├── simple/test.jsonl
├── parallel/test.jsonl
├── multiple/test.jsonl
└── ... (other subsets)
```

## Tool-calling Implementation

### Core Challenges

Tool-calling evaluation presents three distinct technical challenges that differentiate it from standard text generation tasks:

1. **Tool Presentation**: Effectively communicating available tools and their schemas to the model
2. **Response Parsing**: Extracting and validating tool calls from model-generated text
3. **Tool Execution**: Running the called functions and providing results back to the model

### Implementation Approach

NeMo-Skills addresses these challenges by leveraging proven implementations from the original BFCL repository for tool presentation (#1) and execution (#3), while providing flexible parsing options for challenge #2.

### Tool Call Parsing Options

#### Client-Side Parsing (Default)

**When to use**: Standard models supported by the BFCL repository

**How it works**: Utilizes the parsing logic from [BFCL's local inference handlers](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference)

**Requirements**:
- HuggingFace tokenizer access
- Model name specification via `++model_name=<model_id>`

**Example**:
```bash
++model_name=Qwen/Qwen2.5-7B-Instruct
```

**Limitations**: Only works with models that have corresponding handlers in the BFCL repository.

#### Server-Side Parsing

**When to use**:
- Models not supported by BFCL client-side parsing
- Custom tool-calling formats
- Production deployments requiring server-side validation

**Configuration**: Set `++use_client_parsing=False` and specify appropriate server arguments.

**Standard server-side parsing example** (for Qwen models):
```bash
++use_client_parsing=False --server_args="--tool-parser-plugin --tool-call-parser hermes"
```

**Custom parsing example** (for NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5):
```bash
++use_client_parsing=False --server_args="--tool-parser-plugin \"/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/llama_nemotron_toolcall_parser_no_streaming.py\" \
                     --tool-call-parser llama_nemotron_json \
                     --enable-auto-tool-choice"
```

### Custom Tool Calling Formats

Some models implement proprietary tool-calling formats that require specialized parsing logic. NeMo-Skills supports these through custom parsing scripts.

**Example**: The [NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) model uses a bespoke format requiring its dedicated [parsing script](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/main/llama_nemotron_toolcall_parser_no_streaming.py).

## Usage Examples

### Basic BFCL v3 Evaluation

```bash
# Prepare data
ns prepare_data bfcl_v3

# Run evaluation with client-side parsing
ns eval \
    --cluster=local \
    --model=<model_name> \
    --benchmarks=bfcl_v3 \
    ++model_name=Qwen/Qwen2.5-7B-Instruct
```

### Server-Side Parsing Evaluation

```bash
# With standard server-side parser
ns eval \
    --cluster=local \
    --model=<model_name> \
    --benchmarks=bfcl_v3 \
    ++use_client_parsing=False \
    --server_args="--tool-parser-plugin --tool-call-parser hermes"
```

### Custom Model Evaluation

For models with custom tool-calling formats, refer to the complete command examples in the [Llama-Nemotron evaluation tutorial](https://nvidia.github.io/NeMo-Skills/tutorials/2025/08/15/reproducing-llama-nemotron-super-49b-v15-evals/#command-for-bfcl-eval-reasoning-on).

## Configuration Parameters
=======

**Custom parsing example (NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5)**

Some models implement bespoke tool-calling formats that require specialized parsing logic. For example, the [Llama-3.3-Nemotron-Super-49B-v1.5](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) model implements its tool calling logic which requires passing the model-specific parsing script, [llama_nemotron_toolcall_parser_no_streaming.py](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5/blob/main/llama_nemotron_toolcall_parser_no_streaming.py), to the server.


```bash hl_lines="12-16"
ns eval \
    --cluster=local \
    --benchmarks=bfcl_v3 \
    --model=/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/ \
    --server_gpus=2 \
    --server_type=vllm \
    --output_dir=/workspace/llama_nemotron_49b_1_5_tool_calling/ \
    ++inference.tokens_to_generate=65536 \
    ++inference.temperature=0.6 \
    ++inference.top_p=0.95 \
    ++system_message='' \
    ++use_client_parsing=False \
    --server_args="--tool-parser-plugin \"/workspace/Llama-3_3-Nemotron-Super-49B-v1_5/llama_nemotron_toolcall_parser_no_streaming.py\" \
                    --tool-call-parser \"llama_nemotron_json\" \
                    --enable-auto-tool-choice"
```

### Configuration Parameters
>>>>>>> bfd24f30 (BFCL Docs (#753))

| Configuration | True | False |
|---------------|------|-------|
| `++use_client_parsing` | Default | - |
| `++model_name` | Required for client parsing | - |
| `--server_args` | - | Required for server-side parsing |

<<<<<<< HEAD
## References

- [Berkeley Function Call Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [NeMo-Skills Repository](https://github.com/NVIDIA/NeMo-Skills)
- [BFCL Evaluation Methods](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval)
=======


!!!note
    To evaluate individual splits of `bfcl_v3`, such as `simple`, use `benchmarks=bfcl_v3.simple`.

!!!note
    Currently, ns summarize_results does not support benchmarks with custom aggregation requirements like BFCL v3. To handle this, the evaluation pipeline automatically launches a dependent job that processes the individual subset scores using [our scoring script](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/bfcl_v3/bfcl_score.py).
>>>>>>> bfd24f30 (BFCL Docs (#753))
