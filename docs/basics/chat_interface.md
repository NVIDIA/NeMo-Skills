# Chat Interface

This guide explains how to run the interactive Gradio demo for interacting with models.

## Overview

The chat interface provides a web UI where you can interact with a model. When launched with code execution capabilities, the model can use a Python interpreter to solve problems.

## Launching with Docker

The simplest way to run the chat interface is using Docker.

### Prerequisites

- Docker installed on your system
- NVIDIA GPU with appropriate drivers

### Steps

1. Create a Dockerfile with the following content:

```dockerfile
FROM igitman/nemo-skills-vllm:0.6.0 as base

WORKDIR /app/
RUN chown -R 1000:1000 /app

RUN git clone https://github.com/NVIDIA/NeMo-Skills \
    && cd NeMo-Skills \
    && pip install --ignore-installed blinker \
    && pip install -e . \
    && pip install gradio \
    # uncomment if plan to use model that executes Python
    # && pip install -r requirements/code_execution.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY --chown=1000 entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT [ "/app/entrypoint.sh" ]
```

2. Create an entrypoint.sh script:

```bash
#!/bin/bash

# start model server with sandbox inside current container
CUDA_VISIBLE_DEVICES=0,1 ns start_server \
   --model=/models/YourModel \
   --server_gpus=2 \
    # --with_sandbox \  # Only needed if using code execution
   --server_type vllm &

# launch gradio app
cd /app/NeMo-Skills/ \
   && python3 -m nemo_skills.inference.launch_chat_interface \
      --server_type vllm \
    #   --with_code_execution  # Only needed if using code execution

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
```

3. Build and run the Docker container:

```bash
docker build -t ai-chat .
docker run --gpus all -p 7860:7860 -v /path/to/your/models:/models/ ai-chat
```

## Running without Docker

You can also run the chat interface directly without Docker.

### Prerequisites

- Python 3.10+
- NeMo-Skills repository installed
- Required dependencies installed

### Steps

1. Start the model server:

```bash
ns start_server \
   --model=/path/to/your/model \
   --server_gpus=2 \
   --server_type vllm \
   # --with_sandbox # Only needed if using code execution
```

2. In a separate terminal, launch the chat interface:

```bash
# For regular chat:
python -m nemo_skills.inference.launch_chat_interface \
   --server_type vllm \
   --host localhost \
   --prompt_config generic/math \
   --prompt_template openmath-instruct

# For chat with code execution:
python -m nemo_skills.inference.launch_chat_interface \
   --server_type vllm \
   --host localhost \
   --with_code_execution \
   --max_code_executions 8 \
   --prompt_config openmath/tir \
   --prompt_template openmath-instruct
```

## Command Line Options

The chat interface supports the following command line options:

- `--host`: Hostname where both the model server and the execution sandbox are running (default: "localhost")
- `--server_type`: Type of the model server to use (choices: "vllm", "sglang", "trtllm", default: "vllm")
- `--with_code_execution`: Enable code execution capabilities (requires server started with `--with_sandbox`)
- `--max_code_executions`: Maximum number of Python code blocks that can be executed per generation (default: 8)
- `--add_remaining_code_executions`: Append the number of remaining executions after each code output block
- `--prompt_config`: Identifier or path of the prompt config to load (default: "openmath/tir")
- `--prompt_template`: Identifier or path of the prompt template to load (default: "openmath-instruct")

## Using the Interface

Once running, you can access the interface by navigating to `http://localhost:7860` in your web browser. Enter your question in the text box and click Submit or press Enter.
