# Inference

Here are the instructions on how to run inference with our repo.

## Download/convert the model

Get the model you want to use. You can use any model that's supported by vLLM, TensorRT-LLM or NeMo.
You can also use [Nvidia NIM API](https://www.nvidia.com/en-us/ai/) for models that are hosted there.

[Convert the model](../pipelines/checkpoint-conversion.md) if it's not in the format you want to use.
You do not need any conversion if using vLLM inference with HF models
(and can directly use model id if you want vLLM to download it for you).
For fastest inference we recommend to convert the model to TensorRT-LLM format.

## Start the server

Start the server hosting your model. Here is an example (make sure the `/hf_models` mount is defined in your cluster config). Skip this step if you want to use cloud models through an API.

```bash
ns start_server \
    --cluster local \
    --model /hf_models/Meta-Llama-3.1-8B-Instruct \
    --server_type vllm \
    --server_gpus 1 \
    --server_nodes 1
```

If the model needs to execute code, add `--with_sandbox`

## Send inference requests

Click on :material-plus-circle: symbols in the snippet below to learn more details.


=== "Self-hosted models"

    ```python
    from nemo_skills.inference.server.model import get_model
    from nemo_skills.prompt.utils import get_prompt

    llm = get_model(server_type="vllm")  # localhost by default
    prompt = get_prompt('generic/default', 'llama3-instruct') # (1)!
    prompts = [prompt.fill({'question': "What's 2 + 2?"})]
    print(prompts[0]) # (2)!
    outputs = llm.generate(prompts=prompts)
    print(outputs[0]["generation"]) # (3)!
    ```

    1.   Here we use [generic/default](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/default.yaml) config
         and [llama3-instruct](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/template/llama3-instruct.yaml) template.

         See [nemo_skills/prompt](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt) for more config/template options
         or [create your own prompts](prompt-format.md)


    2.   This should print

         ```python-console
         >>> print(prompts[0])
         <|begin_of_text|><|start_header_id|>system<|end_header_id|>

         <|eot_id|><|start_header_id|>user<|end_header_id|>

         What's 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
         ```

         If you don't want to use our prompt class, just create this string yourself

    3.   This should print
         ```python-console
         >>> print(outputs[0]["generation"])
         2 + 2 = 4.
         ```

=== "API models"

    ```python
    from nemo_skills.inference.server.model import get_model
    from nemo_skills.prompt.utils import get_prompt

    llm = get_model( # (1)!
        server_type="openai",  # NIM models are using OpenAI API
        base_url="https://integrate.api.nvidia.com/v1",
        model="meta/llama-3.1-8b-instruct",
    )
    prompt = get_prompt('generic/default') # (2)!

    prompts = [prompt.fill({'question': "What's 2 + 2?"})]

    print(prompts[0]) # (3)!
    outputs = llm.generate(prompts=prompts)
    print(outputs[0]["generation"]) # (4)!
    ```

    1.   Don't forget to define `NVIDIA_API_KEY`.

         To use OpenAI models, use `OPENAI_API_KEY` and set `base_url=https://api.openai.com/v1`.

         To use Azure OpenAI models, use `server_type="azureopenai"` and set `base_url` to your Azure OpenAI endpoint URL.
         You'll also need to set `api_version` (defaults to "2024-12-01-preview") and provide your Azure OpenAI API key.

    2.   Here we use [generic/default](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/default.yaml) config.
         Note that with API models we can't add special tokens, so prompt template is not specified.

         See [nemo_skills/prompt](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt) for more config/template options
         or [create your own prompts](prompt-format.md)


    3.   This should print

         ```python-console
         >>> print(prompts[0])
         [{'role': 'user', 'content': "What's 2 + 2?"}]
         ```

         If you don't want to use our prompt class, just create this list yourself

    4.   This should print
         ```python-console
         >>> print(outputs[0]["generation"])
         2 + 2 = 4.
         ```

=== "With code execution"

    ``` python
    from nemo_skills.code_execution.sandbox import get_sandbox
    from nemo_skills.inference.server.code_execution_model import get_code_execution_model
    from nemo_skills.prompt.utils import get_prompt

    sandbox = get_sandbox()  # localhost by default
    llm = get_code_execution_model(server_type="vllm", sandbox=sandbox)
    prompt = get_prompt('generic/default', 'llama3-instruct') # (1)!
    prompt.config.system = ( # (2)!
        "Environment: ipython\n\n"
        "Use Python to solve this math problem."
    )
    prompts = [prompt.fill({'question': "What's 2 + 2?"})]
    print(prompts[0]) # (3)!
    outputs = llm.generate(prompts=prompts, **prompt.get_code_execution_args()) # (4)!
    print(outputs[0]["generation"]) # (5)!
    ```

    1.   Here we use [generic/default](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/config/generic/default.yaml) config
         and [llama3-instruct](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt/template/llama3-instruct.yaml) template.

         Note how we are updating system message on the next line (you can also include it in the config directly).

         See [nemo_skills/prompt](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/prompt) for more config/template options
         or [create your own prompts](prompt-format.md)

    2.   8B model doesn't always follow these instructions, so using 70B or 405B for code execution is recommended.

    3.   This should print

         ```python-console
         >>> print(prompts[0])
         <|begin_of_text|><|start_header_id|>system<|end_header_id|>

         Environment: ipython

         Use Python to solve this math problem.<|eot_id|><|start_header_id|>user<|end_header_id|>

         What's 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
         ```

         If you don't want to use our prompt class, just create this string yourself

    4.   `prompt.get_code_execution_args()` simply returns a dictionary with start/stop tokens,
         so that we know when to stop LLM generation and how to format the output.

         If you don't want to use our prompt class, just define those parameters directly.

    5.   This should print
         ```python-console
         >>> print(outputs[0]["generation"])
         <|python_tag|>print(2 + 2)<|eom_id|><|start_header_id|>ipython<|end_header_id|>

         completed
         [stdout]
         4
         [/stdout]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

         The answer is 4.
         ```

         The "4" in the stdout is coming directly from Python interpreter running in the sandbox.

Note that for self-hosted models we are explicitly adding all the special tokens before sending prompt to an LLM.
This is necessary to retain flexibility. E.g. this way we can use base model format with
instruct models that we found to work better with few-shot examples.

You can learn more about how our prompt formatting works in the [prompt format docs](../basics/prompt-format.md).

!!! note

    You can also use slurm config when launching a server. If you do that, add `host=<slurm node hostname>`
    to the `get_model/sandbox` calls and define `NEMO_SKILLS_SSH_KEY_PATH` and `NEMO_SKILLS_SSH_SERVER` env vars
    to set the connection through ssh.