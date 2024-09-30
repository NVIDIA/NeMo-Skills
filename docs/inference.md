# Inference

Here are the instructions on how to run inference with our repo.

Make sure to complete [prerequisites](/docs/prerequisites.md).

1. Get the model you want to use. You can use any model that's supported by VLLM, TensorRT-LLM or NeMo.
   You can also use [Nvidia NIM API](https://www.nvidia.com/en-us/ai/) for models that are hosted there.

2. [Convert the model](/docs/checkpoint-conversion.md) if it's not in the format you want to use.
   You do not need any conversion if using VLLM inference with HF models.
   For fastest inference we recommend to convert the model to TensorRT-LLM format.

3. Start the server hosting your model. Here is an example (make sure the `/hf_models` mount is defined in your cluster config).

   ```
   ns start_server \
       --cluster local \
       --model /hf_models/Meta-Llama-3.1-8B-Instruct \
       --server_type vllm \
       --server_gpus 1 \
       --server_nodes 1
   ```

4. Run inference

   ```
   from nemo_skills.inference.server.model import get_model
   from nemo_skills.prompt.utils import get_prompt

   llm = get_model(server_type="vllm")  # localhost by default

   # generic/default prompt doesn't add any instructions
   # see nemo_skills/prompt/config for other available options for prompt config
   # llama3-instruct is the template for llama3 instruct models
   # see nemo_skills/prompt/template for prompt templates
   prompt = get_prompt('generic/default', 'llama3-instruct')

   prompts = [prompt.fill({'question': "What's 2 + 2?"})]

   # you can see exactly what we send to the model including all special tokens
   # if you don't want to use our prompt format, just create this string yourself
   print(prompts[0])
   outputs = llm.generate(prompts=prompts)
   print(outputs[0]["generation"])
   ```

   This should print
   ```
   >>> <|begin_of_text|><|start_header_id|>system<|end_header_id|>

       <|eot_id|><|start_header_id|>user<|end_header_id|>

       What's 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
   >>> 2 + 2 = 4.
   ```

   Note that we are explicitly adding all the special tokens before sending prompt to an LLM.
   This is necessary to retain flexibility. E.g. this way we can use base model format with
   instruct models that we found to work better with few-shot examples.

   You can learn more about how our prompt formatting works in [prompt format docs](/docs/prompt-format.md).

## Using API models

We support using models from [Nvidia NIM API](https://www.nvidia.com/en-us/ai/) as well as OpenAI models.
You need to define `NVIDIA_API_KEY` for this to work.

```
from nemo_skills.inference.server.model import get_model
from nemo_skills.prompt.utils import get_prompt

llm = get_model(
    server_type="openai",  # NIM models are using OpenAI API
    base_url="https://integrate.api.nvidia.com/v1",
    model="meta/llama-3.1-8b-instruct",
)

# generic/default prompt doesn't add any instructions
# see nemo_skills/prompt/config for other available options for prompt config
# note that with API models we can't add special tokens, so prompt template is unused here
prompt = get_prompt('generic/default')

prompts = [prompt.fill({'question': "What's 2 + 2?"})]

# again, you can prepare this yourself if you don't like our prompt utils
print(prompts[0])
outputs = llm.generate(prompts=prompts)
print(outputs[0]["generation"])
```

This should print
```
>>> [{'role': 'system', 'content': ''}, {'role': 'user', 'content': "What's 2 + 2?"}]
>>> 2 + 2 = 4.
```

To use OpenAI models, it's all the same but with `OPENAI_API_KEY` and set `base_url=https://api.openai.com/v1`.

## Using models that execute code

TBD