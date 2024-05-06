# Inference

Here are the instructions on how to run inference with our models. Note that you
cannot use a simple LLM call as our models rely on Python code interpreter to execute
parts of the output.

Make sure to complete [prerequisites](/docs/prerequisites.md) before proceeding.


1. Get the model you want to use. You can use one of our
   [pretrained models](https://huggingface.co/collections/nvidia/openmath-65c5619de2ba059be0775014)
   in a zero-shot setting or any other model with few-shot examples of code usage in the prompt.

2. [Convert the model](/docs/checkpoint-conversion.md) if it's not in the right format.
   You do not need any conversion if using one of our NeMo models and can refer to
   the exact steps on how to convert HF version of the models to TensorRT-LLM format
   [here](/docs/reproducing-results.md#evaluation).

3. Start model server and local [sandbox](/docs/sandbox.md) for code execution.
   We recommend using TensorRT-LLM for fastest inference,
   but NeMo might be easier to setup (works on more GPU types and does not require
   conversion for pretrained models).

   You can also run this command on a slurm cluster and then submit requests from a local workstation by setting up
   `NEMO_SKILLS_SSH_SERVER` and `NEMO_SKILLS_SSH_KEY_PATH` environment variables (if you have ssh access there).

   ```
   python pipeline/start_server.py \
       --model_path <path to the model in the right format> \
       --server_type <nemo or tensorrt_llm> \
       --num_gpus <number of GPUs you want to use>
   ```

   Make sure to provide the path to the `nemo_model` subfolder if using NeMo models.

4. Wait until you see "Server is running on" message, then send requests to the server through Python API or by using our [visualization tool](/visualization/Readme.md).

    ```python
    from dataclasses import asdict

    from nemo_skills.inference.server.model import get_model
    from nemo_skills.code_execution.sandbox import get_sandbox
    from nemo_skills.inference.prompt.utils import Prompt, get_prompt_config
    from nemo_skills.inference.generate_solutions import InferenceConfig

    sandbox = get_sandbox(
        sandbox_type="local",
        host=<IP address of sandbox printed on previous step>,
    )

    llm = get_model(
        server_type=<server type from previous step>,
        host=<IP address of server printed on previous step>,
        sandbox=sandbox,
    )

    # replace with "code_base" if model that was not pretrained with our pipeline
    # you can also write a new .yaml file and put it inside nemo_skills/inference/prompt
    # to customize further
    prompt_config = get_prompt_config("code_sfted")
    prompt_config.few_shot_examples.num_few_shots = 0
    # replace with the following if model that was not pretrained with our pipeline
    # you can pick different few shot examples based on your needs
    # prompt_config.few_shot_examples.num_few_shots = 5
    # prompt_config.few_shot_examples.examples_type = "gsm8k_text_with_code"

    question = (
        "In a dance class of 20 students, 20% enrolled in contemporary dance, "
        "25% of the remaining enrolled in jazz dance, and the rest enrolled in "
        "hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
    )
    # can provide multiple requests
    prompts = [Prompt(prompt_config, input_dict={'question': question})]

    # can provide different inference parameters here
    inference_cfg = InferenceConfig(temperature=0)  # greedy
    outputs = llm(
        prompts=prompts,
        stop_phrases=list(prompt_config.stop_phrases),
        **asdict(inference_cfg),
    )
    print(outputs[0]["generated_solution"])
    # there are also other keys there, like predicted_answer
    # or code error_message that could be useful!
    ```

If using `--server_type=nemo` locally, the NeMo server might sometimes hang around
after `start_server.py` is already killed. In that case you should manually stop it,
e.g. by running

```
docker ps -a -q --filter ancestor=igitman/nemo-skills-sft:0.2.0 | xargs docker stop
```

If for some reason you do not want to use [pipeline/start_server.py](/pipeline/start_server.py) helper script,
you can always start sandbox and LLM server manually. We describe this process in
[evaluation/details](/docs/evaluation.md#details) section.