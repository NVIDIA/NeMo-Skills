# Dataset construction

Here are the commands you can run to re-create our synthetic dataset.
We assume you have `/workspace` defined in your [cluster config](../../basics/cluster-configs.md) and are
running commands with a Slurm config. Change all commands accordingly if running locally or using different paths.

## Math data

### Solution generation

We use problems from [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset. So first,
download them using this Python snippet and put inside `/workspace/open-reasoning/sdg` on your Slurm cluster.

We found that the quality of converted proof problems is not high, so we are excluding them here.

```python
from datasets import concatenate_datasets, load_dataset

def remove_proofs(example):
    return example['problem_type'] != 'converted_proof'

dataset = load_dataset("nvidia/OpenMathReasoning")

dataset['cot'] = dataset['cot'].remove_columns(['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle'])
dataset['additional_problems'] = dataset['additional_problems'].remove_columns(['generation_model', 'generated_solution', 'inference_mode', 'used_in_kaggle'])
full_data = concatenate_datasets([dataset['cot'], dataset['additional_problems']])
full_data = full_data.filter(remove_proofs, num_proc=20)

full_data.to_json("math-problems.jsonl")
```

Next, prepare the [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) to run on Slurm.
Here we assume that model is hosted on 16 H100 GPUs, but other GPU configurations are possible with corresponding
modifications to commands.

To download the model you can run the following from `/workspace` folder on Slurm.
We will also need [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) to use as the judge
for answer correctness.

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-0528 --local-dir DeepSeek-R1-0528
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir Qwen2.5-32B-Instruct
```

The next step is optional, but we recommend sharding the checkpoint to avoid very long loading time.

```python
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

cmd = (
    "python3 nemo_skills/conversion/save_sharded_state.py "
    "    --model-path=/workspace/DeepSeek-R1-0528 "
    "    --output=/workspace/DeepSeek-R1-0528-tp16 "
    "    --tensor-parallel-size=16 "
    "    --context-len=8192 "
    "    --trust-remote-code "
    "    --nnodes 2 "
    "    --dist-init-addr $SLURM_MASTER_NODE:20000 "
    "    --node-rank $SLURM_PROCID "
)

run_cmd(
    ctx=wrap_arguments(cmd),
    cluster="slurm",
    num_gpus=8,
    num_nodes=2,
    container="sglang",
    log_dir="/workspace/DeepSeek-R1-0528-tp16",
)
```

Finally, launch the data generation command. You can adjust `num_chunks` (how many jobs to launch in parallel) and
`dependent_jobs` (how many jobs to launch sequentially in case there is a fixed timeout on cluster) to fit your setup.

```python
from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

cluster = 'slurm'
tokens_to_generate = 32768
num_solutions = 16

# Main generation - this will take a lot of time and GPUs!
# You can select a subset of data to run on if you want to test things
generate(
    ctx=wrap_arguments(
        f"++prompt_config=generic/math "
        f"++inference.temperature=0.6 "
        f"++inference.tokens_to_generate={tokens_to_generate} "
    ),
    cluster=cluster,
    input_file="/workspace/open-reasoning/sdg/math-problems.jsonl",
    output_dir="/workspace/open-reasoning/sdg/solutions",
    expname="r1-0528-math-solutions",
    model="/workspace/DeepSeek-R1-0528-tp16",
    server_type="sglang",
    server_gpus=8,
    server_nodes=2,
    server_args=f"--load-format sharded_state --context-length {tokens_to_generate + 2000}",
    num_random_seeds=num_solutions,
    # set these according to your cluster configuration
    # num_chunks=N,
    # dependent_jobs=M,
)

# Judge step, this one is very fast as it just compares the predicted
# and expected answers for each solution, doesn't check reasoning
generate(
    ctx=wrap_arguments(""),
    cluster=cluster,
    generation_type="math_judge",
    input_dir=f"/workspace/open-reasoning/sdg/solutions",
    output_dir=f"/workspace/open-reasoning/sdg/solutions-judged",
    expname="r1-0528-math-solutions-judge",
    run_after="r1-0528-math-solutions",
    model="/workspace/Qwen2.5-32B-Instruct",
    server_type="sglang",
    server_gpus=8,
    num_random_seeds=num_solutions,
)

# We then change all "expected_answer" values to the majority
# from R1 if there is not a single match. While there are some really
# hard problems for which this will not be correct, we found that
# in most cases when R1 is not able to match GT answer even one time,
# the GT answer itself is not correct.
run_cmd(
    ctx=wrap_arguments(
        "python /nemo_run/code/recipes/openreasoning/scripts/use_majority_if_no_answer.py "
        "    /workspace/open-reasoning/sdg/solutions-judged "
        "    /workspace/open-reasoning/sdg/maj-if-no-correct "
    ),
    cluster=cluster,
    expname="change-to-majority-if-no-correct",
    run_after="r1-0528-math-solutions-judge",
    log_dir="/workspace/open-reasoning/sdg/maj-if-no-correct",
)

# Next we re-judge the data to keep matches with the new majority answer
# (should cover non-string match cases like 0.5 vs 1/2)
generate(
    ctx=wrap_arguments(""),
    cluster=cluster,
    generation_type="math_judge",
    input_dir=f"/workspace/open-reasoning/sdg/maj-if-no-correct",
    output_dir=f"/workspace/open-reasoning/sdg/maj-if-no-correct-judged",
    expname="r1-0528-math-solutions-judge-after-majority",
    run_after="change-to-majority-if-no-correct",
    model="/workspace/Qwen2.5-32B-Instruct",
    server_type="sglang",
    server_gpus=8,
    num_random_seeds=num_solutions,
)

# As the final step we convert this data to the format that can be used for SFT.
# This script will also filter anything not judged as correct
cmd = (
    "python -m nemo_skills.training.prepare_data "
    "    ++prompt_template=qwen-instruct "
    "    ++prompt_config=generic/math "
    "    ++input_files='/workspace/open-reasoning/sdg/maj-if-no-correct-judged/output-rs*.jsonl' "
    "    ++output_path=/workspace/open-reasoning/sft-data-math.jsonl "
    "    ++filters.drop_multi_boxed=false "
    "    ++filters.trim_prefix=false "
    "    ++filters.remove_no_think_tags=true "
    "    ++filters.remove_contaminated=false "  # OpenMathReasoning is already decontaminated
    "    ++filters.remove_len_outlier_solutions=false "
    "    ++filters.remove_len_outlier_problems=false "
    "    ++use_judgement=true "
)
run_cmd(
    ctx=wrap_arguments(cmd),
    cluster=cluster,
    log_dir="/workspace/open-reasoning/sft-data-math-logs",
    expname='prepare-for-sft-math',
    run_after="r1-0528-math-solutions-judge-after-majority",
)
```

The final data that's ready for training will then be available in `/workspace/open-reasoning/sft-data-math.jsonl`.

### GenSelect data

Coming soon!

## Code data

The code data was creating with exactly the same pipeline as used for [OpenCodeReasoning dataset](../opencodereasoning/dataset.md),
except the solutions are generated with [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528).

## Science data

We generate science problems using Qwen2.5-32B-Instruct and Qwen3-235B-A22B LLMs with the prompt below, using few-shot examples to demonstrate the format.
Questions are generated based on difficulty level, topic, and subtopic.
```yaml
system: ""

user: |-
  {examples}Write a question on {topic}, more specifically on {subtopic}, with up to four choices.
  Desired difficulty level: {difficulty}.
  For questions marked "intermediate", the question should require applying basic concepts independently or in simple combinations. The relationships between concepts should be straightforward and directly tied to the problem. Solutions should be accessible with foundational understanding and methodical effort.
  For questions marked "complex", the question should require identifying and combining multiple clear but non-obvious connections between concepts. The relationships may span different areas within the topic, requiring structured reasoning and careful application of techniques. Focus on recognizing patterns and synthesizing information rather than straightforward computation.
  For questions marked "hard", the question should involve intricate or abstract relationships between concepts. Solving them should demand advanced reasoning skills, mastery of topic-specific techniques, and the ability to deal with ambiguities or edge cases. Solutions often require navigating multiple layers of logic or constraints.
  For questions marked "brutal", the question should push problem-solving to the limit, requiring mastery of nuanced relationships across diverse concepts. It should demand precise reasoning, significant abstraction, and fluency in advanced techniques. Problems may include nested or multi-step dependencies, where overlooking a detail can derail the solution.
  For questions marked "borderline unsolvable", the question should challenge even experts, involving exceptionally intricate or unconventional problem structures. It should combine deep abstraction, subtle constraints, and highly interconnected concepts. While solutions exist, they often require extensive expertise and innovative approaches. Avoid speculative or unsolvable topics (e.g., "Does God exist?" or "How can time travel be achieved?").
  Ensure that one of the four choices is correct. You are not required to specify which one is correct, but your question must include a valid answer within the given choices.
```
Full dataset used for this effort is available at [HuggingFace](https://huggingface.co/datasets/nvidia/OpenScience).
Note: HuggingFace version includes questions generated with Qwen2.5-72B-Instruct, which are not used for OpenReasoning.

The next step is to augment these problems using the prompt below, with few-shot examples to demonstrate the format of the output.
```yaml
system: ""

user: |-
  {examples}
  Write a new multiple-choice question inspired by the given one. Make the new question reasonable and solvable.
  Ensure that one of the choices is correct. You are not required to specify which one is correct, but your question must include a valid answer within the given choices.
  Start directly with the question and DO NOT include any phrases such as "Here is a new multiple-choice question inspired by a given one".
  After the question is completed finish your response right away.
  Question:
  {question}
  Write another multiple-choice question inspired by this one.
  Don't just change the numbers and context, but try to create a question that requires another approach to solve.
```

Next, we generate solutions for these problems.
We use DeepSeek-R1-0528 to generate solutions with parameters as described in the math section above.

The final step is to apply majority voting over the solutions generated in the previous step to obtain the final dataset.