# LLM-as-a-judge for math evaluation

!!! info

    This pipeline starting script is [nemo_skills/pipeline/generate.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/pipeline/generate.py)

    All extra parameters are passed to [nemo_skills/inference/llm_math_judge.py](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/inference/llm_math_judge.py)

When evaluating complex mathematical questions, it's very hard to have a rule-based symbolic comparison system.
While we do perform such comparison by default, for most accurate results it's best to use LLM-as-a-judge pipeline.
E.g. symbolic comparison can perform very inaccurately for multi-choice questions where an answer might either be
one of the letters or an expression corresponding to that letter.

If you have an output of the [evaluation script](evaluation.md) on e.g. math benchmark, you can run LLM-as-a-judge
in the following way (assuming you have `/workspace` mounted in your [cluster config](../basics/prerequisites.md#cluster-configs)
and evaluation output available in `/workspace/test-eval/eval-results`).

```bash
ns generate \
    --generation_type=math_judge \
    --cluster=local \
    --model=gpt-4o \
    --server_type=openai \
    --server_address=https://api.openai.com/v1 \
    --output_dir=/workspace/test-eval-judge/eval-results/math \
    ++input_dir=/workspace/test-eval/eval-results/math
```

This will run the judge pipeline on the data inside `eval-results/math` folder and judge solutions from `output.jsonl` file.
If you ran the benchmark with N samples (e.g. using `math:8`) and want to judge all of them, add `--num_random_seeds=8`.
Note that if you want to judge both greedy generations and samples, you'd need to run the command two times.

In this example we use gpt-4o from OpenAI, but you can use Llama-405B (that you can host on cluster yourself) or any
other models. If you have multiple benchmarks, you would need to run the command multiple times.
After the judge pipeline has finished, you can see the results by running

```bash
ns summarize_results /workspace/test-eval-judge --cluster local
```

Which should output something like this

```
------------------------------------------------- aime24 ------------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
greedy          | 30          | 20.00            | 20.00         | 20.00        | 20.00       | 13.33


------------------------------------------------- gsm8k -------------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
greedy          | 1319        | 95.00            | 95.75         | 95.00        | 95.75       | 0.00


-------------------------------------------------- math -------------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
greedy          | 5000        | 67.32            | 67.88         | 67.02        | 68.18       | 2.64


------------------------------------------------- amc23 -------------------------------------------------
evaluation_mode | num_entries | symbolic_correct | judge_correct | both_correct | any_correct | no_answer
greedy          | 40          | 47.50            | 47.50         | 47.50        | 47.50       | 7.50
```

If you want to see where symbolic comparison differs from judge comparison, run with `--debug` option.

We use the following [judge prompt](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/prompt/config/judge/math.yaml)
by default, but you can customize it the same way as you [customize any other prompt](../basics/prompt-format.md).