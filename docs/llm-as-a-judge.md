# LLM-as-a-judge for math evaluation

Make sure to complete [prerequisites](/docs/prerequisites.md).

Please refer to the following docs if you have questions about:
- [Prompt format](/docs/prompt-format.md)
- [Generation parameters](/docs/common-parameters.md)
- [How to self-host models](/docs/generation.md)

When evaluating complex mathematical questions, it's very hard to have a rule-based symbolic comparison system.
While we do perform such comparison by default, for most accurate results it's best to use LLM-as-a-judge pipeline.
E.g. symbolic comparison can perform very inaccurately for multi-choice questions where an answer might either be
one of the letters or an expression corresponding to that letter.

If you have an output of [evaluation script](/docs/evaluation.md) on one of the math datasets, you can run LLM-as-a-judge
in the following way (assuming you have `/workspace` mounted in your cluster config and evaluation output available in
`/workspace/test-eval/eval-results`).

```
ns llm_math_judge \
    --cluster=local \
    --model=gpt-4o \
    --server_type=openai \
    --server_address=https://api.openai.com/v1 \
    --input_files="/workspace/test-eval/eval-results/**/output*.jsonl"
```

This will run the judge pipeline on all benchmarks inside `eval-results` folder and evaluation all `output*.jsonl` files.
In this example we use gpt-4o from OpenAI, but you can use Llama-405B (that you can host on cluster yourself) or any
other models. After the judge pipeline has finished, you can see the results by running

```
ns summarize_results /workspace/test-eval/ --cluster local
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

We use the following [judge prompt](/nemo_skills/prompt/config/judge/math.yaml) by default, but you can customize
it the same way as you [customize any other prompt](/docs/prompt-format.md).