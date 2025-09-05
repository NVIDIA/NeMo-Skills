from pathlib import Path

import ojbench

ojbench.init(
    problem_dirs=[
        Path("OJBench_testdata/NOI"),
        Path("OJBench_testdata/ICPC"),
    ]
)

responses = [
    {
        "id": 1234,
        "prompt": "Write a function to add two integers...",
        "dataset": "ICPC",
        "language": "python",
        "difficulty": "easy",
        "content": "Here is the code:\n```python\ndef add(a, b): return a + b\n```",
    }
]
results = ojbench.judge_jsonl_data(responses, num_workers=4)
