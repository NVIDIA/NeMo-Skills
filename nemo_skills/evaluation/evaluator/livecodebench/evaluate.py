import json
from datetime import datetime

from nemo_skills.evaluation.evaluator.livecodebench.code_generation import load_code_generation_dataset_from_file
from nemo_skills.evaluation.evaluator.livecodebench.metrics import codegen_metrics
from nemo_skills.evaluation.evaluator.livecodebench.pass_k_utils import extract_instance_results


def evaluate(
    custom_output_file: str,
    release_version: str = "release_latest",
    k_list=[1],
    language: str = "python",
    test_file: str = None,
    num_process_evaluate: int = 12,
    debug: bool = False,
    timeout: int = 6,
):
    assert test_file is not None
    custom_outputs = dict()
    with open(custom_output_file, "r") as f:
        for line in f:
            output = json.loads(line)
            if "question_id" not in output:
                assert "task_id" in output
                output["question_id"] = output.pop("task_id")
            custom_outputs[output["question_id"]] = output

    benchmark = load_code_generation_dataset_from_file(test_file, language)
    benchmark = [problem for problem in benchmark if problem.question_id in custom_outputs]
    assert len(custom_outputs) == len(benchmark), f"{len(custom_outputs)} != {len(benchmark)}"
    assert all(isinstance(custom_output, dict) for custom_output in custom_outputs.values())

    save_results, combined_results = [], []
    for instance in benchmark:
        code_list = custom_outputs[instance.question_id]["code_list"]
        output = instance.insert_output(code_list, code_list)
        save_results.append(output)
        combined_results.append((code_list, code_list))

    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        k_list=k_list,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
        debug=debug,
        language=language,
    )

    graded = extract_instance_results(metrics[1])

    metadatas = metrics[2]
    save_eval_results = [
        instance.insert_output_evaluation(outputs_list, extracted_list, graded_list, metadata=meta)
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    # save_eval_results
    output_results = dict()
    output_results["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    for k in metrics[0]:
        if k.startswith("pass@"):
            print(f"{k}: {metrics[0][k]}")
            output_results[k] = metrics[0][k]

    output_results["detail_pass@1"] = dict()
    output_results["eval"] = dict()
    difficulty_wise_pass_at_1 = dict()
    for r in save_eval_results:
        output_results["eval"][r["question_id"]] = r
        if r["difficulty"] not in difficulty_wise_pass_at_1:
            difficulty_wise_pass_at_1[r["difficulty"]] = []
        difficulty_wise_pass_at_1[r["difficulty"]].append(r["pass@1"])

    for tag, v in difficulty_wise_pass_at_1.items():
        pass_at_1 = sum(v) / len(v)
        print(f"{tag} pass@1: {pass_at_1}")
        output_results["detail_pass@1"][tag] = pass_at_1

    with open(custom_output_file[:-6] + "_eval_results.json", "w") as f:
        json.dump(output_results, f, indent=4)


def main():
    from fire import Fire

    Fire(evaluate)


if __name__ == "__main__":
    main()
