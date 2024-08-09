import hashlib
import subprocess


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def test_code_files():
    output_file = "tests/data/processed_output.jsonl"
    subprocess.run(
        [
            "python",
            "nemo_skills/finetuning/prepare_sft_data.py",
            "prediction_jsonl_files='tests/data/output-rs*.test'",
            f"output_path={output_file}",
            "filters.drop_multi_boxed=true",
            "filters.drop_broken_code=true",
            "filters.trim_solutions=true",
            "filters.drop_incorrect_arithmetic=false",
            "filters.split_arithmetic=false",
            "filters.code_text_filter=null",
            "num_output_samples=null",
            "downsampling_method=null",
            "do_shuffle=false",
        ],
        check=True,
    )

    expected_md5 = "5feef230065b71b0ee4debe54fde62bd"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/finetuning/prepare_sft_data.py"


def test_openmathinstruct():
    output_file = "tests/data/processed_output.jsonl"

    subprocess.run(
        [
            "python",
            "nemo_skills/finetuning/prepare_sft_data.py",
            "preprocessed_dataset_files='tests/data/openmathinstruct.test'",
            f"output_path={output_file}",
            "filters.drop_multi_boxed=true",
            "filters.drop_broken_code=true",
            "filters.trim_solutions=true",
            "filters.drop_incorrect_arithmetic=false",
            "filters.split_arithmetic=false",
            "filters.code_text_filter=any_code",
            "num_output_samples=32",
            "downsampling_method=fair",
            "do_shuffle=true",
        ],
        check=True,
    )

    expected_md5 = "31378960fd2857876b6d14f5bd762db6"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/finetuning/prepare_sft_data.py"


def test_code_sft_data():
    output_file = "tests/data/code_processed_output.jsonl"
    subprocess.run(
        [
            "python",
            "nemo_skills/finetuning/prepare_sft_data.py",
            "--config-name=prepare_code_sft_data",
            "preprocessed_dataset_files='tests/data/code-output.test'",
            f"output_path={output_file}",
            "filters.drop_incorrect_code_blocks=false",
            "generation_suffix='\"<|eot_id|>\"'",
        ],
        check=True,
    )

    expected_md5 = "51ad8f704b8c550ddf2027bf9984e154"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/finetuning/prepare_sft_data.py"
