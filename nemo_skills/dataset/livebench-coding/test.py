from datasets import load_dataset

ds = load_dataset("livebench/coding", split="test", trust_remote_code=True)
print(ds)

for item in ds:
    if item["task"] == "coding_completion":
        assert len(item["turns"]) == 1
        assert item["partial_solution"] + "\n" + item["remainder"] == item["solution"]
        print("==" * 20)
        print(item["turns"][0])
        # break
