import json

path = "/home/dgitman/tmp/NeMo-Skills/nemo_skills/dataset/mmlu_boxed/test_copy.jsonl"
save_path = "/home/dgitman/tmp/NeMo-Skills/nemo_skills/dataset/mmlu_boxed/test.jsonl"

with open(path) as fin, open(save_path, "w") as fout:
    for line in fin:
        sample = json.loads(line)
        sample['problem'] += '\n' + sample['options']
        sample['subset_for_metrics'] = sample['subject']
        fout.write(json.dumps(sample) + '\n')
