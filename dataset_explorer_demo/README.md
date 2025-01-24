# Dataset Explorer Demo

1. Download data TBD
2. Retrieve similar questions from OpenMathInstruct2. Do it for all benchmarks you want to compare against.
   Assuming you're running from this folder.

   ```
   python -m nemo_skills.inference.retrieve_similar \
       ++retrieve_from=./data.jsonl \
       ++compare_to="../nemo_skills/dataset/<benchmark>/test.jsonl" \
       ++output_file=./similar-retrieved-openmath2/<benchmark>.jsonl \
       ++top_k=5
   ```

3. Let's do the same for original MATH training set to get a sense of whether OpenMathInstruct-2 is in the same
   distribution or not.

   ```
   python -m nemo_skills.inference.retrieve_similar \
       ++retrieve_from=../nemo_skills/dataset/math/train_full.jsonl \
       ++compare_to="../nemo_skills/dataset/<benchmark>/test.jsonl" \
       ++output_file=./similar-retrieved-math-train/<benchmark>.jsonl \
       ++top_k=5
   ```

4. Start the Gradio demo.

   ```
   python visualize_similar.py
   ```