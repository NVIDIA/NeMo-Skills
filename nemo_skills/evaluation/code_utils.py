def preprocess_code(generation_dict: dict):
    completion = generation_dict['generation']
    completion = completion.strip()
    completion = completion.replace("\r", "")
    if '```' in completion:
        if '```python' in completion:
            def_line = completion.index('```python') + len('```python')
        else:
            def_line = completion.index('```') + len('```')
        completion = completion[def_line:].strip()
        completion = completion.replace('```python', '')
        try:
            next_line = completion.index('```')
            completion = completion[:next_line].strip()
        except:
            print(completion)
            print("================\n")

    if completion.startswith(" "):
        completion = completion.strip()

    generation_dict["completion"] = completion
    return generation_dict
