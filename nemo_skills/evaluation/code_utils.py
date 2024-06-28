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

    if "__name__ == \"__main__\"" in completion:
        next_line = completion.index('if __name__ == "__main__":')
        completion = completion[:next_line].strip()
        # print(completion)

    if "# Example usage" in completion:
        # print(completion)
        next_line = completion.index('# Example usage')
        completion = completion[:next_line].strip()

    if "[/CODE]" in completion:
        next_line = completion.index('[/CODE]')
        completion = completion[:next_line].strip()

    if "[SOL]" in completion:
        next_line = completion.index('[SOL]')
        completion = completion[:next_line].strip()

        if "[/SOL]" in completion:
            next_line = completion.index('[/SOL]')
            completion = completion[:next_line].strip()

    if "[/INST]" in completion:
        next_line = completion.index('[/INST]')
        completion = completion[:next_line].strip()

    if "Explanation:" in completion:
        next_line = completion.index('Explanation:')
        completion = completion[:next_line].strip()

    if "<extra_id_1>" in completion:
        next_line = completion.index('<extra_id_1>')
        completion = completion[:next_line].strip()

    if "here" in completion.lower() and "code" in completion.lower() and "def" in completion:
        # Find first def after "Here is the" and remove everything before that
        next_line = completion.index('def')
        completion = completion[next_line:].strip()

    if completion.startswith(" "):
        completion = completion.strip()

    generation_dict["completion"] = completion
    return generation_dict
