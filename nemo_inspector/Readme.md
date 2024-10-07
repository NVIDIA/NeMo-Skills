# NeMo Inspector tool

## Demo
This is a tool for data analysis, consisting of two pages: "Inference" and "Analyze".

### Overview of the tool
[![Demo of the tool](/nemo_inspector/images/demo.png)](https://www.youtube.com/watch?v=EmBFEl7ydqE)

### Demo of the Inference Page
[![Demo of the inference page](/nemo_inspector/images/inference_page.png)](https://www.youtube.com/watch?v=6utSkPCdNks)

### Demo of the Analyze Page
[![Demo of the analyze page](/nemo_inspector/images/analyze_page.png)](https://www.youtube.com/watch?v=cnPyDlDmQXg)

## Getting Started
Before using this tool, follow the instructions in [prerequisites.md](/docs/prerequisites.md), and install requirements:
```shell
pip install -r requirements/inspector.txt
```
You can adjust parameters in the [inspector_config.yaml](/nemo_inspector/settings/inspector_config.yaml) file or via the command line. Use the following command to launch the program (all parameters are optional):
```shell
python nemo_inspector/nemo_inspector.py \
++server.host=<host>
```
For the "Inference" page, launch the server with the model (see [inference.md](/docs/inference.md)), specify `host` and, if necessary, `ssh_key` and `ssh_server`.

## Inference page
This page enables the analysis of model answers based on different parameters. It offers two modes: "Chat", "Run one sample".

- **Chat** mode facilitates a conversation with the model and requires minimal parameter setup.
- **Run one sample** mode allows you to send a single question to the model. It can be a question from the dataset (with parameters `input_file` or `dataset` and `split`) or a custom question. The answer is validated by comparing it with the `expected_answer` field.

## Analyze page
To use the Analyze page, specify paths to the generations you want to use (if not obtained through the "Inference" page). You can pass parameters via the command line with `++inspector_params.model_prediction.generation1='/some_path/generation1/output-greedy.jsonl'` or add them in an additional config file.

```yaml
inspector_params:
  model_prediction:
    generation1: /some_path/generation1/output-greedy.jsonl
    generation2: /some_path/generation2/output-rs*.jsonl
```

The tool also supports comparison of multiple generations (e.g.
 `generation2` in the config above). All files satisfying the given pattern will be considered for analysis.

On this page, you can sort, filter, and compare generations. You can also add labels to the data and save your modified, filtered, and sorted generation by specifying `save_generations_path`.

### Filtering
You can create custom functions to filter data. There are two modes: Filter Files mode and Filter Questions mode.

#### Filter Files mode
In this mode the functions will filter each sample from different files. It should take a dictionary containing keys representing generation names and values as JSON data from your generation.

Custom filtering functions should return a Boolean value. For instance:

```python
def custom_filtering_function(error_message: str) -> bool:
   # Your code here
   return result

custom_filtering_function(data['generation1']['error_message']) # This line will be used for filtering
```
The last line in the custom filtering function will be used for data filtering; all preceding code within the function is executed but does not directly impact the filtering process.

To apply filters for different generations, separate expressions with '&&' symbols.
 ```python
 data['generation1']['is_correct'] && not data['generation2']['is_correct']
 ```
 Do not write expressions for different generations without separators in this mode.

#### Filter Questions mode
In this mode the function will filter each question. Files will not be filtered. It should take a dictionary containing keys representing generation names and a list of values as JSON data from your generation from each file.

In this mode you should not use the && separator. For instance, an example from the previous mode can be written like this:
 ```python
 data['generation1'][0]['is_correct'] and not data['generation2'][0]['is_correct']
 # Filter questions where the first file of the first generation contains a correct solution and the first file from the second generation contains a wrong solution
 ```
 or like this:
  ```python
 data['generation1'][0]['correct_responses'] == 1 and data['generation2'][0]['correct_responses'] == 0
 # Custom Statistics are dublicated in all JSONs. So here, 'correct_responses' value will be the same for all file for a specific generation and question
 ```
 In this mode you can also compare fields of different generations
   ```python
 data['generation1'][0]['is_correct'] != data['generation2'][0]['is_correct']
 ```
 These examples can not be used in the Filter Files mode

### Sorting
Sorting functions operate similarly to filtering functions, with a few distinctions:

1. Sorting functions operate on individual data entries rather than on dictionaries containing generation name keys.
2. Sorting functions cannot be applied across different generations simultaneously.

Here is an example of a correct sorting function:

```python
def custom_sorting_function(generation: str):
    return len(generation)

custom_sorting_function(data['generation'])
```

### Statistics
There are two types of statistics: "Custom Statistics" and "General Custom Statistics". Custom statistics apply to different samples of a single question. There are some default custom statistics: "correct_responses", "wrong_responses", and "no_responses". General Custom Statistics apply to each sample across all questions. Default general custom statistics - "dataset size", "overall number of samples" and "generations per sample"

![stats](/nemo_inspector/images/stats.png)

You can define your own Custom and General Custom Statistics functions. For Custom Statistics, the function should take an array of JSONs from each file. For General Custom Statistics, the function should take a list of lists of dictionaries, where the first dimension corresponds to the question index and the second dimension to the file index.

Here are examples of correct functions for both statistics types:

```python
# Custom Statistic function
def unique_error_counter(datas):
    unique_errors = set()
    for data in datas:
        unique_errors.add(data.get('error_message'))
    return len(unique_errors)

def number_of_runs(datas):
    return len(datas)

# Mapping function names to functions
{'unique_errors': unique_error_counter, "number_of_runs": number_of_runs}
```
```python
# General Custom Statistic function
def overall_unique_error_counter(datas):
    unique_errors = set()
    for question_data in datas:
        for file_data in question_data:
            unique_errors.add(file_data.get('error_message'))
    return len(unique_errors)

# Mapping function names to functions
{'unique_errors': overall_unique_error_counter}
```
Note that the last line in both statistic sections should be a dictionary where each key is the function's name and the corresponding value is the function itself.
