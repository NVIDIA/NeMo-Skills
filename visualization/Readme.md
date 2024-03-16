# Data Explorer tool

## Demo
This is a tool for data analysis, consisting of two pages: "Inference" and "Analyze".

[![Demo of the tool](/visualization/images/demo.png)](https://youtu.be/SWKNovw55HQ)


## Getting Started
Before using this tool, follow the instructions in [prerequisites.md](/docs/prerequisites.md), and install requirements:
```shell
pip install -r visualization/requirements.txt
```
You can adjust parameters in the [visualization_config.yaml](/visualization/settings/visualization_config.yaml) file or via the command line. Use the following command to launch the program (all parameters are optional):
```shell
python visualization/data_explorer.py \
++server.host=<host>
```
For the "Inference" page, launch the server with the model (see [inference.md](/docs/inference.md)), specify `host` and, if necessary, `ssh_key` and `ssh_server`.

## Inference page
This page enables the analysis of model answers based on different parameters. It offers three modes: "Chat", "Run one sample", and "Run whole dataset".

- **Chat** mode facilitates a conversation with the model and requires minimal parameter setup.
- **Run one sample** mode allows you to send a single question to the model. It can be a question from the dataset (with parameters `data_file` or `dataset` and `split_name`) or a custom question. The answer is validated by comparing it with the `expected_answer` field.
- **Run whole dataset** mode lets you launch the model with chosen parameters on the entire dataset. Results are saved in `visualization/results/output-greedy.jsonl` and `visualization/results/metrics-greedy.jsonl`. If the "use random seed range" flag is enabled, each answer will be sampled with multiple random seeds in the range from `start_random_seed` to `end_random_seed`. After generation is done, you can review the results on the "Analyze" page. The parameters used for dataset launch are also recorded in the `visualization/results/parameters.jsonl` file and displayed on the "Analyze" page.

## Analyze page
To use the Analyze page, specify paths to the datasets you want to use (if not obtained through the "Inference" page). You can pass parameters via the command line with `++visualization_params.model_prediction.model1='/some_path/model1/output-greedy.jsonl'` or add them in an additional config file.

```yaml
visualization_params:
  model_prediction:
    model1: /some_path/model1/output-greedy.jsonl
    model2: /some_path/model2/output-rs*.jsonl
```

The tool also supports comparison of multiple model outputs (e.g. 
 `model1` and `model2` in the config above). All files satisfying the given pattern will be considered for analysis.

On this page, you can sort, filter, and compare model outputs. You can also add labels to the data and save your modified, filtered, and sorted dataset by specifying `save_dataset_path`.

### Filtering
You can create custom functions to filter data. These functions should take a dictionary containing keys representing model names and values as JSON data from your dataset.

Custom filtering functions should return a Boolean value. For instance:

```python
def custom_filtering_function(error_message: str) -> bool:
   # Your code here
   return result

custom_filtering_function(data['model1']['error_message']) # This line will be used for filtering
```
The last line in the custom filtering function will be used for data filtering; all preceding code within the function is executed but does not directly impact the filtering process.

To apply different filters for different models, separate expressions with '&&' symbols. 
 ```python
 data['model1']['is_correct'] && not data['model2']['is_correct']
 ```
 Do not write expressions for different models without separators.


### Sorting
Sorting functions operate similarly to filtering functions, with a few distinctions:

1. Sorting functions operate on individual data entries rather than on dictionaries containing model name keys.
2. Sorting functions cannot be applied across different models simultaneously.

Here is an example of a correct sorting function:

```python
def custom_sorting_function(generated_solution: str):
    return len(generated_solution)

custom_sorting_function(data['generated_solution'])
```

### Statistics
There are two types of statistics: "Custom Statistics" and "General Custom Statistics". Custom statistics apply to different runs of a single question. There are some default custom statistics: "correct_responses", "wrong_responses", and "no_responses". General Custom Statistics apply to each run across all questions. Default general custom statistic - "overall number of runs"

![stats](/visualization/images/stats.png)

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
