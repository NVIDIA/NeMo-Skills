# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from layouts.analyze_page_layouts import (
    get_compare_test_layout,
    get_few_shots_layout,
    get_models_options_layout,
    get_stats_text,
    get_utils_layout,
)
from layouts.base_layouts import (
    get_input_group_layout,
    get_main_page_layout,
    get_results_content_layout,
    get_selector_layout,
    get_single_prompt_output_layout,
    get_switch_layout,
    get_text_area_layout,
    get_utils_field_representation,
)
from layouts.run_prompt_page_layouts import (
    get_few_shots_by_id_layout,
    get_query_params_layout,
    get_run_mode_layout,
    get_run_test_layout,
)
from layouts.table_layouts import (
    get_detailed_answer_column,
    get_filter_answers_layout,
    get_filter_layout,
    get_labels,
    get_model_answers_table_layout,
    get_models_selector_table_cell,
    get_row_detailed_inner_data,
    get_single_prompt_output_layout,
    get_sorting_answers_layout,
    get_sorting_layout,
    get_stats_layout,
    get_table_data,
    get_table_detailed_inner_data,
)
