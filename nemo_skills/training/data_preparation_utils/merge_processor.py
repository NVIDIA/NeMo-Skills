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

import logging

import hydra
from sdp.processors.base_processor import DataEntry

from nemo_skills.training.data_preparation_utils.filters import BaseFilter

LOG = logging.getLogger(__name__)


class MergeProcessor(BaseFilter):
    def __init__(self, processor_configs: list[dict], **kwargs):
        super().__init__(**kwargs)
        self.processors = []
        for processor_cfg in processor_configs:
            if not processor_cfg.pop('should_run', True):
                continue
            processor_cfg['input_manifest_file'] = None
            processor_cfg['output_manifest_file'] = None
            processor = hydra.utils.instantiate(processor_cfg)
            # running runtime tests to fail right-away if something is not
            # matching users expectations
            processor.test()
            self.processors.append(processor)

    def process_dataset_entry(self, data_entry: dict) -> list[DataEntry]:
        # TODO: properly track and report metrics for all processors
        num_modified = 0
        for processor in self.processors:
            data_entry = processor.process_dataset_entry(data_entry)[0]
            if data_entry.metrics.get('num_modified', 0) > 0:
                num_modified = 1
            data_entry = data_entry.data
            if data_entry is None:
                return [DataEntry(data=None, metrics=dict(num_removed=1))]

        return [DataEntry(data=data_entry, metrics=dict(num_modified=num_modified))]

    def finalize(self, metrics: list):
        LOG.info("Number of entries after processing: %d", self.number_of_entries)

        if not metrics:
            return

        if 'num_removed' in metrics[0]:
            num_removed_entries = sum(metric.get('num_removed', 0) for metric in metrics)
            LOG.info("Number of removed entries: %d", num_removed_entries)

        if 'num_modified' in metrics[0]:
            num_modified_entries = sum(metric.get('num_modified', 0) for metric in metrics)
            LOG.info("Number of modified entries: %d", num_modified_entries)
