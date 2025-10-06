# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the declarative pipeline system."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline


class TestCommand:
    """Test Command class functionality."""

    def test_command_basic_string(self):
        """Test creating a Command with a simple string."""
        cmd = Command(command="echo hello", name="test")
        assert cmd.name == "test"
        assert cmd.container == "nemo-skills"
        assert cmd.gpus is None
        assert cmd.nodes == 1

    def test_command_with_metadata(self):
        """Test Command with metadata passed separately."""
        cmd = Command(command="echo hello", name="server", metadata={"port": 8080, "log_prefix": "server"})
        assert cmd.metadata["port"] == 8080
        assert cmd.metadata["log_prefix"] == "server"
        # Command gets wrapped with working_dir by default
        assert "echo hello" in cmd.command

    def test_command_with_callable(self):
        """Test Command with callable that returns tuple."""

        def make_cmd():
            return ("echo world", {"port": 5000})

        cmd = Command(command=make_cmd, name="dynamic")
        assert callable(cmd.command)
        assert cmd.name == "dynamic"

    def test_command_prepare_for_execution_string(self):
        """Test prepare_for_execution with string command."""
        cmd = Command(command="python script.py", gpus=2, name="test")
        cluster_config = {"executor": "local", "containers": {}}

        final_cmd, exec_config = cmd.prepare_for_execution(cluster_config)

        assert "python script.py" in final_cmd
        assert exec_config["num_gpus"] == 2
        assert exec_config["num_nodes"] == 1
        assert exec_config["num_tasks"] == 1

    def test_command_prepare_for_execution_callable(self):
        """Test prepare_for_execution with callable command."""

        def make_cmd():
            return "echo test"

        cmd = Command(command=make_cmd, name="test")
        cluster_config = {"executor": "local", "containers": {}}

        final_cmd, exec_config = cmd.prepare_for_execution(cluster_config)

        assert final_cmd == "echo test"

    def test_command_prepare_for_execution_callable_with_metadata(self):
        """Test prepare_for_execution with callable returning tuple."""

        def make_cmd():
            return ("echo metadata", {"num_tasks": 4, "environment": {"VAR": "value"}})

        cmd = Command(command=make_cmd, name="test")
        cluster_config = {"executor": "local", "containers": {}}

        final_cmd, exec_config = cmd.prepare_for_execution(cluster_config)

        assert final_cmd == "echo metadata"
        assert exec_config["num_tasks"] == 4
        assert exec_config["environment"]["VAR"] == "value"

    def test_command_meta_ref(self):
        """Test meta_ref for accessing metadata."""
        cmd = Command(command="echo test", name="server", metadata={"port": 8080, "host": "localhost"})

        assert cmd.meta_ref("port") == "8080"
        assert cmd.meta_ref("host") == "localhost"

    def test_command_meta_ref_missing_key(self):
        """Test meta_ref with missing key raises KeyError."""
        cmd = Command(command="echo test", name="test")

        with pytest.raises(KeyError, match="Metadata key 'port' not found"):
            cmd.meta_ref("port")

    def test_command_hostname_ref_none(self):
        """Test hostname_ref returns localhost when het_group_index is None."""
        cmd = Command(command="echo test", name="test")
        assert cmd.het_group_index is None
        assert cmd.hostname_ref() == "127.0.0.1"

    def test_command_hostname_ref_heterogeneous(self):
        """Test hostname_ref returns SLURM variable when het_group_index is set."""
        cmd = Command(command="echo test", name="test")
        cmd.het_group_index = 2

        hostname = cmd.hostname_ref()
        assert "$SLURM_JOB_NODELIST_HET_GROUP_2" in hostname
        assert "scontrol" in hostname

    def test_command_with_installation_command(self):
        """Test Command with installation_command."""
        cmd = Command(command="python script.py", installation_command="pip install package", name="test")
        cluster_config = {"executor": "local", "containers": {}}

        final_cmd, _ = cmd.prepare_for_execution(cluster_config)

        # Installation command should be wrapped around the main command
        assert "pip install package" in final_cmd
        assert "python script.py" in final_cmd

    def test_command_env_vars_wrapping(self):
        """Test that env_vars and working_dir are applied to string commands."""
        cmd = Command(
            command="python script.py",
            env_vars={"MY_VAR": "value"},
            working_dir="/custom/path",
            name="test",
        )

        # The command should be wrapped with env setup
        assert "export MY_VAR=value" in cmd.command
        assert "cd /custom/path" in cmd.command


class TestCommandGroup:
    """Test CommandGroup class functionality."""

    def test_commandgroup_basic(self):
        """Test creating a basic CommandGroup."""
        cmd1 = Command(command="echo 1", name="cmd1")
        cmd2 = Command(command="echo 2", name="cmd2")

        group = CommandGroup(commands=[cmd1, cmd2], name="test_group")

        assert group.name == "test_group"
        assert len(group.commands) == 2
        assert group.hardware is not None

    def test_commandgroup_with_hardware(self):
        """Test CommandGroup with HardwareConfig."""
        cmd = Command(command="echo test", name="cmd")
        hardware = HardwareConfig(partition="batch", time_min="01:00:00", num_gpus=8)

        group = CommandGroup(commands=[cmd], hardware=hardware, name="gpu_group")

        assert group.hardware.partition == "batch"
        assert group.hardware.time_min == "01:00:00"
        assert group.hardware.num_gpus == 8

    def test_commandgroup_with_log_dir(self):
        """Test CommandGroup with log_dir."""
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], log_dir="/logs/test", name="group")

        assert group.log_dir == "/logs/test"


class TestPipeline:
    """Test Pipeline class functionality."""

    def test_pipeline_with_groups(self):
        """Test Pipeline with groups parameter (shorthand format)."""
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group")
        cluster_config = {"executor": "local", "containers": {}}

        pipeline = Pipeline(name="test_pipeline", cluster_config=cluster_config, groups=[group])

        assert pipeline.name == "test_pipeline"
        assert len(pipeline.jobs) == 1
        assert "group" in pipeline.jobs[0]

    def test_pipeline_with_jobs(self):
        """Test Pipeline with jobs parameter (full format with dependencies)."""
        cmd1 = Command(command="echo 1", name="cmd1")
        group1 = CommandGroup(commands=[cmd1], name="group1", log_dir="/logs")

        cmd2 = Command(command="echo 2", name="cmd2")
        group2 = CommandGroup(commands=[cmd2], name="group2", log_dir="/logs")

        jobs = [
            {"name": "job1", "group": group1},
            {"name": "job2", "group": group2, "dependencies": ["job1"]},
        ]
        cluster_config = {"executor": "local", "containers": {}}

        pipeline = Pipeline(name="test_pipeline", cluster_config=cluster_config, jobs=jobs)

        assert pipeline.name == "test_pipeline"
        assert len(pipeline.jobs) == 2

    def test_pipeline_cannot_specify_both_groups_and_jobs(self):
        """Test that Pipeline raises error when both groups and jobs are specified."""
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group")
        jobs = [{"name": "job", "group": group}]
        cluster_config = {"executor": "local", "containers": {}}

        with pytest.raises(ValueError, match="Cannot specify both 'groups' and 'jobs'"):
            Pipeline(name="test", cluster_config=cluster_config, groups=[group], jobs=jobs)

    def test_pipeline_must_specify_either_groups_or_jobs(self):
        """Test that Pipeline raises error when neither groups nor jobs are specified."""
        cluster_config = {"executor": "local", "containers": {}}
        with pytest.raises(ValueError, match="Must specify either 'groups' or 'jobs'"):
            Pipeline(name="test", cluster_config=cluster_config)

    def test_pipeline_with_run_after(self):
        """Test Pipeline with run_after parameter."""
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group")
        cluster_config = {"executor": "local", "containers": {}}

        pipeline = Pipeline(name="test", cluster_config=cluster_config, groups=[group], run_after="other_exp")

        assert pipeline.run_after == "other_exp"

    def test_pipeline_with_run_after_list(self):
        """Test Pipeline with run_after as list."""
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group")
        cluster_config = {"executor": "local", "containers": {}}

        pipeline = Pipeline(name="test", cluster_config=cluster_config, groups=[group], run_after=["exp1", "exp2"])

        assert pipeline.run_after == ["exp1", "exp2"]

    def test_pipeline_cluster_config_passed_directly(self):
        """Test that cluster_config is passed directly (no more string resolution)."""
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group")
        cluster_config = {"executor": "local", "containers": {}}

        pipeline = Pipeline(name="test", cluster_config=cluster_config, groups=[group])

        # cluster_config is stored as-is
        assert pipeline.cluster_config == cluster_config


class TestPipelineExecution:
    """Test Pipeline execution and job management."""

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    @patch("nemo_skills.pipeline.utils.declarative.run_exp")
    def test_pipeline_run_basic(self, mock_run_exp, mock_env_vars, mock_get_exp):
        """Test basic pipeline execution."""
        # Setup mocks
        mock_config = {
            "executor": "none",
            "containers": {"nemo-skills": "container:latest"},
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}

        mock_exp = MagicMock()
        mock_exp.add.return_value = "task_handle_1"
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        # Create pipeline
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group", log_dir="/logs")
        pipeline = Pipeline(name="test", cluster_config=mock_config, groups=[group], skip_hf_home_check=True)

        # Run pipeline
        result = pipeline.run(dry_run=True)

        # Verify
        assert result == mock_exp
        mock_exp.add.assert_called_once()

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    @patch("nemo_skills.pipeline.utils.declarative.run_exp")
    def test_pipeline_run_with_dependencies(self, mock_run_exp, mock_env_vars, mock_get_exp):
        """Test pipeline execution with job dependencies."""
        # Setup mocks
        mock_config = {
            "executor": "none",
            "containers": {"nemo-skills": "container:latest"},
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}

        mock_exp = MagicMock()
        mock_exp.add.side_effect = ["handle_1", "handle_2"]
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        # Create pipeline with dependencies
        cmd1 = Command(command="echo 1", name="cmd1")
        group1 = CommandGroup(commands=[cmd1], name="group1", log_dir="/logs")

        cmd2 = Command(command="echo 2", name="cmd2")
        group2 = CommandGroup(commands=[cmd2], name="group2", log_dir="/logs")

        jobs = [
            {"name": "job1", "group": group1},
            {"name": "job2", "group": group2, "dependencies": ["job1"]},
        ]
        pipeline = Pipeline(name="test", cluster_config=mock_config, jobs=jobs, skip_hf_home_check=True)

        # Run pipeline
        pipeline.run(dry_run=True)

        # Verify both jobs were added
        assert mock_exp.add.call_count == 2

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    @patch("nemo_skills.pipeline.utils.declarative.is_mounted_filepath")
    @patch("nemo_skills.pipeline.utils.declarative.get_executor")
    def test_pipeline_hf_home_validation(self, mock_get_executor, mock_is_mounted, mock_env_vars, mock_get_exp):
        """Test HF_HOME validation."""
        mock_config = {
            "executor": "slurm",
            "containers": {"nemo-skills": "container:latest"},
            "account": "test_account",
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}
        mock_is_mounted.return_value = True
        mock_get_executor.return_value = MagicMock()

        mock_exp = MagicMock()
        mock_exp.add.return_value = "handle"
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group", log_dir="/logs")
        pipeline = Pipeline(name="test", cluster_config=mock_config, groups=[group])

        # Should not raise
        pipeline.run(dry_run=True)

        # Verify executor was created
        assert mock_get_executor.called

    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    def test_pipeline_hf_home_missing(self, mock_env_vars):
        """Test that missing HF_HOME raises error."""
        mock_config = {"executor": "slurm", "containers": {}}
        mock_env_vars.return_value = {}  # No HF_HOME

        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group", log_dir="/logs")
        pipeline = Pipeline(name="test", cluster_config=mock_config, groups=[group])

        with pytest.raises(RuntimeError, match="HF_HOME is missing"):
            pipeline.run(dry_run=True)

    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    @patch("nemo_skills.pipeline.utils.declarative.is_mounted_filepath")
    def test_pipeline_hf_home_not_mounted(self, mock_is_mounted, mock_env_vars):
        """Test that non-mounted HF_HOME raises error."""
        mock_config = {"executor": "slurm", "containers": {}}
        mock_env_vars.return_value = {"HF_HOME": "/hf"}
        mock_is_mounted.return_value = False

        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group", log_dir="/logs")
        pipeline = Pipeline(name="test", cluster_config=mock_config, groups=[group])

        with pytest.raises(RuntimeError, match="is not a mounted path"):
            pipeline.run(dry_run=True)


class TestHetGroupIndices:
    """Test het_group_index assignment."""

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    def test_het_group_index_non_heterogeneous(self, mock_env_vars, mock_get_exp):
        """Test that non-heterogeneous jobs have het_group_index=None."""
        mock_config = {
            "executor": "none",
            "containers": {"nemo-skills": "container:latest"},
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}

        mock_exp = MagicMock()
        mock_exp.add.return_value = "handle"
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        # Create single-group job with multiple components
        cmd1 = Command(command="echo 1", name="cmd1")
        cmd2 = Command(command="echo 2", name="cmd2")
        group = CommandGroup(commands=[cmd1, cmd2], name="group", log_dir="/logs")

        pipeline = Pipeline(name="test", cluster_config=mock_config, groups=[group], skip_hf_home_check=True)
        pipeline.run(dry_run=True)

        # Both commands should have None het_group_index (localhost communication)
        assert cmd1.het_group_index is None
        assert cmd2.het_group_index is None
        assert cmd1.hostname_ref() == "127.0.0.1"
        assert cmd2.hostname_ref() == "127.0.0.1"

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    def test_het_group_index_heterogeneous(self, mock_env_vars, mock_get_exp):
        """Test that heterogeneous jobs get per-job het_group_index."""
        mock_config = {
            "executor": "none",
            "containers": {"nemo-skills": "container:latest"},
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}

        mock_exp = MagicMock()
        mock_exp.add.return_value = "handle"
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        # Create multi-group heterogeneous job
        cmd1 = Command(command="echo 1", name="cmd1")
        group1 = CommandGroup(commands=[cmd1], name="group1", log_dir="/logs")

        cmd2 = Command(command="echo 2", name="cmd2")
        group2 = CommandGroup(commands=[cmd2], name="group2", log_dir="/logs")

        jobs = [{"name": "hetjob", "groups": [group1, group2]}]
        pipeline = Pipeline(name="test", cluster_config=mock_config, jobs=jobs, skip_hf_home_check=True)
        pipeline.run(dry_run=True)

        # Commands should have het_group_index 0 and 1
        assert cmd1.het_group_index == 0
        assert cmd2.het_group_index == 1
        assert "$SLURM_JOB_NODELIST_HET_GROUP_0" in cmd1.hostname_ref()
        assert "$SLURM_JOB_NODELIST_HET_GROUP_1" in cmd2.hostname_ref()

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    def test_het_group_index_per_job_not_global(self, mock_env_vars, mock_get_exp):
        """Test that het_group_index is per-job, not global across pipeline."""
        mock_config = {
            "executor": "none",
            "containers": {"nemo-skills": "container:latest"},
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}

        mock_exp = MagicMock()
        mock_exp.add.side_effect = ["handle1", "handle2"]
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        # Create two separate heterogeneous jobs
        cmd1 = Command(command="echo 1", name="cmd1")
        group1 = CommandGroup(commands=[cmd1], name="group1", log_dir="/logs")

        cmd2 = Command(command="echo 2", name="cmd2")
        group2 = CommandGroup(commands=[cmd2], name="group2", log_dir="/logs")

        cmd3 = Command(command="echo 3", name="cmd3")
        group3 = CommandGroup(commands=[cmd3], name="group3", log_dir="/logs")

        cmd4 = Command(command="echo 4", name="cmd4")
        group4 = CommandGroup(commands=[cmd4], name="group4", log_dir="/logs")

        jobs = [
            {"name": "hetjob1", "groups": [group1, group2]},
            {"name": "hetjob2", "groups": [group3, group4]},
        ]
        pipeline = Pipeline(name="test", cluster_config=mock_config, jobs=jobs, skip_hf_home_check=True)
        pipeline.run(dry_run=True)

        # Both jobs should have het_group_index starting from 0
        assert cmd1.het_group_index == 0
        assert cmd2.het_group_index == 1
        assert cmd3.het_group_index == 0  # Starts from 0 again!
        assert cmd4.het_group_index == 1


class TestDependencyResolution:
    """Test dependency resolution in Pipeline."""

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    def test_dependency_none_handling(self, mock_env_vars, mock_get_exp):
        """Test that explicit None dependencies are handled correctly."""
        mock_config = {
            "executor": "none",
            "containers": {"nemo-skills": "container:latest"},
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}

        mock_exp = MagicMock()
        mock_exp.add.return_value = "handle"
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group", log_dir="/logs")

        jobs = [{"name": "job", "group": group, "dependencies": None}]
        pipeline = Pipeline(name="test", cluster_config=mock_config, jobs=jobs, skip_hf_home_check=True)

        # Should not raise
        pipeline.run(dry_run=True)

    @patch("nemo_skills.pipeline.utils.declarative.get_exp")
    @patch("nemo_skills.pipeline.utils.declarative.get_env_variables")
    def test_pipeline_run_after_applies_to_jobs(self, mock_env_vars, mock_get_exp):
        """Test that pipeline-level run_after applies to jobs without dependencies."""
        mock_config = {
            "executor": "none",
            "containers": {"nemo-skills": "container:latest"},
        }
        mock_env_vars.return_value = {"HF_HOME": "/hf"}

        mock_exp = MagicMock()
        mock_exp.add.return_value = "handle"
        mock_get_exp.return_value.__enter__.return_value = mock_exp

        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group", log_dir="/logs")

        pipeline = Pipeline(
            name="test", cluster_config=mock_config, groups=[group], run_after="other_exp", skip_hf_home_check=True
        )

        # Should not raise and should apply run_after
        pipeline.run(dry_run=True)


class TestErrorHandling:
    """Test error handling in Pipeline."""

    def test_pipeline_job_missing_group_or_groups(self):
        """Test that job spec without group or groups raises error."""
        mock_config = {"executor": "none", "containers": {}}
        jobs = [{"name": "bad_job"}]  # Missing 'group' or 'groups'

        with pytest.raises(ValueError, match="must have either 'group' or 'groups'"):
            pipeline = Pipeline(name="test", cluster_config=mock_config, jobs=jobs)
            pipeline.run(dry_run=True)

    def test_commandgroup_missing_log_dir(self):
        """Test that CommandGroup without log_dir raises error during execution."""
        mock_config = {"executor": "none", "containers": {}}
        cmd = Command(command="echo test", name="cmd")
        group = CommandGroup(commands=[cmd], name="group")  # No log_dir

        pipeline = Pipeline(name="test", cluster_config=mock_config, groups=[group])

        with pytest.raises(ValueError, match="must have log_dir set"):
            pipeline.run(dry_run=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
