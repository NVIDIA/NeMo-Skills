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

"""Wrapper that evaluates Lean4 proofs using sandbox execution."""

from typing import Any, Dict

from nemo_skills.code_execution.sandbox import extract_proof_only
from nemo_skills.code_execution.utils import clean_formal_generation
from nemo_skills.dataset.utils import get_lean4_header
from nemo_skills.inference.model.base import BaseModel
from nemo_skills.inference.model.wrapper import ContextAwareModel, ContextAwareWrapper


class LeanEvalWrapper(ContextAwareWrapper):
    """Wrapper that evaluates Lean4 proofs using the GenerationTask's sandbox."""

    def default_config(self) -> Dict[str, Any]:
        return {
            "timeout": 30.0,
            "answer_format": "lean4-proof",  # "lean4-proof" or "lean4-statement"
            "use_predicted_proof_key": False,
            "final_answer_key": "**FINAL ANSWER**",
            "restate_formal_statement": True,
            "strip_theorem_from_proof": True,
            "extract_code_mode": "last",  # "first" or "last"
        }

    def configure(self, overrides=None, context=None):
        super().configure(overrides, context)

        # Get sandbox from context (passed from GenerationTask)
        self.sandbox = self.context.get("sandbox") if self.context else None

        if self.sandbox is None:
            raise ValueError("LeanEvalWrapper requires sandbox in context")

    def wrap(self, model: BaseModel) -> BaseModel:
        return LeanEvalModel(model, self.config, self.sandbox)


class LeanEvalModel(ContextAwareModel):
    """Model that evaluates Lean4 proofs from generation."""

    def __init__(self, model: BaseModel, config: dict, sandbox):
        super().__init__(model, config)
        self.sandbox = sandbox

    async def post_process(self, result, data_point, all_data):
        """Extract and evaluate Lean4 proof from generation."""
        try:
            # Prepare predicted_proof based on format (replicating batch_evaluate_results logic)
            predicted_proof = await self._prepare_predicted_proof(result, data_point)

            # Execute proof and get raw compiler output
            compiler_output = await self._execute_lean_proof(predicted_proof)

            # Add evaluation results to output
            result["predicted_proof"] = predicted_proof
            result["proof_status"] = self._determine_proof_status(compiler_output)
            result["lean_evaluation"] = compiler_output

        except Exception as e:
            result["proof_status"] = "error"
            result["lean_evaluation"] = {
                "process_status": "error",
                "stdout": "",
                "stderr": f"Error during evaluation: {str(e)}",
                "timeout": self.config["timeout"],
            }

        return result

    async def _prepare_predicted_proof(self, result, data_point):
        """Prepare the predicted proof based on answer format."""
        generation = result["generation"]
        answer_format = self.config["answer_format"]

        if answer_format == "lean4-proof":
            if not self.config["use_predicted_proof_key"]:
                # Clean the generation and extract the formal proof
                cleaned_generation = clean_formal_generation(
                    generation,
                    final_answer_key=self.config["final_answer_key"],
                    extract_code_mode=self.config["extract_code_mode"],
                )

                # Combine header + formal_statement + proof
                header = data_point.get("header", "") if data_point else ""
                formal_statement = (
                    data_point.get("formal_statement", "")
                    if data_point and self.config["restate_formal_statement"]
                    else ""
                )

                if self.config["strip_theorem_from_proof"]:
                    proof_part = extract_proof_only(cleaned_generation)
                    predicted_proof = header + formal_statement + proof_part
                else:
                    predicted_proof = cleaned_generation

            else:
                # Use existing predicted_proof key
                if not data_point or "predicted_proof" not in data_point:
                    raise ValueError("predicted_proof key not found in data_point or data_point is None")
                predicted_proof = data_point["predicted_proof"]

        elif answer_format == "lean4-statement":
            if not self.config["use_predicted_proof_key"]:
                # Clean generation and add header + sorry
                cleaned_generation = clean_formal_generation(
                    generation, extract_code_mode=self.config["extract_code_mode"]
                )
                header = get_lean4_header()
                predicted_proof = header + cleaned_generation + "\n sorry"
            else:
                if not data_point or "predicted_proof" not in data_point:
                    raise ValueError("predicted_proof key not found in data_point or data_point is None")
                predicted_proof = data_point["predicted_proof"]
        else:
            raise ValueError(f"Unknown answer_format: {answer_format}")

        return predicted_proof

    async def _execute_lean_proof(self, pred_output):
        """Execute Lean proof and return raw compiler output."""
        try:
            output, _ = await self.sandbox.execute_code(
                generated_code=pred_output,
                language="lean4",
                timeout=self.config["timeout"],
            )

            # Return the raw compiler output with timeout info
            return {**output, "timeout": self.config["timeout"]}

        except Exception as e:
            return {
                "process_status": "error",
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "timeout": self.config["timeout"],
            }

    def _determine_proof_status(self, compiler_output):
        """Determine proof status from compiler output (replicates sandbox.is_proof_correct logic)."""
        import re

        process_status = compiler_output.get("process_status", "unknown")

        if process_status == "timeout":
            return "timeout"
        elif process_status != "completed":
            return process_status

        # Check stdout and stderr for proof status indicators
        stdout = compiler_output.get("stdout", "").lower()
        stderr = compiler_output.get("stderr", "").lower()
        combined = stdout + "\n" + stderr

        # Check for sorry (incomplete proof)
        if re.search(r"\bsorry\b", combined) is not None:
            return "has_sorry"

        # If process completed without errors, consider it successful
        return "completed"
