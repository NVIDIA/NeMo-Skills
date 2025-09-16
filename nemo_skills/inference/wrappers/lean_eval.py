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

            # Evaluate proof using sandbox
            proof_status = await self._is_proof_correct(predicted_proof)

            # Add evaluation results to output
            result["predicted_proof"] = predicted_proof
            result["proof_status"] = proof_status
            result["lean_evaluation"] = {
                "success": proof_status == "correct",
                "status": proof_status,
                "timeout": self.config["timeout"],
            }

        except Exception as e:
            result["lean_evaluation"] = {"success": False, "status": "error", "error": str(e)}

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

    async def _is_proof_correct(self, pred_output):
        """Check if the proof is correct using sandbox's is_proof_correct method."""
        return await self.sandbox.is_proof_correct(pred_output, timeout=self.config["timeout"])
