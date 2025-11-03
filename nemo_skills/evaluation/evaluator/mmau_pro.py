# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
import logging
import re
from typing import Any

from tqdm import tqdm

from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MMAUProEvaluatorConfig:
    """Configuration for MMAU-Pro evaluation."""

    # Prompt configuration
    prompt_config: str = "eval/speechlm/mmau-pro"

    # NVEmbed settings
    embedding_model: str = "nvidia/NV-Embed-v2"
    use_nvembed: bool = True


def eval_mmau_pro(cfg):
    """Evaluate MMAU-Pro dataset using nemo-skills framework.

    This evaluator processes JSONL files with speech/audio language model outputs
    and evaluates them using two main approaches:
    - Closed-form questions: NVEmbed similarity matching evaluation
    - Open-ended questions: LLM as a Judge evaluation
    - Instruction following questions: Audio Instruction Following (AIF) format evaluation

    All categories maintain separate logging to track sample counts and success rates.
    """
    # Extract only the fields that belong to MMAUProEvaluatorConfig
    config_fields = {"prompt_config", "embedding_model", "use_nvembed"}
    config_kwargs = {k: v for k, v in cfg.items() if k in config_fields}
    eval_config = MMAUProEvaluatorConfig(**config_kwargs)

    jsonl_file = cfg["input_file"]
    LOG.info(f"Evaluating {jsonl_file}")

    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    samples_to_evaluate = sum(
        1 for sample in data if "nvembed_confidence" not in sample or sample.get("category") == "instruction following"
    )
    samples_already_done = len(data) - samples_to_evaluate

    if samples_already_done > 0:
        LOG.info(f"Resuming evaluation: {samples_already_done}/{len(data)} samples already have nvembed_confidence")

    for idx, sample in enumerate(tqdm(data, desc="Evaluating samples")):
        evaluated_sample = evaluate_sample(sample, eval_config)
        data[idx] = evaluated_sample

    # Write all results at once
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in data:
            fout.write(json.dumps(sample) + "\n")

    LOG.info(f"Evaluation completed for {jsonl_file}")


def evaluate_sample(sample: dict[str, Any], config: MMAUProEvaluatorConfig) -> dict[str, Any]:
    sample = sample.copy()
    category = sample.get("category", "unknown")

    # Add subset_for_metrics only for closed_form subcategories (not for open_ended or instruction_following)
    # This creates per-category breakdowns only for closed_form (e.g., music, speech, sound)
    if category not in ["open", "instruction following"]:
        sample["subset_for_metrics"] = category

    if category == "instruction following":
        sample = evaluate_instruction_following(sample, config)
    else:
        choices = sample.get("choices", [])

        if config.use_nvembed and choices and len(choices) > 1:
            sample = evaluate_closed_form_with_nvembed(sample, config)
        else:
            if "requires_judge" not in sample:
                sample["requires_judge"] = True
                sample["predicted_answer"] = sample.get("generation", "")
            sample["is_correct"] = False

        sample["expected_answer"] = sample.get("expected_answer", "")
        sample["choices"] = sample.get("choices", [])

    sample["is_correct"] = get_overall_correctness(sample, category)
    return sample


def get_overall_correctness(sample: dict[str, Any], category: str) -> bool:
    if "judgement" not in sample:
        return sample.get("is_correct", False)

    return extract_judge_result(sample.get("judgement", ""))


def extract_judge_result(judgement_text: str) -> bool:
    """Extract judge result from judgement text (nemo-skills pattern)."""
    import re

    if re.search(r"\byes\b", judgement_text, re.IGNORECASE):
        return True
    elif re.search(r"\bno\b", judgement_text, re.IGNORECASE):
        return False
    else:
        return False


# ===========================================
# Open-ended questions are handled by separate judge pipeline
# ===========================================

# ===========================================
# Closed-ended questions evaluation using NVEmbed similarity matching
# ===========================================


def evaluate_with_nvembed_similarity(
    model_prediction: str, choices: list, ground_truth: str, config: MMAUProEvaluatorConfig
) -> tuple[str, float]:
    """NVEmbed-based evaluation: match predictions to choices using embedding similarity."""
    import torch
    import torch.nn.functional as F

    model = load_nvembed_model(config.embedding_model)
    device = next(model.parameters()).device

    with torch.no_grad():
        prediction_embedding = model.encode([model_prediction], instruction="", max_length=4096, device=device)
        choice_embeddings = model.encode(choices, instruction="", max_length=4096, device=device)

    prediction_embedding = F.normalize(prediction_embedding, p=2, dim=1)
    choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)

    scores = (prediction_embedding @ choice_embeddings.T) * 100
    scores = scores.squeeze()

    if scores.dim() == 0:
        scores = scores.unsqueeze(0)

    best_choice_idx = torch.argmax(scores).item()
    matched_choice = choices[best_choice_idx]
    confidence = torch.max(scores).item()

    return matched_choice, confidence


def load_nvembed_model(model_name: str = "nvidia/NV-Embed-v2"):
    """Load NVEmbed model using HuggingFace AutoModel with GPU support."""

    import os

    import torch
    from transformers import AutoModel

    if not hasattr(load_nvembed_model, "_cache"):
        load_nvembed_model._cache = {}

    if model_name in load_nvembed_model._cache:
        return load_nvembed_model._cache[model_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use explicit cache directory (respects HF_HOME env var)
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        LOG.info(f"Using HuggingFace cache directory: {cache_dir}")

    try:
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=False
        )
        model.to(device)
        model.eval()
        load_nvembed_model._cache[model_name] = model
        LOG.info(f"Successfully loaded {model_name} on {device}")
        return model
    except Exception as e:
        LOG.error(f"Failed to load {model_name}: {e}")
        raise


def evaluate_closed_form_with_nvembed(sample: dict[str, Any], config: MMAUProEvaluatorConfig) -> dict[str, Any]:
    """Evaluate closed-form questions using NVEmbed similarity matching."""

    if "nvembed_confidence" in sample:
        LOG.info("Skipping sample - nvembed_confidence already exists")
        return sample

    generation = sample.get("generation", "").strip()
    choices = sample.get("choices", [])
    expected_answer = sample.get("expected_answer", "")

    LOG.info(f"NVEmbed evaluation for generation='{generation[:50]}'")

    if not generation or not choices:
        LOG.warning("Missing generation or choices for nvembed evaluation")
        sample.update({"nvembed_success": False, "is_correct": False, "error": "missing_generation_or_choices"})
        return sample

    try:
        matched_choice, confidence = evaluate_with_nvembed_similarity(generation, choices, expected_answer, config)

        is_correct = matched_choice.strip().lower() == expected_answer.strip().lower()

        sample.update(
            {"nvembed_matched_choice": matched_choice, "nvembed_confidence": confidence, "is_correct": is_correct}
        )

        return sample

    except Exception as e:
        LOG.error(f"NVEmbed evaluation failed: {e}")
        sample.update(
            {
                "is_correct": False,
            }
        )
        return sample


# ======================================================
# Audio Instruction Following (AIF) Evaluation Function
# ======================================================


def evaluate_instruction_following(sample: dict[str, Any], config: MMAUProEvaluatorConfig) -> dict[str, Any]:
    """Evaluate instruction following questions using AIF (Audio Instruction Following) criteria."""
    generation = sample.get("generation", "").strip()

    LOG.info(f"AIF evaluation for generation='{generation}'")

    if not generation:
        LOG.info("Empty generation detected for instruction following question")
        sample.update({"is_correct": False, "error": "empty_generation"})
        return sample

    success = evaluate_aif_constraints(
        generation, sample.get("task_identifier", ""), sample.get("kwargs", {}) or {}, sample
    )

    sample.update({"is_correct": success})

    return sample


def evaluate_aif_constraints(
    response: str, task_identifier: str, kwargs: dict[str, Any], sample_data: dict[str, Any]
) -> bool:
    def count_words(text):
        return len(text.split())

    def count_sentences(text):
        return len([s for s in re.split(r"[.!?]+", text.strip()) if s.strip()])

    def count_paragraphs(text):
        return len([p for p in text.split("***") if p.strip()])

    def count_bullet_points(text):
        return len(re.findall(r"(?:^|\n)\s*\*\s+", text))

    def count_highlighted_sections(text):
        return len(re.findall(r"\*([^*]+)\*", text))

    def count_placeholders(text):
        return len(re.findall(r"\[[^\]]+\]", text))

    def count_capital_words(text):
        return len([word for word in text.split() if word.isupper()])

    def count_keyword_frequency(text, keyword):
        return len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", text.lower()))

    def has_title(text):
        return bool(re.search(r"<<[^>]+>>", text))

    def has_postscript(text, marker):
        return re.sub(r"[^a-zA-Z]", "", marker).lower() in re.sub(r"[^a-zA-Z]", "", text).lower()

    def starts_with_phrase(text, phrase):
        return re.sub(r"[^a-zA-Z ]", "", text).lower().startswith(re.sub(r"[^a-zA-Z ]", "", phrase).lower())

    def ends_with_phrase(text, phrase):
        return re.sub(r"[^a-zA-Z ]", "", text).lower().endswith(re.sub(r"[^a-zA-Z ]", "", phrase).lower())

    def is_wrapped_in_quotes(text):
        return text.strip().startswith('"') and text.strip().endswith('"')

    def has_no_commas(text):
        return "," not in text

    def check_sections(text, num_sections, splitter):
        sections = [s for s in re.split(rf"\s*{re.escape(splitter)}\s*", text.strip()) if s.strip()]
        return len(sections) == num_sections

    checks = {
        "Include Keywords": lambda: all(k.lower() in response.lower() for k in kwargs.get("keywords", "").split(", ")),
        "Keyword Frequency": lambda: count_keyword_frequency(response, kwargs.get("keyword", ""))
        == kwargs.get("N", 0),
        "Forbidden Words": lambda: not any(
            w.lower() in response.lower() for w in kwargs.get("forbidden_words", "").split(", ")
        ),
        "Number Paragraphs": lambda: count_paragraphs(response) == kwargs.get("N", 0),
        "Number Words (at least)": lambda: count_words(response) >= kwargs.get("N", 0),
        "Number Words (at most)": lambda: count_words(response) <= kwargs.get("N", 0),
        "Number Words (range)": lambda: kwargs.get("N1", 0) <= count_words(response) <= kwargs.get("N2", 999),
        "Number Sentences (at least)": lambda: count_sentences(response) >= kwargs.get("N", 0),
        "Number Sentences (at most)": lambda: count_sentences(response) <= kwargs.get("N", 0),
        "Number Sentences (range)": lambda: kwargs.get("N1", 0) <= count_sentences(response) <= kwargs.get("N2", 999),
        "Postscript": lambda: has_postscript(response, kwargs.get("postscript_marker", "")),
        "Number Placeholder": lambda: count_placeholders(response) >= kwargs.get("N", 0),
        "Number Bullets": lambda: count_bullet_points(response) == kwargs.get("N", 0),
        "Title": lambda: has_title(response),
        "Minimum Number Highlighted Section": lambda: count_highlighted_sections(response) >= kwargs.get("N", 0),
        "Multiple Sections": lambda: check_sections(response, kwargs.get("N", 0), kwargs.get("section_splitter", "")),
        "Repeat Prompt": lambda: response.strip()
        .lower()
        .startswith(sample_data.get("prompt_transcription", "").strip().lower()),
        "Two Responses": lambda: len(response.split("******")) == 2
        and response.split("******")[0].lower().strip() != response.split("******")[1].lower().strip(),
        "All Uppercase": lambda: response.isupper(),
        "All Lowercase": lambda: response.islower(),
        "All-capital Words (at least)": lambda: count_capital_words(response) >= kwargs.get("N", 0),
        "All-capital Words (at most)": lambda: count_capital_words(response) <= kwargs.get("N", 0),
        "All-capital Words (range)": lambda: kwargs.get("N1", 0)
        <= count_capital_words(response)
        <= kwargs.get("N2", 999),
        "Start Checker": lambda: starts_with_phrase(response, kwargs.get("start_phrase", "")),
        "End Checker": lambda: ends_with_phrase(response, kwargs.get("end_phrase", "")),
        "Quotation": lambda: is_wrapped_in_quotes(response),
        "No Commas": lambda: has_no_commas(response),
    }

    return checks.get(task_identifier, lambda: False)()
