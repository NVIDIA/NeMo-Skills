from enum import Enum


class GenerationType(str, Enum):
    generate = "generate"
    math_judge = "math_judge"
    check_contamination = "check_contamination"


GENERATION_MODULE_MAP = {
    GenerationType.generate: "nemo_skills.inference.generate",
    GenerationType.math_judge: "nemo_skills.inference.llm_math_judge",
    GenerationType.check_contamination: "nemo_skills.inference.check_contamination",
}
