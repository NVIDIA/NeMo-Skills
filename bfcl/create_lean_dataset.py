#!/usr/bin/env python3
"""
Create Lean 4 Theorem Proving Dataset for BFCL

This script creates a BFCL-compatible dataset for evaluating LLM theorem proving
capabilities using Lean 4.
"""

import json
import os
from typing import Dict, Any, List

from nemo_skills.dataset.bfcl_v3.utils import convert_to_tool, func_doc_language_specific_pre_processing

# Function schemas for Lean 4 operations
LEAN_FUNCTIONS = [
    {
        "name": "execute_lean_code",
        "description": "Execute Lean 4 code such as theorems, definitions, or commands. This is the main function for proving theorems and running Lean code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Lean 4 code to execute. This can be a theorem to prove, a definition, or any other Lean command."
                },
                "mode": {
                    "type": "string",
                    "enum": ["proof", "command"],
                    "description": "Execution mode: 'proof' for theorems and proofs, 'command' for other Lean commands like #check or #eval",
                    "default": "proof"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "start_interactive_theorem",
        "description": "Load a theorem for interactive development. This allows you to work on a theorem step by step, editing specific parts.",
        "parameters": {
            "type": "object",
            "properties": {
                "theorem_code": {
                    "type": "string",
                    "description": "The complete theorem code to load for interactive editing. Should include theorem declaration and initial proof structure."
                }
            },
            "required": ["theorem_code"]
        }
    },
    {
        "name": "edit_proof_clause",
        "description": "Edit a specific clause or part of an interactive theorem proof. Use this to refine specific parts of a proof.",
        "parameters": {
            "type": "object",
            "properties": {
                "clause_id": {
                    "type": "string",
                    "description": "ID of the clause to edit (e.g., 'sorry_0', 'have_h1', 'main_proof_0')"
                },
                "new_content": {
                    "type": "string",
                    "description": "New content to replace the clause with"
                }
            },
            "required": ["clause_id", "new_content"]
        }
    },
    {
        "name": "add_proof_structure",
        "description": "Add new proof structure lines to an interactive theorem. Use this to add 'have' statements or other proof structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "structure_lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of proof structure lines to add (e.g., ['have h1 : P := by sorry', 'exact h1'])"
                }
            },
            "required": ["structure_lines"]
        }
    },
    {
        "name": "validate_lean",
        "description": "Run Lean validation commands like #check, #eval, or #print to verify types, evaluate expressions, or inspect definitions.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Validation command to run (e.g., '#check Nat.add_comm', '#eval 2 + 2', '#print List')"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "get_proof_state",
        "description": "Get the current state of an interactive proof session including code, messages, goals, and available clauses to edit.",
        "parameters": {
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "enum": ["full_panel", "current_code", "messages", "editable_clauses", "suggestions"],
                    "description": "Type of information to retrieve about the current proof state",
                    "default": "full_panel"
                }
            },
            "required": []
        }
    }
]


def create_lean_theorem_problem(
    problem_id: str,
    theorem_statement: str,
    user_query: str,
    expected_approach: str = "",
) -> Dict[str, Any]:
    """
    Create a BFCL problem for Lean theorem proving.

    Args:
        problem_id: Unique identifier for the problem
        theorem_statement: The theorem to prove in Lean syntax
        user_query: Natural language description of what to prove
        expected_approach: Optional hint about expected approach
        single_turn: Whether this is a single-turn or multi-turn problem

    Returns:
        BFCL-compatible problem dictionary
    """
    # Process functions for BFCL
    functions = func_doc_language_specific_pre_processing(LEAN_FUNCTIONS, "python")
    tools = convert_to_tool(functions)

    # Create the problem structure
    problem = {
        "id": problem_id,
        "question": [
            # Turn 1: Start the theorem
            [
                {
                    "role": "user",
                    "content": (
                        f"{user_query}\n\nTheorem to prove:\n```lean\n{theorem_statement}\n```\n\nStart by setting up the interactive theorem. "
                    )
                }
            ],
            # # Turn 2: Work on the proof
            [
                {
                    "role": "user",
                    "content":
                        (
                            "Now work on proving this theorem step by step. Add the necessary proof structure and fill in the details. "
                            "At the end, return the complete proof following a **Final Answer** tag."
                        ),
                }
            ],
        ],
        "function": functions,
        "tools": tools,
        "single_turn": False,
        "involved_classes": ["LeanAPI"],
        "initial_config": {"LeanAPI": {}}
    }

    return problem


def create_conjunction_associativity_problem() -> Dict[str, Any]:
    """Create a problem for proving conjunction associativity."""

    theorem_code = """theorem conjunction_assoc (A B C : Prop) : (A ∧ B) ∧ C → A ∧ (B ∧ C) := by
  sorry"""

    user_query = """Prove that conjunction is associative. That is, show that (A ∧ B) ∧ C implies A ∧ (B ∧ C).

This is a fundamental theorem in propositional logic. You'll need to:
1. Assume the hypothesis (A ∧ B) ∧ C
2. Extract the individual components A, B, and C
3. Reconstruct them in the form A ∧ (B ∧ C)

Use the Lean 4 theorem prover to complete this proof step by step."""

    expected_approach = """Expected approach:
- Use 'intro h' to introduce the hypothesis
- Use 'h.left' and 'h.right' to destructure conjunctions
- Use angle bracket notation ⟨·, ·⟩ to construct conjunctions
- Or use 'have' statements to break down the proof into steps"""

    return create_lean_theorem_problem(
        problem_id="lean_conjunction_assoc_001",
        theorem_statement=theorem_code,
        user_query=user_query,
        expected_approach=expected_approach,
    )


def create_implication_transitivity_problem() -> Dict[str, Any]:
    """Create a problem for proving implication transitivity."""

    theorem_code = """theorem impl_trans (P Q R : Prop) : (P → Q) → (Q → R) → (P → R) := by
  sorry"""

    user_query = """Prove that implication is transitive. That is, if P implies Q and Q implies R, then P implies R.

This is another fundamental theorem in logic. You'll need to work with nested implications."""

    expected_approach = """Expected approach:
- Use multiple 'intro' statements for the implications
- Apply function application to chain the implications
- Or use intermediate 'have' statements"""

    return create_lean_theorem_problem(
        problem_id="lean_impl_trans_001",
        theorem_statement=theorem_code,
        user_query=user_query,
        expected_approach=expected_approach,
    )


def create_interactive_theorem_problem() -> Dict[str, Any]:
    """Create a multi-turn interactive theorem proving problem."""

    theorem_code = """theorem demorgan_and (P Q : Prop) : ¬(P ∧ Q) → (¬P ∨ ¬Q) := by
  sorry"""

    # Multi-turn problem for interactive development
    problem = create_lean_theorem_problem(
        problem_id="lean_demorgan_interactive_001",
        theorem_statement=theorem_code,
        user_query="Prove one of De Morgan's laws: ¬(P ∧ Q) → (¬P ∨ ¬Q). This requires classical logic.",
        expected_approach="This is more complex and requires classical reasoning (law of excluded middle).",
    )

    return problem


def create_lean_theorem_dataset() -> str:
    """Create a complete BFCL dataset for Lean theorem proving."""

    problems = [
        create_conjunction_associativity_problem(),
        create_implication_transitivity_problem(),
        create_interactive_theorem_problem()
    ]

    dataset_file = "lean_theorems_dataset.jsonl"

    with open(dataset_file, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem) + "\n")

    print(f"✅ Created dataset with {len(problems)} problems: {dataset_file}")
    print("\nProblems included:")
    for i, problem in enumerate(problems, 1):
        print(f"  {i}. {problem['id']}")

    return dataset_file


def main():
    """Main function to create the dataset."""
    print("🧮 CREATING LEAN 4 THEOREM PROVING DATASET")
    print("=" * 60)
    print()
    print("This creates a BFCL-compatible dataset for evaluating LLM")
    print("theorem proving capabilities using Lean 4.")
    print()

    # Create the dataset
    dataset_file = create_lean_theorem_dataset()

    print()
    print("🎯 THEOREMS TO PROVE:")
    print("  • Conjunction associativity: (A ∧ B) ∧ C → A ∧ (B ∧ C)")
    print("  • Implication transitivity: (P → Q) → (Q → R) → (P → R)")
    print("  • De Morgan's law: ¬(P ∧ Q) → (¬P ∨ ¬Q)")
    print()
    print("🔬 EVALUATION ASPECTS:")
    print("  • Mathematical reasoning")
    print("  • Tool selection and usage")
    print("  • Interactive proof development")
    print("  • Multi-turn problem solving")
    print()
    print("📁 NEXT STEPS:")
    print(f"  1. Use {dataset_file} with BFCL evaluation")
    print("  2. Set BFCL_TOOLS_CONFIG=lean_tools_config.json")
    print("  3. Run BFCL with OpenAI or other models")
    print()
    print("✅ Dataset creation complete!")


if __name__ == "__main__":
    main()
