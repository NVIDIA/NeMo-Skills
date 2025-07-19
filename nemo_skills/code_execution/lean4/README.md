# Lean 4 Theorem Proving Utilities

This submodule provides comprehensive Lean 4 theorem proving capabilities for NeMo-Skills.

## Components

### 1. Core Prover (`prover.py`)

**Class**: `LeanProver`

A simple but minimally sufficient Lean 4 prover interface that focuses on:
- Simple interface for executing proof steps
- Clear distinction between proofs with sorry vs complete proofs
- User-managed proof state with backtracking support
- Mathlib and aesop imports via TempRequireProject
- ProofStep execution for granular tactic application
- Command execution for standalone operations
- Incremental proof building with tactic manipulation

**Key Classes**:
- `ProofResult`: Result of a proof execution
- `ProofInProgress`: Represents a proof being built incrementally

### 2. Interactive Agent (`interactive_agent.py`)

**Class**: `InteractiveLeanAgent`

VS Code-like interactive development experience that mimics how mathematicians like Terence Tao work with Lean 4:
- Real-time compiler feedback and messages
- Position-aware editing with goal state tracking
- Targeted updates with immediate validation
- Interactive development workflow for LLM agents

**Key Classes**:
- `Position`: Position in the Lean file (line, column)
- `LeanMessage`: A compiler message (error, warning, info) at a specific position
- `ProofGoal`: A proof goal at a specific position
- `EditableClause`: A clause/section of code that can be edited

## Usage

### Basic Prover Usage

```python
from nemo_skills.code_execution.lean4 import LeanProver

# Initialize with mathlib support
prover = LeanProver(mathlib_enabled=True)

# Execute a simple theorem
result = prover.run("theorem test : 1 + 1 = 2 := by simp")
print(f"Success: {result.success}")
print(f"Complete: {result.proof_complete}")

# Start incremental proof building
proof_result = prover.start_proof("my_theorem", "(a b : Nat) : a + b = b + a")
if proof_result.success:
    # Apply tactics step by step
    step_result = prover.apply_tactic_to_proof("my_theorem", "rw [add_comm]")
```

### Interactive Agent Usage

```python
from nemo_skills.code_execution.lean4 import InteractiveLeanAgent

# Create interactive agent
agent = InteractiveLeanAgent(mathlib_enabled=True)

# Load a theorem with structure
theorem_code = """theorem demo (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h2 h, h1 h⟩"""

result = agent.load_theorem(theorem_code)
print(f"Editable clauses: {list(result['editable_clauses'])}")

# Edit specific clauses with immediate feedback
edit_result = agent.edit_clause("have_h1", "intro h; exact h.left")
print(f"Edit successful: {edit_result['compilation_result']['success']}")

edit_result = agent.edit_clause("have_h2", "intro h; exact h.right")
print(f"Edit successful: {edit_result['compilation_result']['success']}")

# Get interactive panel state (like VS Code)
panel = agent.get_interactive_panel()
print("Current messages:", panel['messages'])
print("Current goals:", panel['goals'])
```

### Demo Functions

Run the interactive demo to see the agent in action:

```python
from nemo_skills.code_execution.lean4 import demo_interactive_agent

# Run the full interactive development demo
demo_interactive_agent()
```

## Migration from Old Modules

If you're migrating from the old standalone modules:

```python
# Old imports (deprecated, but still work with warnings)
from nemo_skills.code_execution.lean_prover import LeanProver
from lean_interactive_agent import InteractiveLeanAgent

# New imports (preferred)
from nemo_skills.code_execution.lean4 import LeanProver, InteractiveLeanAgent
```

## Key Features

### LeanProver Features
- ✅ Mathlib integration with TempRequireProject
- ✅ Incremental proof state management
- ✅ Backtracking support for proof exploration
- ✅ Multi-step tactic execution
- ✅ Environment management
- ✅ Clear success/failure/sorry status reporting

### InteractiveLeanAgent Features
- ✅ Real-time compilation and feedback (like VS Code)
- ✅ Position-aware editing with goal tracking
- ✅ Targeted clause updates (edit specific parts of proofs)
- ✅ Interactive panel showing messages/goals
- ✅ Incremental development with immediate validation
- ✅ Editable clause identification and management
- ✅ Next action suggestions for LLM agents

## Perfect for LLM Agents

Both components are designed to work seamlessly with LLM-driven theorem proving:
- **LeanProver**: Provides granular control over proof steps for systematic exploration
- **InteractiveLeanAgent**: Mimics human mathematician workflows for natural interaction
- **Immediate feedback**: Every edit provides instant compilation results
- **Structured development**: Supports complex proof construction workflows

This enables LLM agents to work with Lean 4 in the same way that expert mathematicians like Terence Tao develop proofs interactively.
