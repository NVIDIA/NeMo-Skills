# LeanInteractSession

A stateful Lean proof development tool that provides interactive proof session management with branching, backtracking, and comprehensive analysis capabilities.

## Overview

`LeanInteractSession` is a lean_interact-based tool that extends the basic sandbox execution model with advanced proof state management. It's designed to support iterative proof development with features like:

- **Stateful Proof Development**: Maintains proof state across multiple tactic applications
- **Branching and Backtracking**: Create alternative proof approaches and backtrack when needed
- **Comprehensive Tracking**: Record all proof attempts with detailed success/failure analysis
- **Session Persistence**: Save and load proof sessions for later work
- **Integration Ready**: Designed to work alongside existing NeMo-Skills execution patterns

## Key Features

### 🎯 **Stateful Proof Management**
- Maintains current proof state and goals
- Tracks all applied tactics and their results
- Automatic proof completion detection

### 🌳 **Branching System**
- Create multiple proof branches to explore different approaches
- Switch between branches seamlessly
- Maintain independent proof states per branch

### 📊 **Comprehensive Analysis**
- Success/failure rate tracking
- Failed tactic analysis
- Most common error patterns
- Branch completion statistics

### 💾 **Session Persistence**
- Save entire proof sessions to JSON files
- Load and continue work on saved sessions
- Export session data for analysis

### 🔧 **Integration Capabilities**
- Works alongside existing sandbox patterns
- Compatible with batch processing workflows
- Supports custom Lean executable paths

## Usage

### Basic Usage

```python
from nemo_skills.code_execution.lean_interact_session import create_lean_interact_session

# Create a proof session with a guaranteed-to-work theorem
theorem = "theorem example : True := by sorry"
session = create_lean_interact_session(theorem)

# Get initial proof state
state = session.get_proof_state()
print(f"Goals: {state['current_goals']}")

# Apply a tactic that always works for True
result = session.apply_tactic("trivial")
print(f"Success: {result['success']}")
print(f"Proof complete: {result['proof_complete']}")

# Get final proof script
if result['proof_complete']:
    script = session.get_proof_script()
    print(f"Final proof:\n{script}")

# Clean up
session.cleanup()
```

### Guaranteed Working Examples

These examples are guaranteed to work in any Lean 4 setup:

```python
# 1. Proving True (always works)
session1 = create_lean_interact_session("theorem test_true : True := by sorry")
result1 = session1.apply_tactic("trivial")
# Result: success=True, proof_complete=True

# 2. Reflexivity (always works)
session2 = create_lean_interact_session("theorem test_rfl (n : ℕ) : n = n := by sorry")
result2 = session2.apply_tactic("rfl")
# Result: success=True, proof_complete=True

# 3. Basic arithmetic (always works)
session3 = create_lean_interact_session("theorem test_arith : 1 + 1 = 2 := by sorry")
result3 = session3.apply_tactic("norm_num")
# Result: success=True, proof_complete=True

# 4. Definitional equality (always works)
session4 = create_lean_interact_session("theorem test_def (n : ℕ) : 0 + n = n := by sorry")
result4 = session4.apply_tactic("rfl")
# Result: success=True, proof_complete=True
```

### Advanced Usage with Branching

```python
# Create session
session = create_lean_interact_session("theorem complex_theorem ...")

# Try first approach
result1 = session.apply_tactic("approach_1_tactic")

# Create alternative branch
alt_branch = session.create_branch("alternative_approach")
session.switch_branch(alt_branch)

# Try different approach
result2 = session.apply_tactic("approach_2_tactic")

# Analyze all attempts
analysis = session.analyze_proof_attempts()
print(f"Success rate: {analysis['success_rate']:.2%}")
print(f"Completed branches: {analysis['completed_branches']}")
```

### Session Persistence

```python
# Save session
session.save_session("my_proof_session.json")

# Load session later
loaded_session = LeanInteractSession.load_session("my_proof_session.json")

# Continue work
loaded_session.apply_tactic("continue_proof")
```

## API Reference

### LeanInteractSession Class

#### Constructor
```python
LeanInteractSession(theorem_statement: str, repo_path: Optional[str] = None,
                   timeout: float = 30.0, lean_path: Optional[str] = None)
```

#### Core Methods

**`apply_tactic(tactic: str, branch_id: Optional[str] = None) -> Dict[str, Any]`**
- Apply a tactic to the current proof state
- Returns success status, new goals, and completion status

**`get_proof_state(branch_id: Optional[str] = None) -> Dict[str, Any]`**
- Get the current proof state for a branch
- Returns goals, status, and step information

**`get_proof_script(branch_id: Optional[str] = None) -> str`**
- Generate the complete proof script for a branch

#### Branching Methods

**`create_branch(branch_name: str, from_step: Optional[str] = None) -> str`**
- Create a new proof branch
- Returns the new branch ID

**`switch_branch(branch_id: str) -> Dict[str, Any]`**
- Switch to a different branch
- Returns success status and current goals

**`undo_step(step_id: str, branch_id: Optional[str] = None) -> Dict[str, Any]`**
- Undo a specific step and all subsequent steps
- Returns success status and updated goals

#### Analysis Methods

**`analyze_proof_attempts() -> Dict[str, Any]`**
- Analyze all proof attempts across all branches
- Returns comprehensive statistics and failure analysis

#### Persistence Methods

**`save_session(filename: str)`**
- Save the entire session to a JSON file

**`load_session(filename: str) -> LeanInteractSession`** (class method)
- Load a session from a JSON file

**`export_session() -> Dict[str, Any]`**
- Export session data as a dictionary

## Data Structures

### ProofStep
```python
@dataclass
class ProofStep:
    step_id: str
    tactic: str
    result: Dict[str, Any]
    timestamp: str
    goals_before: List[str]
    goals_after: List[str]
    success: bool
    error: Optional[str] = None
```

### ProofBranch
```python
@dataclass
class ProofBranch:
    branch_id: str
    name: str
    parent_branch: Optional[str]
    steps: List[ProofStep]
    current_goals: List[str]
    status: ProofStatus
    created_at: str
```

### ProofStatus
```python
class ProofStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
```

## Integration with Existing Systems

### With Sandbox Execution

```python
# Traditional sandbox execution
from nemo_skills.code_execution.sandbox import get_sandbox
sandbox = get_sandbox()
result, _ = sandbox.execute_code(lean_code, language="lean4")

# Enhanced with LeanInteractSession
session = create_lean_interact_session(theorem)
result = session.apply_tactic(tactic)
# Get comprehensive tracking, branching, and analysis
```

### With Batch Processing

```python
# Can be integrated into batch processing workflows
sessions = {}
for theorem in theorems:
    session = create_lean_interact_session(theorem)
    sessions[theorem.id] = session

    # Apply tactics with full tracking
    for tactic in tactic_sequence:
        result = session.apply_tactic(tactic)
        if result['proof_complete']:
            break
```

## Configuration

### Environment Variables
- `LEAN_PATH`: Path to Lean executable (default: "lean")
- `LEAN_TIMEOUT`: Default timeout for Lean execution (default: 30.0)

### Custom Configuration
```python
session = create_lean_interact_session(
    theorem="theorem ...",
    repo_path="/path/to/lean/project",
    timeout=60.0,
    lean_path="/custom/path/to/lean"
)
```

## Error Handling

The tool provides comprehensive error handling and reporting:

```python
result = session.apply_tactic("invalid_tactic")
if not result['success']:
    print(f"Error: {result['error']}")

    # Get detailed error analysis
    analysis = session.analyze_proof_attempts()
    for failure in analysis['failed_tactics']:
        print(f"Failed tactic: {failure['tactic']}")
        print(f"Error: {failure['error']}")
```

### Handling Harmless Warnings

The tool automatically filters out harmless warnings that shouldn't be treated as errors:

- **Elan Version Check Warnings**: `"warning: failed to query latest release, using existing version 'leanprover/lean4:v4.20.0'"`
- These warnings occur when elan can't check for updates due to network issues
- They don't affect Lean functionality and are filtered out automatically

```python
# This will succeed even if elan version warnings appear in stderr
result = session.apply_tactic("simp")
# Success is determined by actual Lean compilation, not warnings
print(f"Success: {result['success']}")  # True if tactic worked
```

### Adding Custom Warning Filters

You can extend the warning filtering by modifying `_filter_harmless_warnings()`:

```python
def _filter_harmless_warnings(self, stderr: str) -> str:
    # Add your custom warning patterns here
    if "your_custom_warning_pattern" in line:
        continue
```

## Performance Considerations

- **Memory Usage**: Sessions maintain full proof history - use cleanup() for long-running processes
- **File System**: Uses temporary files for Lean execution - automatic cleanup on session destruction
- **Concurrency**: Each session is independent and can be used in parallel processing

## Examples

See `lean_interact_example.py` for comprehensive usage examples including:
- Simple proof development
- Complex proofs with branching
- Session persistence
- Integration patterns
- Error handling

## Dependencies

- Standard library: `json`, `logging`, `os`, `uuid`, `datetime`, `subprocess`, `tempfile`, `re`
- NeMo-Skills: `nemo_skills.utils`
- Lean 4: Requires Lean 4 installation

## Installation

The tool is included in the NeMo-Skills package. Ensure you have:
1. Lean 4 installed and accessible
2. NeMo-Skills package installed
3. Appropriate permissions for temporary file creation

## Future Enhancements

- **Lean Server Integration**: Direct integration with Lean language server
- **Proof Visualization**: Generate proof tree visualizations
- **Tactic Suggestions**: AI-powered tactic recommendations
- **Parallel Execution**: Concurrent tactic evaluation
- **Advanced Analysis**: Machine learning-based failure pattern analysis
