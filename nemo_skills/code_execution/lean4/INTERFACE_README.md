# Lean 4 Interactive Interfaces

This directory contains interactive interfaces for working with Lean 4 proofs, designed to mimic the VS Code Lean 4 extension experience and provide an agentic editing environment.

## 🎯 Overview

**Two complementary tools:**
- **Panel Viewer** (`panel_viewer.py`) - Shows proof state and compiler feedback like VS Code
- **Agentic Editor** (`agentic_editor.py`) - Interactive command-line interface for LLM-style editing

## 🚀 Quick Start

### 1. Run the Demo
```bash
python demo_interfaces.py
```

### 2. Launch Agentic Editor Directly
```bash
python agentic_editor.py
```

### 3. Test Panel Viewer
```bash
python panel_viewer.py
```

## 📋 Panel Viewer Features

The panel viewer displays information similar to VS Code's Lean 4 extension:

- **Compilation Status** - Success/failure with error counts
- **Compiler Messages** - Errors, warnings, and info messages with icons
- **Proof Goals** - Current goals at specific positions
- **Editable Clauses** - Identified sections you can edit
- **Annotated Code View** - Line-by-line annotations like VS Code hover
- **Watch Mode** - Continuous monitoring of changes

### Example Usage:
```python
from interactive_agent import InteractiveLeanAgent
from panel_viewer import LeanPanelViewer

agent = InteractiveLeanAgent(mathlib_enabled=True)
viewer = LeanPanelViewer(agent)

# Load theorem (e.g., copied from VS Code)
result = agent.load_theorem(your_theorem_code)

# View full panel state
viewer.display_full_panel()

# Show quick status
viewer.display_quick_status()

# View code with annotations
viewer.display_code_with_annotations()
```

## 🤖 Agentic Editor Features

Interactive command-line interface simulating how an LLM would work:

### Core Commands:
- `demo` - Load sample theorems
- `load [file]` - Load theorem from file or interactive input
- `panel` - Show complete state (like VS Code extension)
- `clauses` - Show all editable clauses
- `edit <clause_id> <new_content>` - Make targeted edits
- `suggest` - Get AI suggestions for next actions
- `fix` - Attempt automatic fixes

### Analysis Commands:
- `status` - Quick status summary
- `analyze` - Detailed proof analysis
- `messages` - Show compiler messages
- `goals` - Show proof goals
- `history` - Edit history

### Utility Commands:
- `save [filename]` - Save current proof
- `watch` - Monitor changes continuously
- `clear` - Clear screen
- `info` - Session information

### Example Session:
```
lean🤖> demo
🎯 Loading demo theorem 1...
✅ Demo loaded! Try 'panel', 'clauses', or 'edit sorry_0 <new_content>'

lean🤖> clauses
📝 EDITABLE CLAUSES (DETAILED)
📝 have_h1
   Type: have
   Content: sorry

📝 have_h2
   Type: have
   Content: sorry

lean🤖> edit have_h1 intro h; exact h.left
✏️ Editing clause 'have_h1'...
   New content: intro h; exact h.left

✅ SUCCESS (compilation successful)

lean🤖> suggest
🤖 AI SUGGESTIONS:
1. Work on sorry clauses: have_h2
2. Address compiler warnings
```

## 🔄 VS Code Workflow

### Copy from VS Code → Analyze:
1. Copy your theorem from VS Code
2. Run `python demo_interfaces.py`
3. Choose option 5 (Quick Panel Test)
4. Paste your theorem
5. Compare the feedback with VS Code

### Make Edits → Verify:
1. Use the agentic editor to make changes
2. View the panel state after each edit
3. Copy final proof back to VS Code if desired

## 📊 Interface Comparison

| Feature | Panel Viewer | Agentic Editor | VS Code Extension |
|---------|-------------|----------------|-------------------|
| Compilation Status | ✅ | ✅ | ✅ |
| Error Messages | ✅ | ✅ | ✅ |
| Goal Display | ✅ | ✅ | ✅ |
| Interactive Editing | ❌ | ✅ | ✅ |
| AI Suggestions | ❌ | ✅ | ❌ |
| Auto-fix Attempts | ❌ | ✅ | ❌ |
| Command History | ❌ | ✅ | ❌ |
| Watch Mode | ✅ | ✅ | ✅ |

## 🎯 Use Cases

### For Proof Development:
- Load incomplete proofs and work on them interactively
- Get AI suggestions for next steps
- Make targeted edits to specific clauses
- Monitor compilation status in real-time

### For VS Code Verification:
- Copy proofs from VS Code to verify compiler feedback matches
- Test different versions of proofs
- Debug compilation issues outside of VS Code

### For LLM Agent Development:
- Understand how an agent would interact with Lean 4
- Test agent strategies for proof completion
- Develop automated proof assistance tools

## 🛠️ Technical Details

### Architecture:
```
InteractiveLeanAgent (core)
├── LeanPanelViewer (display)
└── AgenticLeanEditor (interaction)
```

### Key Classes:

**InteractiveLeanAgent:**
- `load_theorem()` - Load and compile theorem
- `edit_clause()` - Make targeted edits
- `get_interactive_panel()` - Get current state
- `suggest_next_actions()` - AI suggestions

**LeanPanelViewer:**
- `display_full_panel()` - Complete state display
- `display_compilation_result()` - Show compilation feedback
- `display_code_with_annotations()` - Annotated code view
- `watch_mode()` - Continuous monitoring

**AgenticLeanEditor:**
- Command-line interface using Python's `cmd` module
- Tab completion for commands
- Edit history tracking
- Interactive help system

### Dependencies:
- `interactive_agent.py` - Core agent functionality
- `prover.py` - Lean 4 compilation interface
- Python standard library (`cmd`, `readline`, etc.)

## 🎉 Examples

See `demo_interfaces.py` for comprehensive examples of:
- Loading theorems from VS Code
- Making agentic edits
- Viewing compiler feedback
- Interactive proof development workflows

## 🔧 Customization

You can extend these interfaces by:
- Adding new display formats to `LeanPanelViewer`
- Adding new commands to `AgenticLeanEditor`
- Implementing custom AI suggestion algorithms
- Adding support for different theorem types

---

These interfaces bridge the gap between VS Code's visual experience and programmatic agent interaction, enabling both human verification and automated proof development workflows.
