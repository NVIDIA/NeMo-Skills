#!/usr/bin/env python3
"""
Agentic Lean 4 Editor - Interactive LLM-Style Interface

This provides an interactive command-line interface for editing Lean 4 proofs
using the InteractiveLeanAgent, simulating how an LLM would work with the system.
"""

import os
import sys
import cmd
import readline
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_agent import InteractiveLeanAgent
from panel_viewer import LeanPanelViewer


class AgenticLeanEditor(cmd.Cmd):
    """
    Interactive command-line editor for Lean 4 proofs.
    Simulates how an LLM agent would interact with the Interactive Lean Agent.
    """

    intro = """
🤖 AGENTIC LEAN 4 EDITOR
========================
Welcome to the interactive Lean 4 proof editor!
This simulates how an LLM agent would work with the Interactive Lean Agent.

Type 'help' or '?' for available commands.
Type 'demo' to load a sample theorem and start editing.
    """

    prompt = "lean🤖> "

    def __init__(self):
        super().__init__()
        self.agent = InteractiveLeanAgent(mathlib_enabled=True)
        self.viewer = LeanPanelViewer(self.agent)
        self.current_theorem_name = None
        self.edit_history = []

    # ========================================
    # Core Commands
    # ========================================

    def do_load(self, line):
        """Load a theorem from input or file.
        Usage: load [filename]
        If no filename, enters multi-line input mode."""

        if line.strip():
            # Load from file
            try:
                with open(line.strip(), 'r') as f:
                    theorem_code = f.read()
                print(f"📁 Loading theorem from {line.strip()}")
            except FileNotFoundError:
                print(f"❌ File not found: {line.strip()}")
                return
        else:
            # Multi-line input mode
            print("📝 Enter your theorem (type 'END' on a new line to finish):")
            lines = []
            while True:
                try:
                    line_input = input()
                    if line_input.strip() == 'END':
                        break
                    lines.append(line_input)
                except EOFError:
                    break
            theorem_code = '\n'.join(lines)

        if not theorem_code.strip():
            print("❌ No theorem code provided.")
            return

        # Load the theorem
        print("🔄 Loading and compiling theorem...")
        result = self.agent.load_theorem(theorem_code)

        # Extract theorem name
        import re
        theorem_match = re.search(r'theorem\s+(\w+)', theorem_code)
        if theorem_match:
            self.current_theorem_name = theorem_match.group(1)

        # Show compilation result
        self.viewer.display_compilation_result(result)

        print("✅ Theorem loaded! Type 'panel' to see the full state.")

    def do_demo(self, line):
        """Load a demo theorem to start experimenting."""
        demo_theorems = {
            "1": """theorem demo_and_comm (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h2 h, h1 h⟩""",

            "2": """theorem demo_impl_trans (P Q R : Prop) : (P → Q) → (Q → R) → (P → R) := by
  have step1 : (P → Q) → (Q → R) → P → R := by sorry
  exact step1""",

            "3": """theorem demo_assoc (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  have h3 : P ∧ Q → P := by sorry
  have h4 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h3 (h1 h), ⟨h4 (h1 h), h2 h⟩⟩"""
        }

        choice = line.strip() if line.strip() in demo_theorems else "1"
        theorem_code = demo_theorems[choice]

        print(f"🎯 Loading demo theorem {choice}...")
        result = self.agent.load_theorem(theorem_code)

        self.current_theorem_name = f"demo_{choice}"
        self.viewer.display_compilation_result(result)

        print("✅ Demo loaded! Try 'panel', 'clauses', or 'edit sorry_0 <new_content>'")

    def do_panel(self, line):
        """Show the complete interactive panel (like VS Code extension)."""
        self.viewer.display_full_panel()

    def do_status(self, line):
        """Show quick status summary."""
        self.viewer.display_quick_status()

    def do_messages(self, line):
        """Show detailed compiler messages."""
        panel = self.agent.get_interactive_panel()
        self.viewer._display_messages(panel['messages'])

    def do_goals(self, line):
        """Show current proof goals."""
        panel = self.agent.get_interactive_panel()
        self.viewer._display_goals(panel['goals'])

    def do_clauses(self, line):
        """Show all editable clauses with details."""
        self.viewer.display_editable_clauses_detailed()

    def do_code(self, line):
        """Show current code with annotations."""
        self.viewer.display_code_with_annotations()

    # ========================================
    # Editing Commands
    # ========================================

    def do_edit(self, line):
        """Edit a specific clause.
        Usage: edit <clause_id> <new_content>
        Example: edit sorry_0 intro h; exact h.left"""

        parts = line.split(' ', 1)
        if len(parts) < 2:
            print("❌ Usage: edit <clause_id> <new_content>")
            print("💡 Use 'clauses' to see available clause IDs")
            return

        clause_id = parts[0]
        new_content = parts[1]

        if clause_id not in self.agent.editable_clauses:
            print(f"❌ Clause '{clause_id}' not found.")
            print("💡 Available clauses:")
            for cid in self.agent.editable_clauses.keys():
                print(f"   - {cid}")
            return

        print(f"✏️ Editing clause '{clause_id}'...")
        print(f"   New content: {new_content}")

        # Make the edit
        edit_result = self.agent.edit_clause(clause_id, new_content)

        # Record in history
        self.edit_history.append({
            'clause_id': clause_id,
            'old_content': edit_result['old_content'],
            'new_content': new_content,
            'timestamp': len(self.edit_history) + 1
        })

        # Show feedback
        self.viewer.display_edit_feedback(edit_result)

    def do_suggest(self, line):
        """Get AI suggestions for next actions."""
        suggestions = self.agent.suggest_next_actions()

        print("🤖 AI SUGGESTIONS:")
        print("-" * 50)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

        # Additional context-aware suggestions
        sorry_clauses = [cid for cid in self.agent.editable_clauses.keys() if 'sorry' in cid]
        if sorry_clauses:
            print("\n💡 SPECIFIC SUGGESTIONS:")
            print(f"   Try: edit {sorry_clauses[0]} <your_proof>")

        print("\n🔧 COMMON TACTICS:")
        print("   - intro h        (introduce hypothesis)")
        print("   - exact h.left   (use left part of hypothesis)")
        print("   - exact h.right  (use right part of hypothesis)")
        print("   - exact ⟨a, b⟩   (construct pair)")

    def do_fix(self, line):
        """Attempt to automatically fix common issues."""
        print("🔧 Analyzing current state for auto-fixes...")

        # Look for common patterns to fix
        fixes_applied = 0

        # Fix simple sorry clauses with obvious solutions
        for clause_id, clause in self.agent.editable_clauses.items():
            if clause.clause_type == 'sorry':
                # Suggest based on context
                if 'left' in clause_id or 'h1' in clause_id:
                    suggestion = "intro h; exact h.left"
                elif 'right' in clause_id or 'h2' in clause_id:
                    suggestion = "intro h; exact h.right"
                else:
                    continue  # Skip if we can't suggest

                print(f"   🔧 Suggesting fix for {clause_id}: {suggestion}")
                response = input(f"   Apply this fix? (y/N): ")
                if response.lower() == 'y':
                    edit_result = self.agent.edit_clause(clause_id, suggestion)
                    fixes_applied += 1
                    print(f"   ✅ Applied fix to {clause_id}")

        if fixes_applied == 0:
            print("   No obvious fixes found. Try 'suggest' for manual suggestions.")
        else:
            print(f"   ✅ Applied {fixes_applied} fixes!")

    # ========================================
    # Analysis Commands
    # ========================================

    def do_analyze(self, line):
        """Analyze the current proof structure and provide insights."""
        print("🔍 PROOF ANALYSIS")
        print("=" * 50)

        # Basic statistics
        panel = self.agent.get_interactive_panel()
        error_count = sum(1 for msg in self.agent.current_messages if msg.severity == 'error')
        warning_count = sum(1 for msg in self.agent.current_messages if msg.severity == 'warning')
        sorry_count = len([c for c in self.agent.editable_clauses.values() if c.clause_type == 'sorry'])

        print(f"📊 Statistics:")
        print(f"   Errors: {error_count}")
        print(f"   Warnings: {warning_count}")
        print(f"   Sorry clauses: {sorry_count}")
        print(f"   Editable clauses: {len(self.agent.editable_clauses)}")

        # Analyze proof structure
        print(f"\n🏗️ Proof Structure:")
        have_clauses = [c for c in self.agent.editable_clauses.values() if c.clause_type == 'have']
        tactic_clauses = [c for c in self.agent.editable_clauses.values() if c.clause_type in ['tactic', 'tactic_block']]

        print(f"   Have statements: {len(have_clauses)}")
        print(f"   Tactic blocks: {len(tactic_clauses)}")

        # Completion assessment
        if error_count == 0 and sorry_count == 0:
            print(f"\n🎉 Proof appears complete!")
        elif error_count == 0:
            print(f"\n⚠️ No errors, but {sorry_count} sorry clauses remain.")
        else:
            print(f"\n❌ {error_count} errors need to be fixed first.")

    def do_history(self, line):
        """Show edit history."""
        if not self.edit_history:
            print("No edits made yet.")
            return

        print("📜 EDIT HISTORY")
        print("=" * 50)
        for i, edit in enumerate(self.edit_history, 1):
            print(f"{i}. {edit['clause_id']}: {edit['old_content']} → {edit['new_content']}")

    def do_undo(self, line):
        """Undo the last edit (simplified - reloads from history)."""
        if not self.edit_history:
            print("Nothing to undo.")
            return

        print("⏪ Undo not implemented yet - consider reloading the theorem.")

    # ========================================
    # Utility Commands
    # ========================================

    def do_clear(self, line):
        """Clear the screen."""
        os.system('clear' if os.name == 'posix' else 'cls')

    def do_save(self, line):
        """Save current theorem to a file.
        Usage: save [filename]"""

        filename = line.strip() if line.strip() else f"{self.current_theorem_name}.lean"

        try:
            with open(filename, 'w') as f:
                f.write(self.agent.current_code)
            print(f"💾 Saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving: {e}")

    def do_watch(self, line):
        """Start watch mode to monitor changes continuously."""
        print("Starting watch mode...")
        try:
            self.viewer.watch_mode()
        except KeyboardInterrupt:
            print("Watch mode ended.")

    def do_quit(self, line):
        """Exit the editor."""
        print("👋 Goodbye!")
        return True

    def do_exit(self, line):
        """Exit the editor."""
        return self.do_quit(line)

    # ========================================
    # Help and Information
    # ========================================

    def do_info(self, line):
        """Show information about the current session."""
        print("ℹ️ SESSION INFO")
        print("=" * 50)
        print(f"Current theorem: {self.current_theorem_name or 'None'}")
        print(f"Compilation ID: {self.agent.compilation_id}")
        print(f"Edits made: {len(self.edit_history)}")
        print(f"Agent mathlib: {'Enabled' if self.agent.prover.mathlib_enabled else 'Disabled'}")

    def do_examples(self, line):
        """Show example commands."""
        print("💡 EXAMPLE COMMANDS")
        print("=" * 50)
        print()
        print("📥 Loading:")
        print("   demo              - Load a demo theorem")
        print("   load myfile.lean  - Load from file")
        print("   load              - Enter theorem interactively")
        print()
        print("👀 Viewing:")
        print("   panel             - Full interactive panel")
        print("   clauses           - Show editable clauses")
        print("   code              - Code with annotations")
        print("   status            - Quick status")
        print()
        print("✏️ Editing:")
        print("   edit sorry_0 intro h; exact h.left")
        print("   edit have_h1 intro h; exact h.right")
        print("   suggest           - Get AI suggestions")
        print("   fix               - Auto-fix common issues")
        print()
        print("🔍 Analysis:")
        print("   analyze           - Analyze proof structure")
        print("   history           - Show edit history")
        print("   messages          - Show compiler messages")

    def help_editing(self):
        """Detailed help for editing commands."""
        print("""
🎯 EDITING WORKFLOW:

1. Load a theorem:
   demo                    # Load sample theorem

2. View current state:
   panel                   # See everything
   clauses                 # See what you can edit

3. Make edits:
   edit sorry_0 intro h; exact h.left
   edit have_h1 intro h; exact h.right

4. Get suggestions:
   suggest                 # AI suggestions
   fix                     # Auto-fix attempts

5. Monitor progress:
   status                  # Quick check
   analyze                 # Detailed analysis

🔧 Common Lean 4 tactics:
   intro h                 # Introduce hypothesis h
   exact e                 # Provide exact proof e
   exact h.left            # Use left part of conjunction
   exact h.right           # Use right part of conjunction
   exact ⟨a, b⟩            # Construct pair/conjunction
        """)

    def default(self, line):
        """Handle unknown commands."""
        print(f"❓ Unknown command: {line}")
        print("💡 Type 'help' for available commands or 'examples' for examples.")


def main():
    """Main entry point for the agentic editor."""
    try:
        editor = AgenticLeanEditor()
        editor.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Please report this issue.")


if __name__ == "__main__":
    main()
