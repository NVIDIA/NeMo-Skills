#!/usr/bin/env python3
"""
Lean 4 Panel Viewer - Interactive State Display

This provides a visual interface to view the proof state and compiler feedback
from the InteractiveLeanAgent, mimicking the VS Code Lean 4 extension panels.
"""

import os
import sys
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_agent import InteractiveLeanAgent, LeanMessage, ProofGoal, EditableClause


class LeanPanelViewer:
    """
    Visual panel viewer for Interactive Lean 4 development.
    Displays compilation results, messages, goals, and editable clauses.
    """

    def __init__(self, agent: InteractiveLeanAgent):
        self.agent = agent
        self.last_compilation_id = 0

    def display_full_panel(self, show_code: bool = True) -> None:
        """Display the complete interactive panel state."""
        panel = self.agent.get_interactive_panel()

        self._print_header("LEAN 4 INTERACTIVE PANEL")

        if show_code:
            self._display_current_code(panel['current_code'])

        self._display_compilation_status(panel)
        self._display_messages(panel['messages'])
        self._display_goals(panel['goals'])
        self._display_editable_clauses(panel['editable_clauses'])
        self._display_suggestions()

        self._print_separator()

    def display_compilation_result(self, result: Dict[str, Any]) -> None:
        """Display results from a compilation/edit operation."""
        self._print_header("COMPILATION RESULT")

        # Status indicators
        status_line = "Status: "
        if result['success']:
            status_line += "✅ SUCCESS"
        else:
            status_line += "❌ FAILED"

        if result.get('has_errors'):
            status_line += " (with errors)"
        elif result.get('has_warnings'):
            status_line += " (with warnings)"

        if result.get('has_sorry'):
            status_line += " [contains sorry]"

        print(status_line)
        print(f"Compilation ID: {result.get('compilation_id', 'N/A')}")
        print()

        # Messages
        if result.get('messages'):
            self._display_messages([str(msg) for msg in result['messages']])

    def display_edit_feedback(self, edit_result: Dict[str, Any]) -> None:
        """Display feedback from an edit operation."""
        self._print_header("EDIT FEEDBACK")

        print(f"Clause ID: {edit_result['clause_id']}")
        print(f"Clause Type: {edit_result['clause_type']}")
        print(f"Edit Status: {'✅ Applied' if edit_result['edit_successful'] else '❌ Failed'}")
        print()

        print("CHANGE SUMMARY:")
        print(f"  Old: {edit_result['old_content']}")
        print(f"  New: {edit_result['new_content']}")
        print()

        # Show compilation result
        self.display_compilation_result(edit_result['compilation_result'])

    def display_quick_status(self) -> None:
        """Display a quick status summary (like a status bar)."""
        panel = self.agent.get_interactive_panel()

        # Count message types
        error_count = sum(1 for msg in self.agent.current_messages if msg.severity == 'error')
        warning_count = sum(1 for msg in self.agent.current_messages if msg.severity == 'warning')
        goal_count = len(self.agent.current_goals)
        clause_count = len(self.agent.editable_clauses)

        status_parts = []

        if error_count > 0:
            status_parts.append(f"❌ {error_count} errors")
        if warning_count > 0:
            status_parts.append(f"⚠️ {warning_count} warnings")
        if goal_count > 0:
            status_parts.append(f"🎯 {goal_count} goals")
        if clause_count > 0:
            status_parts.append(f"✏️ {clause_count} editable")

        if not status_parts:
            status_parts.append("✅ All good")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] {' | '.join(status_parts)}")

    def display_code_with_annotations(self) -> None:
        """Display code with line-by-line annotations for messages and goals."""
        self._print_header("ANNOTATED CODE VIEW")

        lines = self.agent.current_code.split('\n')

        for i, line in enumerate(lines, 1):
            # Print line number and content
            print(f"{i:3d}: {line}")

            # Check for messages at this line
            line_messages = self.agent.get_messages_at_position(i-1, 0)
            for msg in line_messages:
                icon = "❌" if msg.severity == 'error' else "⚠️" if msg.severity == 'warning' else "ℹ️"
                print(f"     {icon} {msg.message}")

            # Check for goals at this line
            goal = self.agent.get_goal_at_position(i-1, 0)
            if goal:
                print(f"     🎯 {goal.goal_text}")

        print()

    def display_editable_clauses_detailed(self) -> None:
        """Display detailed view of all editable clauses."""
        self._print_header("EDITABLE CLAUSES (DETAILED)")

        if not self.agent.editable_clauses:
            print("No editable clauses found.")
            return

        for clause_id, clause in self.agent.editable_clauses.items():
            print(f"📝 {clause_id}")
            print(f"   Type: {clause.clause_type}")
            print(f"   Position: Line {clause.start_pos.line + 1}, Col {clause.start_pos.column}")
            print(f"   Content: {clause.content}")
            print(f"   Full: {clause}")
            print()

    def watch_mode(self, refresh_interval: int = 2) -> None:
        """Continuous monitoring mode that refreshes the panel."""
        import time

        print("🔍 WATCH MODE - Press Ctrl+C to exit")
        print("=" * 60)

        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')

                # Check if compilation changed
                current_id = self.agent.compilation_id
                if current_id != self.last_compilation_id:
                    print("🔄 COMPILATION UPDATED")
                    self.last_compilation_id = current_id

                self.display_full_panel(show_code=False)
                self.display_quick_status()

                print(f"\n⏱️ Refreshing every {refresh_interval}s... (Ctrl+C to exit)")
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\n👋 Watch mode ended.")

    def _display_current_code(self, code: str) -> None:
        """Display the current code with syntax highlighting."""
        self._print_subheader("CURRENT CODE")

        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
        print()

    def _display_compilation_status(self, panel: Dict[str, Any]) -> None:
        """Display compilation status summary."""
        self._print_subheader("COMPILATION STATUS")

        # Determine overall status
        has_errors = any('error' in msg.lower() for msg in panel['messages'])
        has_warnings = any('warning' in msg.lower() for msg in panel['messages'])

        if has_errors:
            print("Status: ❌ COMPILATION FAILED")
        elif has_warnings:
            print("Status: ⚠️ COMPILED WITH WARNINGS")
        else:
            print("Status: ✅ COMPILATION SUCCESSFUL")

        print(f"Compilation ID: {panel['compilation_id']}")
        print()

    def _display_messages(self, messages: List[str]) -> None:
        """Display compiler messages."""
        self._print_subheader("COMPILER MESSAGES")

        if not messages:
            print("No messages.")
            print()
            return

        for i, msg in enumerate(messages, 1):
            # Add appropriate icon based on message type
            if '[error]' in msg.lower():
                icon = "❌"
            elif '[warning]' in msg.lower():
                icon = "⚠️"
            else:
                icon = "ℹ️"

            print(f"{icon} {msg}")

        print()

    def _display_goals(self, goals: List[str]) -> None:
        """Display proof goals."""
        self._print_subheader("PROOF GOALS")

        if not goals:
            print("No active goals.")
            print()
            return

        for i, goal in enumerate(goals, 1):
            print(f"🎯 Goal {i}: {goal}")

        print()

    def _display_editable_clauses(self, clauses: Dict[str, str]) -> None:
        """Display editable clauses."""
        self._print_subheader("EDITABLE CLAUSES")

        if not clauses:
            print("No editable clauses identified.")
            print()
            return

        for clause_id, description in clauses.items():
            # Add icon based on clause type
            if 'sorry' in clause_id:
                icon = "💔"
            elif 'have' in clause_id:
                icon = "📝"
            elif 'tactic' in clause_id:
                icon = "🔧"
            else:
                icon = "✏️"

            print(f"{icon} {clause_id}: {description}")

        print()

    def _display_suggestions(self) -> None:
        """Display next action suggestions."""
        self._print_subheader("SUGGESTED NEXT ACTIONS")

        suggestions = self.agent.suggest_next_actions()

        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

        print()

    def _print_header(self, title: str) -> None:
        """Print a main header."""
        print("=" * 80)
        print(f" {title}")
        print("=" * 80)
        print()

    def _print_subheader(self, title: str) -> None:
        """Print a subsection header."""
        print(f"📋 {title}")
        print("-" * 60)

    def _print_separator(self) -> None:
        """Print a section separator."""
        print("=" * 80)
        print()


def demo_panel_viewer():
    """Demonstrate the panel viewer with a sample theorem."""
    print("🚀 LEAN 4 PANEL VIEWER DEMO")
    print("=" * 80)

    # Create agent and load theorem
    agent = InteractiveLeanAgent(mathlib_enabled=True)
    viewer = LeanPanelViewer(agent)

    # Load a theorem to demonstrate
    theorem_code = """theorem demo_theorem (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h2 h, h1 h⟩"""

    print("Loading theorem...")
    result = agent.load_theorem(theorem_code)

    # Show full panel
    viewer.display_full_panel()

    # Demonstrate edit feedback
    print("Making an edit to h1...")
    edit_result = agent.edit_clause("have_h1", "intro h; exact h.left")
    viewer.display_edit_feedback(edit_result)

    # Show annotated code view
    viewer.display_code_with_annotations()

    # Show detailed clauses
    viewer.display_editable_clauses_detailed()

    print("✅ Panel viewer demo complete!")
    print("\nTry running viewer.watch_mode() for continuous monitoring!")


if __name__ == "__main__":
    demo_panel_viewer()
