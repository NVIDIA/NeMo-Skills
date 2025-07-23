#!/usr/bin/env python3
"""
Demo Script for Lean 4 Interactive Interfaces

This demonstrates how to use the panel viewer and agentic editor
to work with Lean 4 proofs, including copying from VS Code.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_agent import InteractiveLeanAgent
from panel_viewer import LeanPanelViewer
from agentic_editor import AgenticLeanEditor


def demo_panel_viewer():
    """Demonstrate the panel viewer with a VS Code-style proof."""
    print("🎯 PANEL VIEWER DEMO")
    print("=" * 80)
    print("This shows how to view proof state and compiler feedback,")
    print("similar to what you'd see in VS Code's Lean 4 extension.")
    print()

    # Create agent and viewer
    agent = InteractiveLeanAgent(mathlib_enabled=True)
    viewer = LeanPanelViewer(agent)

    # Example: A proof you might copy from VS Code
    vscode_proof = """theorem from_vscode (P Q R : Prop) : (P ∧ Q) → (P ∧ R) → (P ∧ (Q ∧ R)) := by
  have extract_P : (P ∧ Q) → P := by intro h; exact h.left
  have extract_Q : (P ∧ Q) → Q := by intro h; exact h.right
  have extract_R : (P ∧ R) → R := by intro h; exact h.right
  intros h1 h2
  constructor
  · exact extract_P h1
  · constructor
    · exact extract_Q h1
    · exact extract_R h2"""

    print("📋 Loading this theorem (as if copied from VS Code):")
    print("-" * 60)
    print(vscode_proof)
    print()

    # Load and compile
    print("🔄 Compiling...")
    result = agent.load_theorem(vscode_proof)

    # Show full panel state
    print("📊 PANEL STATE (like VS Code extension):")
    viewer.display_full_panel()

    # Show quick status
    print("📱 QUICK STATUS (like status bar):")
    viewer.display_quick_status()
    print()

    return agent, viewer


def demo_agentic_editing():
    """Demonstrate agentic editing workflow."""
    print("\n🤖 AGENTIC EDITING DEMO")
    print("=" * 80)
    print("This shows how to use the agentic editor interface")
    print("to make edits as if you were an LLM agent.")
    print()

    # Create agent and viewer
    agent = InteractiveLeanAgent(mathlib_enabled=True)
    viewer = LeanPanelViewer(agent)

    # Start with a proof that has sorries
    incomplete_proof = """theorem agentic_demo (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h2 h, h1 h⟩"""

    print("📝 Starting with incomplete proof:")
    print("-" * 60)
    print(incomplete_proof)
    print()

    # Load the proof
    result = agent.load_theorem(incomplete_proof)
    print("🔄 Initial compilation:")
    viewer.display_compilation_result(result)

    # Show editable clauses
    print("✏️ EDITABLE CLAUSES:")
    viewer.display_editable_clauses_detailed()

    # Simulate LLM-style editing
    print("🤖 SIMULATING LLM AGENT EDITS:")
    print("-" * 60)

    # Edit h1
    print("1. Fixing h1...")
    edit_result = agent.edit_clause("have_h1", "intro h; exact h.left")
    print(f"   Result: {'✅ Success' if edit_result['compilation_result']['success'] else '❌ Failed'}")

    # Edit h2
    print("2. Fixing h2...")
    edit_result = agent.edit_clause("have_h2", "intro h; exact h.right")
    print(f"   Result: {'✅ Success' if edit_result['compilation_result']['success'] else '❌ Failed'}")

    # Show final state
    print("\n📊 FINAL STATE:")
    viewer.display_full_panel()

    # Show suggestions
    print("🤖 AI SUGGESTIONS:")
    suggestions = agent.suggest_next_actions()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")

    return agent, viewer


def demo_vscode_workflow():
    """Demonstrate copying a proof from VS Code and working with it."""
    print("\n📋 VS CODE WORKFLOW DEMO")
    print("=" * 80)
    print("This shows how you might copy a proof from VS Code,")
    print("paste it into the tool, and verify you get the same feedback.")
    print()

    # Create agent and viewer
    agent = InteractiveLeanAgent(mathlib_enabled=True)
    viewer = LeanPanelViewer(agent)

    # Example of a proof that might have issues in VS Code
    vscode_problematic = """theorem problematic_proof (P Q : Prop) : P → Q → P ∧ Q := by
  intros h1 h2
  constructor
  exact h1
  exact h2"""

    print("📋 Proof copied from VS Code (with potential formatting issues):")
    print("-" * 60)
    print(vscode_problematic)
    print()

    # Load and analyze
    result = agent.load_theorem(vscode_problematic)

    print("🔍 ANALYSIS (should match VS Code feedback):")
    viewer.display_compilation_result(result)

    # Show annotated code view
    print("📝 ANNOTATED CODE VIEW (like VS Code hover info):")
    viewer.display_code_with_annotations()

    return agent, viewer


def interactive_demo():
    """Run an interactive demo where user can choose what to explore."""
    print("🚀 INTERACTIVE DEMO MENU")
    print("=" * 80)
    print()
    print("Choose a demo to run:")
    print("1. Panel Viewer Demo - See proof state and compiler feedback")
    print("2. Agentic Editor Demo - Make edits like an LLM agent")
    print("3. VS Code Workflow Demo - Copy from VS Code and analyze")
    print("4. Launch Full Agentic Editor - Interactive command-line interface")
    print("5. Quick Panel Test - Test with your own theorem")
    print()

    while True:
        try:
            choice = input("Enter choice (1-5) or 'q' to quit: ").strip()

            if choice == 'q':
                print("👋 Goodbye!")
                break

            elif choice == '1':
                demo_panel_viewer()

            elif choice == '2':
                demo_agentic_editing()

            elif choice == '3':
                demo_vscode_workflow()

            elif choice == '4':
                print("🚀 Launching full agentic editor...")
                print("Type 'help' for commands or 'demo' to start with a sample theorem.")
                print()
                editor = AgenticLeanEditor()
                editor.cmdloop()

            elif choice == '5':
                quick_panel_test()

            else:
                print("❌ Invalid choice. Please enter 1-5 or 'q'.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def quick_panel_test():
    """Quick test where user can paste their own theorem."""
    print("\n📝 QUICK PANEL TEST")
    print("=" * 80)
    print("Paste your theorem below (type 'END' on a new line when done):")
    print()

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        except EOFError:
            break

    theorem_code = '\n'.join(lines)

    if not theorem_code.strip():
        print("❌ No theorem provided.")
        return

    print("🔄 Analyzing your theorem...")

    # Create agent and viewer
    agent = InteractiveLeanAgent(mathlib_enabled=True)
    viewer = LeanPanelViewer(agent)

    # Load and analyze
    result = agent.load_theorem(theorem_code)

    # Show results
    print("\n📊 ANALYSIS RESULTS:")
    viewer.display_full_panel()

    print("\n💡 You can now:")
    print("- Copy this output to compare with VS Code")
    print("- Note any differences in error messages or warnings")
    print("- Use the editable clause IDs to make targeted edits")


def main():
    """Main entry point."""
    print("🎯 LEAN 4 INTERACTIVE INTERFACES DEMO")
    print("=" * 80)
    print()
    print("This demo shows two complementary tools:")
    print("• Panel Viewer - Shows proof state and compiler feedback (like VS Code)")
    print("• Agentic Editor - Interactive editing interface (like LLM agent)")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] == "panel":
            demo_panel_viewer()
        elif sys.argv[1] == "agentic":
            demo_agentic_editing()
        elif sys.argv[1] == "vscode":
            demo_vscode_workflow()
        elif sys.argv[1] == "editor":
            editor = AgenticLeanEditor()
            editor.cmdloop()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Options: panel, agentic, vscode, editor")
    else:
        interactive_demo()


if __name__ == "__main__":
    main()
