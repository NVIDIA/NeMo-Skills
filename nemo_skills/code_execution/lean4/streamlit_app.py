#!/usr/bin/env python3
"""
Unified Streamlit App for Lean 4 Interactive Development

This combines both the panel viewer and agentic editor functionality
into a single web-based interface for comprehensive Lean 4 proof development.
"""

import streamlit as st
import os
import sys
import time
from typing import Dict, Any, List

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from interactive_agent import InteractiveLeanAgent
    from panel_viewer import LeanPanelViewer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the correct directory with nemo-skills activated")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Lean 4 Interactive Development",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = InteractiveLeanAgent(mathlib_enabled=True)
    st.session_state.viewer = LeanPanelViewer(st.session_state.agent)
    st.session_state.edit_history = []
    st.session_state.current_theorem_name = None
    st.session_state.theorem_history = []

def load_theorem_sidebar():
    """Sidebar for loading theorems."""
    with st.sidebar:
        st.header("📋 Theorem Management")

        # Demo theorems
        demo_options = {
            "None": "",
            "Simple Conjunction": """theorem simple_conj (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h2 h, h1 h⟩""",
            "Multiple Sorries": """theorem multi_sorry (P Q R : Prop) : P ∧ Q ∧ R → R ∧ Q ∧ P := by
  have extract_P : P ∧ Q ∧ R → P := by sorry
  have extract_Q : P ∧ Q ∧ R → Q := by sorry
  have extract_R : P ∧ Q ∧ R → R := by sorry
  intro h
  exact ⟨extract_R h, extract_Q h, extract_P h⟩""",
            "Associativity": """theorem assoc_demo (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  have h3 : P ∧ Q → P := by sorry
  have h4 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h3 (h1 h), ⟨h4 (h1 h), h2 h⟩⟩""",
            "Implication Chain": """theorem impl_chain (P Q R S : Prop) : (P → Q) → (Q → R) → (R → S) → (P → S) := by
  have step1 : (P → Q) → (Q → R) → P → R := by sorry
  have step2 : (P → R) → (R → S) → P → S := by sorry
  intros h1 h2 h3
  exact h3 (step1 h1 h2)"""
        }

        selected_demo = st.selectbox("Choose demo theorem:", list(demo_options.keys()))

        if selected_demo != "None" and st.button("Load Demo"):
            theorem_code = demo_options[selected_demo]
            with st.spinner("Loading and compiling theorem..."):
                result = st.session_state.agent.load_theorem(theorem_code)
                st.session_state.current_theorem_name = selected_demo
                st.session_state.edit_history = []
                st.session_state.theorem_history.append({
                    'name': selected_demo,
                    'code': theorem_code,
                    'timestamp': time.time()
                })
            st.success(f"Loaded: {selected_demo}")
            st.rerun()

        # Custom theorem input
        st.subheader("Custom Theorem")
        custom_theorem = st.text_area(
            "Paste your theorem here:",
            height=200,
            placeholder="theorem my_theorem (P Q : Prop) : P → Q → P ∧ Q := by\n  sorry"
        )

        if st.button("Load Custom Theorem"):
            if custom_theorem.strip():
                with st.spinner("Loading and compiling theorem..."):
                    result = st.session_state.agent.load_theorem(custom_theorem)
                    st.session_state.current_theorem_name = "Custom"
                    st.session_state.edit_history = []
                    st.session_state.theorem_history.append({
                        'name': 'Custom',
                        'code': custom_theorem,
                        'timestamp': time.time()
                    })
                st.success("Custom theorem loaded!")
                st.rerun()
            else:
                st.error("Please enter a theorem.")

        # History
        if st.session_state.theorem_history:
            st.subheader("📜 Recent Theorems")
            for i, entry in enumerate(reversed(st.session_state.theorem_history[-3:])):
                if st.button(f"{entry['name'][:15]}...", key=f"history_{i}"):
                    result = st.session_state.agent.load_theorem(entry['code'])
                    st.session_state.current_theorem_name = entry['name']
                    st.session_state.edit_history = []
                    st.rerun()

        # Current status
        if st.session_state.agent.current_code:
            st.divider()
            st.subheader("📊 Current Status")

            error_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'error')
            warning_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'warning')
            sorry_count = len([c for c in st.session_state.agent.editable_clauses.values() if c.clause_type == 'sorry'])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Errors", error_count)
                st.metric("Warnings", warning_count)
            with col2:
                st.metric("Sorries", sorry_count)
                st.metric("Clauses", len(st.session_state.agent.editable_clauses))

            if error_count == 0 and sorry_count == 0:
                st.success("✅ Complete!")
            elif error_count == 0:
                st.warning(f"⚠️ {sorry_count} sorries remain")
            else:
                st.error(f"❌ {error_count} errors to fix")

def panel_viewer_tab():
    """Panel viewer functionality."""
    st.header("🔍 Panel Viewer")
    st.markdown("*VS Code-style proof state and compiler feedback*")

    if not st.session_state.agent.current_code:
        st.info("👋 Load a theorem from the sidebar to see the panel state.")
        return

    # Status bar
    col1, col2, col3, col4, col5 = st.columns(5)

    error_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'error')
    warning_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'warning')
    goal_count = len(st.session_state.agent.current_goals)
    clause_count = len(st.session_state.agent.editable_clauses)

    with col1:
        if error_count > 0:
            st.error(f"❌ {error_count} errors")
        else:
            st.success("✅ No errors")

    with col2:
        if warning_count > 0:
            st.warning(f"⚠️ {warning_count} warnings")
        else:
            st.info("No warnings")

    with col3:
        if goal_count > 0:
            st.info(f"🎯 {goal_count} goals")
        else:
            st.success("No goals")

    with col4:
        st.info(f"✏️ {clause_count} clauses")

    with col5:
        st.info(f"🔄 #{st.session_state.agent.compilation_id}")

    st.divider()

    # Sub-tabs for different views
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "📊 Overview",
        "💬 Messages",
        "🎯 Goals",
        "📝 Code"
    ])

    with subtab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Compilation Status")
            panel = st.session_state.agent.get_interactive_panel()
            has_errors = any('error' in msg.lower() for msg in panel['messages'])
            has_warnings = any('warning' in msg.lower() for msg in panel['messages'])

            if has_errors:
                st.error("❌ COMPILATION FAILED")
            elif has_warnings:
                st.warning("⚠️ COMPILED WITH WARNINGS")
            else:
                st.success("✅ COMPILATION SUCCESSFUL")

            st.write(f"**Compilation ID:** {panel['compilation_id']}")

            # Quick metrics
            st.subheader("📈 Metrics")
            st.metric("Errors", error_count)
            st.metric("Warnings", warning_count)
            st.metric("Goals", goal_count)
            st.metric("Editable Clauses", clause_count)

        with col2:
            st.subheader("🤖 AI Suggestions")
            suggestions = st.session_state.agent.suggest_next_actions()
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"{i}. {suggestion}")

            st.subheader("📋 Current Theorem")
            code_preview = st.session_state.agent.current_code[:200]
            if len(st.session_state.agent.current_code) > 200:
                code_preview += "..."
            st.code(code_preview, language="lean")

    with subtab2:
        st.subheader("💬 Compiler Messages")
        messages = st.session_state.agent.current_messages

        if not messages:
            st.info("No compiler messages.")
        else:
            for msg in messages:
                if msg.severity == 'error':
                    st.error(f"❌ **Error:** {msg.message}")
                elif msg.severity == 'warning':
                    st.warning(f"⚠️ **Warning:** {msg.message}")
                else:
                    st.info(f"ℹ️ **Info:** {msg.message}")

    with subtab3:
        st.subheader("🎯 Proof Goals")
        goals = st.session_state.agent.current_goals

        if not goals:
            st.success("No active goals.")
        else:
            for i, goal in enumerate(goals, 1):
                with st.expander(f"Goal {i}", expanded=True):
                    st.code(goal.goal_text, language="lean")
                    st.caption(f"Position: Line {goal.position.line + 1}, Column {goal.position.column}")

    with subtab4:
        st.subheader("📝 Annotated Code")
        lines = st.session_state.agent.current_code.split('\n')

        annotated_lines = []
        for i, line in enumerate(lines, 1):
            line_info = f"{i:3d}: {line}"

            # Check for messages at this line
            line_messages = st.session_state.agent.get_messages_at_position(i-1, 0)
            for msg in line_messages:
                icon = "❌" if msg.severity == 'error' else "⚠️" if msg.severity == 'warning' else "ℹ️"
                line_info += f"\n     {icon} {msg.message}"

            # Check for goals at this line
            goal = st.session_state.agent.get_goal_at_position(i-1, 0)
            if goal:
                line_info += f"\n     🎯 {goal.goal_text[:100]}..."

            annotated_lines.append(line_info)

        st.code('\n'.join(annotated_lines), language="lean")

def agentic_editor_tab():
    """Agentic editor functionality."""
    st.header("🤖 Agentic Editor")
    st.markdown("*Interactive editing with real-time feedback*")

    if not st.session_state.agent.current_code:
        st.info("👋 Load a theorem from the sidebar to start editing.")
        return

    # Show current theorem
    st.subheader("📝 Current Theorem")
    st.code(st.session_state.agent.current_code, language="lean")

    # All agentic editing functions in tabs
    st.subheader("🔧 Agentic Editing Tools")

    # Create tabs for all agentic functions
    agentic_tab1, agentic_tab2, agentic_tab3 = st.tabs([
        "✏️ Edit Clauses",
        "🏗️ Proof Structure",
        "📍 Position & AI Tools"
    ])

    with agentic_tab1:
        st.write("**Edit specific parts of your proof:**")

        clauses = st.session_state.agent.editable_clauses

        if not clauses:
            st.info("No editable clauses found.")
            return

        # Clause selection
        clause_ids = list(clauses.keys())
        selected_clause_id = st.selectbox(
            "Select clause to edit:",
            clause_ids,
            format_func=lambda x: f"{get_clause_icon(clauses[x].clause_type)} {x} ({clauses[x].clause_type})"
        )

        if selected_clause_id:
            clause = clauses[selected_clause_id]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Current Content:**")
                st.code(clause.content, language="lean")
                st.caption(f"Type: {clause.clause_type}")
                st.caption(f"Position: Line {clause.start_pos.line + 1}, Col {clause.start_pos.column}")

            with col2:
                st.write("**Edit Content:**")

                # Handle quick tactic selection
                tactic_key = f"selected_tactic_{selected_clause_id}"
                if tactic_key not in st.session_state:
                    st.session_state[tactic_key] = clause.content

                # Quick tactic buttons
                st.write("**Quick Tactics:**")
                tactic_col1, tactic_col2, tactic_col3 = st.columns(3)

                with tactic_col1:
                    if st.button("intro h", key="agentic_tactic_intro"):
                        st.session_state[tactic_key] = "intro h"
                        st.rerun()
                    if st.button("exact h.left", key="agentic_tactic_left"):
                        st.session_state[tactic_key] = "exact h.left"
                        st.rerun()

                with tactic_col2:
                    if st.button("exact h.right", key="agentic_tactic_right"):
                        st.session_state[tactic_key] = "exact h.right"
                        st.rerun()
                    if st.button("constructor", key="agentic_tactic_constructor"):
                        st.session_state[tactic_key] = "constructor"
                        st.rerun()

                with tactic_col3:
                    if st.button("rfl", key="agentic_tactic_rfl"):
                        st.session_state[tactic_key] = "rfl"
                        st.rerun()
                    if st.button("simp", key="agentic_tactic_simp"):
                        st.session_state[tactic_key] = "simp"
                        st.rerun()

                # Text area with current value
                new_content = st.text_area(
                    "New content:",
                    value=st.session_state[tactic_key],
                    height=100,
                    key=f"agentic_edit_{selected_clause_id}"
                )

                # Update tactic state if user manually edited
                if new_content != st.session_state[tactic_key]:
                    st.session_state[tactic_key] = new_content

                # Apply edit
                if st.button("🔄 Apply Edit", type="primary", key="agentic_apply_edit"):
                    if new_content != clause.content:
                        with st.spinner("Applying edit and recompiling..."):
                            edit_result = st.session_state.agent.edit_clause(selected_clause_id, new_content)

                            # Record in history
                            st.session_state.edit_history.append({
                                'clause_id': selected_clause_id,
                                'old_content': edit_result['old_content'],
                                'new_content': new_content,
                                'timestamp': time.time(),
                                'success': edit_result['compilation_result']['success']
                            })

                        if edit_result['compilation_result']['success']:
                            st.success("✅ Edit applied successfully!")
                        else:
                            st.error("❌ Edit applied but compilation failed")

                        st.rerun()
                    else:
                        st.info("No changes to apply.")

    with agentic_tab2:
        st.write("**➕ Add Proof Structure:**")
        st.write("Build proof scaffolding with structured elements:")

        # Structure templates
        structure_templates = {
            "Basic Have Clause": ["have h1 : P := by sorry"],
            "Conjunction Split": [
                "have h1 : P := by sorry",
                "have h2 : Q := by sorry",
                "exact ⟨h1, h2⟩"
            ],
            "Implication Chain": [
                "have step1 : P → Q := by sorry",
                "have step2 : Q → R := by sorry",
                "intro h",
                "exact step2 (step1 h)"
            ],
            "Case Analysis": [
                "cases h with",
                "| left hp => sorry",
                "| right hq => sorry"
            ]
        }

        selected_template = st.selectbox(
            "Choose structure template:",
            ["Custom"] + list(structure_templates.keys()),
            key="agentic_structure_template"
        )

        if selected_template == "Custom":
            structure_input = st.text_area(
                "Enter proof structure lines (one per line):",
                height=100,
                placeholder="have h1 : P := by sorry\nhave h2 : Q := by sorry\nexact ⟨h1, h2⟩",
                key="agentic_custom_structure"
            )
            structure_lines = [line.strip() for line in structure_input.split('\n') if line.strip()]
        else:
            structure_lines = structure_templates[selected_template]
            st.code('\n'.join(structure_lines), language="lean")

        if st.button("➕ Add Structure to Proof", type="primary", key="agentic_add_structure"):
            if structure_lines:
                with st.spinner("Adding proof structure..."):
                    try:
                        result = st.session_state.agent.add_proof_structure(structure_lines)

                        # Record in history
                        st.session_state.edit_history.append({
                            'clause_id': 'proof_structure',
                            'old_content': 'N/A',
                            'new_content': '\n'.join(structure_lines),
                            'timestamp': time.time(),
                            'success': result['compilation_result']['success'],
                            'structure_add': True
                        })

                        if result['compilation_result']['success']:
                            st.success("✅ Proof structure added successfully!")
                        else:
                            st.error("❌ Structure added but compilation failed")

                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Failed to add structure: {e}")
            else:
                st.error("Please enter some structure lines.")

    with agentic_tab3:
        # Position-aware tools
        st.write("**🎯 Position-Aware Queries:**")
        st.write("Get goals and messages at specific line positions:")

        # Line number input
        max_lines = len(st.session_state.agent.current_code.split('\n'))
        line_number = st.number_input(
            "Enter line number to check:",
            min_value=1,
            max_value=max_lines,
            value=1,
            key="agentic_line_number"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎯 Get Goal at Position", key="agentic_get_goal"):
                try:
                    goal = st.session_state.agent.get_goal_at_position(line_number - 1, 0)
                    if goal:
                        st.success(f"**Goal found at line {line_number}:**")
                        st.code(goal.goal_text, language="lean")
                        st.caption(f"Proof state ID: {goal.proof_state_id}")
                    else:
                        st.info(f"No goal found at line {line_number}")
                except Exception as e:
                    st.error(f"Error getting goal: {e}")

        with col2:
            if st.button("💬 Get Messages at Position", key="agentic_get_messages"):
                try:
                    messages = st.session_state.agent.get_messages_at_position(line_number - 1, 0)
                    if messages:
                        st.success(f"**Messages found at line {line_number}:**")
                        for msg in messages:
                            if msg.severity == 'error':
                                st.error(f"❌ {msg.message}")
                            elif msg.severity == 'warning':
                                st.warning(f"⚠️ {msg.message}")
                            else:
                                st.info(f"ℹ️ {msg.message}")
                    else:
                        st.info(f"No messages found at line {line_number}")
                except Exception as e:
                    st.error(f"Error getting messages: {e}")

        st.divider()

        # AI Suggestions
        st.write("**🤖 AI Suggestions:**")
        suggestions = st.session_state.agent.suggest_next_actions()

        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"Suggestion {i}: {suggestion}", expanded=False):
                st.write(suggestion)

                # Context-specific suggestions
                if "sorry" in suggestion.lower():
                    sorry_clauses = [cid for cid in st.session_state.agent.editable_clauses.keys() if 'sorry' in cid]
                    if sorry_clauses:
                        st.write("**Sorry clauses to work on:**")
                        for clause_id in sorry_clauses:
                            st.write(f"- {clause_id}")

        # Auto-fix suggestions
        st.divider()
        st.write("**🔧 Auto-Fix Suggestions:**")

        # Look for obvious fixes
        fixes = []
        for clause_id, clause in st.session_state.agent.editable_clauses.items():
            if clause.clause_type == 'sorry':
                if 'left' in clause_id or 'h1' in clause_id:
                    fixes.append({
                        'clause_id': clause_id,
                        'suggestion': 'intro h; exact h.left',
                        'reason': 'Likely needs left projection'
                    })
                elif 'right' in clause_id or 'h2' in clause_id:
                    fixes.append({
                        'clause_id': clause_id,
                        'suggestion': 'intro h; exact h.right',
                        'reason': 'Likely needs right projection'
                    })

        if not fixes:
            st.info("No obvious auto-fixes available.")
        else:
            st.write("**Suggested automatic fixes:**")

            for i, fix in enumerate(fixes):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**{fix['clause_id']}**")
                    st.caption(fix['reason'])

                with col2:
                    st.code(fix['suggestion'], language="lean")

                with col3:
                    if st.button("Apply", key=f"agentic_autofix_{i}"):
                        with st.spinner("Applying auto-fix..."):
                            edit_result = st.session_state.agent.edit_clause(
                                fix['clause_id'],
                                fix['suggestion']
                            )

                            st.session_state.edit_history.append({
                                'clause_id': fix['clause_id'],
                                'old_content': 'sorry',
                                'new_content': fix['suggestion'],
                                'timestamp': time.time(),
                                'success': edit_result['compilation_result']['success'],
                                'auto_fix': True
                            })

                        if edit_result['compilation_result']['success']:
                            st.success(f"✅ Auto-fix applied to {fix['clause_id']}")
                        else:
                            st.error(f"❌ Auto-fix failed for {fix['clause_id']}")

                        st.rerun()

    # Real-time compilation feedback at the bottom
    st.divider()
    st.subheader("📊 Compilation Feedback")
    messages = st.session_state.agent.current_messages

    if messages:
        for msg in messages:
            if msg.severity == 'error':
                st.error(f"❌ {msg.message}")
            elif msg.severity == 'warning':
                st.warning(f"⚠️ {msg.message}")
            else:
                st.info(f"ℹ️ {msg.message}")
    else:
        st.success("✅ No compilation messages")

def tools_tab():
    """Additional tools and utilities."""
    st.header("🔧 Tools & Analysis")

    if not st.session_state.agent.current_code:
        st.info("👋 Load a theorem to access tools.")
        return

    # Sub-tabs for tools - removed proof structure and position tools since they're in Agentic Editor
    tool_tab1, tool_tab2 = st.tabs([
        "🤖 AI Assistant",
        "📜 History"
    ])

    with tool_tab1:
        st.subheader("🤖 AI Suggestions")
        suggestions = st.session_state.agent.suggest_next_actions()

        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"Suggestion {i}: {suggestion}", expanded=False):
                st.write(suggestion)

        # Common tactics reference
        with st.expander("🔧 Common Lean 4 Tactics", expanded=False):
            st.markdown("""
            - `intro h` - Introduce hypothesis h
            - `exact e` - Provide exact proof e
            - `exact h.left` - Use left part of conjunction
            - `exact h.right` - Use right part of conjunction
            - `exact ⟨a, b⟩` - Construct pair/conjunction
            - `constructor` - Break down goal into components
            - `rw [lemma]` - Rewrite using lemma
            - `simp` - Simplify the goal
            - `trivial` - Prove trivial goals
            - `rfl` - Prove by reflexivity
            """)

    with tool_tab2:
        st.subheader("📜 Edit History")

        if not st.session_state.edit_history:
            st.info("No edits made yet.")
        else:
            for i, edit in enumerate(reversed(st.session_state.edit_history)):
                with st.expander(
                    f"Edit {len(st.session_state.edit_history) - i}: {edit['clause_id']} " +
                    ("✅" if edit['success'] else "❌"),
                    expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Before:**")
                        st.code(edit['old_content'], language="lean")

                    with col2:
                        st.write("**After:**")
                        st.code(edit['new_content'], language="lean")

                    st.caption(f"Time: {time.ctime(edit['timestamp'])}")
                    if edit.get('auto_fix'):
                        st.caption("🤖 Applied via auto-fix")
                    if edit.get('structure_add'):
                        st.caption("🏗️ Added via proof structure")

def get_clause_icon(clause_type):
    """Get appropriate icon for clause type."""
    icons = {
        'sorry': '💔',
        'have': '📝',
        'tactic': '🔧',
        'tactic_block': '🔧',
        'main_proof': '🎯',
        'proof_line': '✏️'
    }
    return icons.get(clause_type, '✏️')

def main():
    """Main Streamlit app."""
    st.title("🧮 Lean 4 Interactive Development")
    st.markdown("*VS Code-style panel viewer + LLM-style agentic editor*")

    # Load theorem sidebar
    load_theorem_sidebar()

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Panel Viewer",
        "🤖 Agentic Editor",
        "🔧 Tools"
    ])

    with tab1:
        panel_viewer_tab()

    with tab2:
        agentic_editor_tab()

    with tab3:
        tools_tab()

    # Footer
    st.divider()
    st.markdown("💡 **Tip:** Use the Panel Viewer to see VS Code-style feedback, then switch to the Agentic Editor to make targeted edits!")

if __name__ == "__main__":
    main()
