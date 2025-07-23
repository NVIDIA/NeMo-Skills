#!/usr/bin/env python3
"""
Streamlit Panel Viewer for Lean 4 Interactive Development

This provides a web-based interface to view proof state and compiler feedback
from the InteractiveLeanAgent, similar to VS Code Lean 4 extension panels.
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
    page_title="Lean 4 Panel Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = InteractiveLeanAgent(mathlib_enabled=True)
    st.session_state.viewer = LeanPanelViewer(st.session_state.agent)
    st.session_state.last_compilation_id = 0
    st.session_state.theorem_history = []

def display_status_bar():
    """Display a status bar with key metrics."""
    col1, col2, col3, col4, col5 = st.columns(5)

    # Count message types
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
            st.success("No active goals")

    with col4:
        st.info(f"✏️ {clause_count} editable clauses")

    with col5:
        compilation_id = st.session_state.agent.compilation_id
        st.info(f"🔄 Compilation #{compilation_id}")

def display_compilation_status():
    """Display detailed compilation status."""
    st.subheader("📊 Compilation Status")

    # Get panel data
    panel = st.session_state.agent.get_interactive_panel()

    # Determine overall status
    has_errors = any('error' in msg.lower() for msg in panel['messages'])
    has_warnings = any('warning' in msg.lower() for msg in panel['messages'])

    if has_errors:
        st.error("❌ COMPILATION FAILED")
    elif has_warnings:
        st.warning("⚠️ COMPILED WITH WARNINGS")
    else:
        st.success("✅ COMPILATION SUCCESSFUL")

    st.write(f"**Compilation ID:** {panel['compilation_id']}")

def display_messages():
    """Display compiler messages with proper formatting."""
    st.subheader("💬 Compiler Messages")

    messages = st.session_state.agent.current_messages

    if not messages:
        st.info("No messages.")
        return

    for i, msg in enumerate(messages):
        if msg.severity == 'error':
            st.error(f"❌ **Error:** {msg.message}")
        elif msg.severity == 'warning':
            st.warning(f"⚠️ **Warning:** {msg.message}")
        else:
            st.info(f"ℹ️ **Info:** {msg.message}")

def display_proof_goals():
    """Display current proof goals."""
    st.subheader("🎯 Proof Goals")

    goals = st.session_state.agent.current_goals

    if not goals:
        st.success("No active goals.")
        return

    for i, goal in enumerate(goals, 1):
        with st.expander(f"Goal {i}", expanded=True):
            st.code(goal.goal_text, language="lean")
            st.caption(f"Position: Line {goal.position.line + 1}, Column {goal.position.column}")

def display_editable_clauses():
    """Display editable clauses with details."""
    st.subheader("✏️ Editable Clauses")

    clauses = st.session_state.agent.editable_clauses

    if not clauses:
        st.info("No editable clauses identified.")
        return

    for clause_id, clause in clauses.items():
        with st.expander(f"{get_clause_icon(clause.clause_type)} {clause_id}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Type:** {clause.clause_type}")
                st.write(f"**Position:** Line {clause.start_pos.line + 1}, Col {clause.start_pos.column}")

            with col2:
                st.write(f"**Content:**")
                st.code(clause.content, language="lean")

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

def display_code_with_annotations():
    """Display code with line-by-line annotations."""
    st.subheader("📝 Annotated Code")

    if not st.session_state.agent.current_code:
        st.info("No code loaded.")
        return

    lines = st.session_state.agent.current_code.split('\n')

    # Create annotated code display
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

def main():
    """Main Streamlit app."""
    st.title("🔍 Lean 4 Interactive Panel Viewer")
    st.markdown("*Real-time proof state and compiler feedback (like VS Code extension)*")

    # Sidebar for controls
    with st.sidebar:
        st.header("📋 Controls")

        # Load theorem section
        st.subheader("Load Theorem")

        # Demo theorems
        demo_options = {
            "None": "",
            "Demo 1: Conjunction Commutativity": """theorem demo_and_comm (P Q : Prop) : P ∧ Q → Q ∧ P := by
  have h1 : P ∧ Q → P := by sorry
  have h2 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h2 h, h1 h⟩""",
            "Demo 2: Implication Transitivity": """theorem demo_impl_trans (P Q R : Prop) : (P → Q) → (Q → R) → (P → R) := by
  have step1 : (P → Q) → (Q → R) → P → R := by sorry
  exact step1""",
            "Demo 3: Associativity": """theorem demo_assoc (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  have h3 : P ∧ Q → P := by sorry
  have h4 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h3 (h1 h), ⟨h4 (h1 h), h2 h⟩⟩"""
        }

        selected_demo = st.selectbox("Choose demo theorem:", list(demo_options.keys()))

        if selected_demo != "None" and st.button("Load Demo"):
            theorem_code = demo_options[selected_demo]
            with st.spinner("Loading and compiling theorem..."):
                result = st.session_state.agent.load_theorem(theorem_code)
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
            st.subheader("📜 History")
            for i, entry in enumerate(reversed(st.session_state.theorem_history[-5:])):
                if st.button(f"{entry['name'][:20]}...", key=f"history_{i}"):
                    result = st.session_state.agent.load_theorem(entry['code'])
                    st.rerun()

        # Auto-refresh option
        st.subheader("🔄 Auto Refresh")
        auto_refresh = st.checkbox("Auto-refresh every 5 seconds")
        if auto_refresh:
            time.sleep(5)
            st.rerun()

    # Main content area
    if st.session_state.agent.current_code:
        # Status bar
        display_status_bar()
        st.divider()

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "💬 Messages",
            "🎯 Goals",
            "✏️ Clauses",
            "📝 Code",
            "🔧 Tools"
        ])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                display_compilation_status()

                # Quick stats
                st.subheader("📈 Quick Stats")
                error_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'error')
                warning_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'warning')
                goal_count = len(st.session_state.agent.current_goals)
                clause_count = len(st.session_state.agent.editable_clauses)

                st.metric("Errors", error_count, delta=None)
                st.metric("Warnings", warning_count, delta=None)
                st.metric("Goals", goal_count, delta=None)
                st.metric("Editable Clauses", clause_count, delta=None)

            with col2:
                # AI Suggestions
                st.subheader("🤖 AI Suggestions")
                suggestions = st.session_state.agent.suggest_next_actions()
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"{i}. {suggestion}")

                # Current theorem info
                if st.session_state.agent.current_code:
                    st.subheader("📋 Current Theorem")
                    code_preview = st.session_state.agent.current_code[:200]
                    if len(st.session_state.agent.current_code) > 200:
                        code_preview += "..."
                    st.code(code_preview, language="lean")

        with tab2:
            display_messages()

        with tab3:
            display_proof_goals()

        with tab4:
            display_editable_clauses()

        with tab5:
            display_code_with_annotations()

        with tab6:
            st.subheader("🔧 Agentic Tools")
            st.write("*Additional tools for interactive proof development*")

            # Split into columns for different tool categories
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🏗️ Proof Structure**")

                # Proof structure suggestions
                try:
                    suggestions = st.session_state.agent.get_proof_structure_suggestions()
                    with st.expander("📋 Proof Patterns", expanded=False):
                        for suggestion in suggestions:
                            if isinstance(suggestion, list):
                                st.code('\n'.join(suggestion), language="lean")
                            else:
                                st.write(suggestion)
                except AttributeError:
                    st.info("Pattern suggestions not available.")

                # Add structure interface
                st.write("**➕ Add Structure:**")
                structure_options = {
                    "Basic Have": ["have h1 : P := by sorry"],
                    "Conjunction": ["have h1 : P := by sorry", "have h2 : Q := by sorry", "exact ⟨h1, h2⟩"],
                    "Custom": []
                }

                selected_structure = st.selectbox(
                    "Choose structure:",
                    list(structure_options.keys()),
                    key="panel_structure"
                )

                if selected_structure == "Custom":
                    custom_structure = st.text_area(
                        "Enter structure lines:",
                        height=60,
                        key="panel_custom_structure"
                    )
                    structure_lines = [line.strip() for line in custom_structure.split('\n') if line.strip()]
                else:
                    structure_lines = structure_options[selected_structure]
                    if structure_lines:
                        st.code('\n'.join(structure_lines), language="lean")

                if st.button("➕ Add to Proof", key="panel_add_structure"):
                    if structure_lines:
                        with st.spinner("Adding structure..."):
                            try:
                                result = st.session_state.agent.add_proof_structure(structure_lines)
                                if result['compilation_result']['success']:
                                    st.success("✅ Structure added!")
                                    st.rerun()
                                else:
                                    st.error("❌ Structure added but compilation failed")
                            except Exception as e:
                                st.error(f"Error: {e}")

            with col2:
                st.write("**📍 Position Tools**")

                # Position-aware queries
                max_lines = len(st.session_state.agent.current_code.split('\n')) if st.session_state.agent.current_code else 1
                line_num = st.number_input(
                    "Line number:",
                    min_value=1,
                    max_value=max_lines,
                    value=1,
                    key="panel_line_num"
                )

                query_col1, query_col2 = st.columns(2)

                with query_col1:
                    if st.button("🎯 Get Goal", key="panel_get_goal"):
                        try:
                            goal = st.session_state.agent.get_goal_at_position(line_num - 1, 0)
                            if goal:
                                st.success(f"**Goal at line {line_num}:**")
                                st.code(goal.goal_text, language="lean")
                            else:
                                st.info(f"No goal at line {line_num}")
                        except Exception as e:
                            st.error(f"Error: {e}")

                with query_col2:
                    if st.button("💬 Get Messages", key="panel_get_messages"):
                        try:
                            messages = st.session_state.agent.get_messages_at_position(line_num - 1, 0)
                            if messages:
                                st.success(f"**Messages at line {line_num}:**")
                                for msg in messages:
                                    if msg.severity == 'error':
                                        st.error(f"❌ {msg.message}")
                                    elif msg.severity == 'warning':
                                        st.warning(f"⚠️ {msg.message}")
                                    else:
                                        st.info(f"ℹ️ {msg.message}")
                            else:
                                st.info(f"No messages at line {line_num}")
                        except Exception as e:
                            st.error(f"Error: {e}")

                # AI suggestions (enhanced)
                st.write("**🤖 AI Suggestions:**")
                suggestions = st.session_state.agent.suggest_next_actions()
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"{i}. {suggestion}")

    else:
        # Welcome screen
        st.info("👋 Welcome! Please load a theorem using the sidebar to get started.")

        st.markdown("""
        ### 🎯 How to use this panel:

        1. **Load a theorem** from the sidebar (demo or custom)
        2. **View the panel state** in real-time as you would in VS Code
        3. **Check different tabs** for detailed information:
           - **Overview**: General status and AI suggestions
           - **Messages**: Compiler errors, warnings, and info
           - **Goals**: Current proof goals that need to be solved
           - **Clauses**: Editable sections you can modify
           - **Code**: Your theorem with line-by-line annotations

        ### 🔄 VS Code Comparison:
        This panel shows the same information you'd see in VS Code's Lean 4 extension:
        - Compilation status and error messages
        - Proof goals at cursor positions
        - Interactive feedback as you edit
        """)

if __name__ == "__main__":
    main()
