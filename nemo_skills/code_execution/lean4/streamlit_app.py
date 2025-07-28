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
    """Panel viewer with real-time feedback - VS Code style layout."""
    st.header("🔍 Panel Viewer")
    st.markdown("*VS Code-style interface with real-time feedback*")

    if not st.session_state.agent.current_code:
        st.info("👋 Load a theorem from the sidebar to start viewing.")
        return

    # Instruction for the split layout
    st.info("💡 **Layout:** Edit on the LEFT, see live results on the RIGHT (like VS Code)")
    st.markdown("---")

    # Two-column layout: Editor on left, Info panel on right
    left_col, right_col = st.columns([0.6, 0.4], gap="large")

    with left_col:
        st.subheader("🤖 Agentic Editor")
        st.markdown("*Make edits and see results immediately →*")

        # All agentic editing functions in sub-tabs within the left column
        edit_tab1, edit_tab2, edit_tab3 = st.tabs([
            "✏️ Edit", "🏗️ Structure", "📍 Position & AI"
        ])

        with edit_tab1:
            st.write("**Edit specific clauses:**")

            clauses = st.session_state.agent.editable_clauses

            if not clauses:
                st.info("No editable clauses found.")
            else:
                # Clause selection
                clause_ids = list(clauses.keys())
                selected_clause_id = st.selectbox(
                    "Select clause:",
                    clause_ids,
                    format_func=lambda x: f"{get_clause_icon(clauses[x].clause_type)} {x}",
                    key="panel_clause_select"
                )

                if selected_clause_id:
                    clause = clauses[selected_clause_id]

                    st.write("**Current:**")
                    st.code(clause.content, language="lean")
                    st.caption(f"Type: {clause.clause_type} | Line {clause.start_pos.line + 1}")

                    # Quick tactics
                    st.write("**Quick Tactics:**")
                    tactic_col1, tactic_col2 = st.columns(2)

                    # Handle quick tactic selection
                    tactic_key = f"panel_selected_tactic_{selected_clause_id}"
                    if tactic_key not in st.session_state:
                        st.session_state[tactic_key] = clause.content

                    with tactic_col1:
                        if st.button("intro h", key="panel_tactic_intro"):
                            st.session_state[tactic_key] = "intro h"
                            st.rerun()
                        if st.button("exact h.left", key="panel_tactic_left"):
                            st.session_state[tactic_key] = "exact h.left"
                            st.rerun()
                        if st.button("rfl", key="panel_tactic_rfl"):
                            st.session_state[tactic_key] = "rfl"
                            st.rerun()

                    with tactic_col2:
                        if st.button("exact h.right", key="panel_tactic_right"):
                            st.session_state[tactic_key] = "exact h.right"
                            st.rerun()
                        if st.button("constructor", key="panel_tactic_constructor"):
                            st.session_state[tactic_key] = "constructor"
                            st.rerun()
                        if st.button("simp", key="panel_tactic_simp"):
                            st.session_state[tactic_key] = "simp"
                            st.rerun()

                    # Text area with current value
                    new_content = st.text_area(
                        "New content:",
                        value=st.session_state[tactic_key],
                        height=80,
                        key=f"panel_edit_{selected_clause_id}"
                    )

                    # Update tactic state if user manually edited
                    if new_content != st.session_state[tactic_key]:
                        st.session_state[tactic_key] = new_content

                    # Apply edit
                    if st.button("🔄 Apply Edit", type="primary", key="panel_apply_edit"):
                        if new_content != clause.content:
                            with st.spinner("Applying edit..."):
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
                                st.success("✅ Edit applied!")
                            else:
                                st.error("❌ Edit failed")

                            st.rerun()
                        else:
                            st.info("No changes to apply.")

        with edit_tab2:
            st.write("**Add Proof Structure:**")

            # Simplified structure templates for the panel
            structure_templates = {
                "Basic Have": ["have h1 : P := by sorry"],
                "Conjunction": [
                    "have h1 : P := by sorry",
                    "have h2 : Q := by sorry",
                    "exact ⟨h1, h2⟩"
                ],
                "Implication": [
                    "have step1 : P → Q := by sorry",
                    "intro h",
                    "exact step1 h"
                ]
            }

            selected_template = st.selectbox(
                "Template:",
                ["Custom"] + list(structure_templates.keys()),
                key="panel_structure_template"
            )

            if selected_template == "Custom":
                structure_input = st.text_area(
                    "Structure lines:",
                    height=80,
                    placeholder="have h1 : P := by sorry",
                    key="panel_custom_structure"
                )
                structure_lines = [line.strip() for line in structure_input.split('\n') if line.strip()]
            else:
                structure_lines = structure_templates[selected_template]
                st.code('\n'.join(structure_lines), language="lean")

            if st.button("➕ Add Structure", key="panel_add_structure"):
                if structure_lines:
                    with st.spinner("Adding structure..."):
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
                                st.success("✅ Structure added!")
                            else:
                                st.error("❌ Structure failed")

                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Failed: {e}")
                else:
                    st.error("Enter structure lines.")

        with edit_tab3:
            st.write("**Position Tools:**")

            max_lines = len(st.session_state.agent.current_code.split('\n'))
            line_number = st.number_input(
                "Line:",
                min_value=1,
                max_value=max_lines,
                value=1,
                key="panel_position_line"
            )

            pos_col1, pos_col2 = st.columns(2)

            with pos_col1:
                if st.button("🎯 Goal", key="panel_get_goal"):
                    try:
                        goal = st.session_state.agent.get_goal_at_position(line_number - 1, 0)
                        if goal:
                            st.success(f"**Goal at line {line_number}:**")
                            st.code(goal.goal_text, language="lean")
                        else:
                            st.info(f"No goal at line {line_number}")
                    except Exception as e:
                        st.error(f"Error: {e}")

            with pos_col2:
                if st.button("💬 Messages", key="panel_get_messages"):
                    try:
                        messages = st.session_state.agent.get_messages_at_position(line_number - 1, 0)
                        if messages:
                            st.success(f"**Messages at line {line_number}:**")
                            for msg in messages:
                                if msg.severity == 'error':
                                    st.error(f"❌ {msg.message}")
                                elif msg.severity == 'warning':
                                    st.warning(f"⚠️ {msg.message}")
                                else:
                                    st.info(f"ℹ️ {msg.message}")
                        else:
                            st.info(f"No messages at line {line_number}")
                    except Exception as e:
                        st.error(f"Error: {e}")

            st.write("**AI Suggestions:**")
            suggestions = st.session_state.agent.suggest_next_actions()

            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"{i}. {suggestion}")

    with right_col:
        st.subheader("�� Live Info Panel")
        st.markdown("*← Updates automatically after each edit*")

        # Add a visual separator
        st.markdown("---")

        # Compilation Status at the top
        st.write("**🔄 Compilation Status**")
        messages = st.session_state.agent.current_messages

        if messages:
            error_count = sum(1 for msg in messages if msg.severity == 'error')
            warning_count = sum(1 for msg in messages if msg.severity == 'warning')
            info_count = sum(1 for msg in messages if msg.severity == 'info')

            status_col1, status_col2, status_col3 = st.columns(3)
            with status_col1:
                if error_count > 0:
                    st.error(f"❌ {error_count} errors")
                else:
                    st.success("✅ No errors")

            with status_col2:
                if warning_count > 0:
                    st.warning(f"⚠️ {warning_count} warnings")
                else:
                    st.info("ℹ️ No warnings")

            with status_col3:
                st.info(f"📝 {info_count} messages")
        else:
            st.success("✅ Clean compilation")

        st.divider()

        # Current Theorem
        st.write("**📝 Current Theorem**")
        with st.expander("View Code", expanded=True):
            st.code(st.session_state.agent.current_code, language="lean")

        st.divider()

        # Messages
        st.write("**💬 Messages**")
        if messages:
            for i, msg in enumerate(messages):
                with st.expander(f"{msg.severity.title()} {i+1}", expanded=False):
                    if msg.severity == 'error':
                        st.error(f"❌ {msg.message}")
                    elif msg.severity == 'warning':
                        st.warning(f"⚠️ {msg.message}")
                    else:
                        st.info(f"ℹ️ {msg.message}")
                    st.caption(f"Position: Line {msg.start_pos.line + 1}, Col {msg.start_pos.column}")
        else:
            st.success("✅ No messages")

        st.divider()

        # Goals
        st.write("**🎯 Proof Goals**")
        goals = st.session_state.agent.current_goals

        if goals:
            for i, goal in enumerate(goals):
                with st.expander(f"Goal {i+1}", expanded=False):
                    st.code(goal.goal_text, language="lean")
                    st.caption(f"Position: Line {goal.position.line + 1}, Col {goal.position.column}")
                    if goal.proof_state_id:
                        st.caption(f"Proof State ID: {goal.proof_state_id}")
        else:
            st.info("No active goals")

        st.divider()

        # Editable Clauses Summary
        st.write("**✏️ Editable Clauses**")
        clauses = st.session_state.agent.editable_clauses

        if clauses:
            clause_summary = {}
            for cid, clause in clauses.items():
                clause_type = clause.clause_type
                if clause_type not in clause_summary:
                    clause_summary[clause_type] = 0
                clause_summary[clause_type] += 1

            summary_text = ", ".join([f"{count} {ctype}" for ctype, count in clause_summary.items()])
            st.info(f"📊 Found: {summary_text}")

            with st.expander("Clause Details", expanded=False):
                for cid, clause in clauses.items():
                    st.write(f"**{cid}** ({clause.clause_type})")
                    st.code(clause.content[:50] + ("..." if len(clause.content) > 50 else ""), language="lean")
        else:
            st.info("No editable clauses found")

        st.divider()

        # Metrics
        st.write("**📈 Metrics**")
        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            st.metric("Compilation ID", st.session_state.agent.compilation_id)
            st.metric("Code Lines", len(st.session_state.agent.current_code.split('\n')))

        with metrics_col2:
            st.metric("Editable Clauses", len(clauses))
            st.metric("Edit History", len(st.session_state.edit_history))

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
        "🔍 Panel Viewer (Editor + Info)",
        "🤖 Agentic Editor (Full)",
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
