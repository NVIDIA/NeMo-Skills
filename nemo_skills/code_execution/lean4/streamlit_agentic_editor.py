#!/usr/bin/env python3
"""
Streamlit Agentic Editor for Lean 4 Interactive Development

This provides a web-based interface for interactive editing of Lean 4 proofs,
simulating how an LLM agent would work with the Interactive Lean Agent.
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
    page_title="Lean 4 Agentic Editor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = InteractiveLeanAgent(mathlib_enabled=True)
    st.session_state.viewer = LeanPanelViewer(st.session_state.agent)
    st.session_state.edit_history = []
    st.session_state.current_theorem_name = None

def display_edit_interface():
    """Display the main editing interface."""
    st.subheader("✏️ Interactive Editing")

    clauses = st.session_state.agent.editable_clauses

    if not clauses:
        st.info("No editable clauses found. Load a theorem first.")
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
                if st.button("intro h", key="tactic_intro"):
                    st.session_state[tactic_key] = "intro h"
                    st.rerun()
                if st.button("exact h.left", key="tactic_left"):
                    st.session_state[tactic_key] = "exact h.left"
                    st.rerun()

            with tactic_col2:
                if st.button("exact h.right", key="tactic_right"):
                    st.session_state[tactic_key] = "exact h.right"
                    st.rerun()
                if st.button("constructor", key="tactic_constructor"):
                    st.session_state[tactic_key] = "constructor"
                    st.rerun()

            with tactic_col3:
                if st.button("rfl", key="tactic_rfl"):
                    st.session_state[tactic_key] = "rfl"
                    st.rerun()
                if st.button("simp", key="tactic_simp"):
                    st.session_state[tactic_key] = "simp"
                    st.rerun()

                        # Text area with current value
            new_content = st.text_area(
                "New content:",
                value=st.session_state[tactic_key],
                height=100,
                key=f"edit_{selected_clause_id}"
            )

            # Update tactic state if user manually edited
            if new_content != st.session_state[tactic_key]:
                st.session_state[tactic_key] = new_content

            # Apply edit button
            if st.button("🔄 Apply Edit", type="primary"):
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

                    # Show result
                    if edit_result['compilation_result']['success']:
                        st.success("✅ Edit applied successfully!")
                    else:
                        st.error("❌ Edit applied but compilation failed")

                    st.rerun()
                else:
                    st.info("No changes to apply.")

def display_ai_suggestions():
    """Display AI suggestions for next actions."""
    st.subheader("🤖 AI Suggestions")

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

def display_auto_fix():
    """Display auto-fix suggestions and apply them."""
    st.subheader("🔧 Auto-Fix Suggestions")

    # Look for obvious fixes
    fixes = []

    for clause_id, clause in st.session_state.agent.editable_clauses.items():
        if clause.clause_type == 'sorry':
            # Suggest based on context
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
        return

    st.write("**Suggested automatic fixes:**")

    for i, fix in enumerate(fixes):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.write(f"**{fix['clause_id']}**")
            st.caption(fix['reason'])

        with col2:
            st.code(fix['suggestion'], language="lean")

        with col3:
            if st.button("Apply", key=f"autofix_{i}"):
                with st.spinner("Applying auto-fix..."):
                    edit_result = st.session_state.agent.edit_clause(
                        fix['clause_id'],
                        fix['suggestion']
                    )

                    # Record in history
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

def display_edit_history():
    """Display the edit history."""
    st.subheader("📜 Edit History")

    if not st.session_state.edit_history:
        st.info("No edits made yet.")
        return

    # Reverse chronological order
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

def display_proof_analysis():
    """Display detailed proof analysis."""
    st.subheader("🔍 Proof Analysis")

    # Basic statistics
    error_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'error')
    warning_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'warning')
    sorry_count = len([c for c in st.session_state.agent.editable_clauses.values() if c.clause_type == 'sorry'])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Errors", error_count)
    with col2:
        st.metric("Warnings", warning_count)
    with col3:
        st.metric("Sorry Clauses", sorry_count)
    with col4:
        st.metric("Total Clauses", len(st.session_state.agent.editable_clauses))

    # Completion assessment
    if error_count == 0 and sorry_count == 0:
        st.success("🎉 Proof appears complete!")
    elif error_count == 0:
        st.warning(f"⚠️ No errors, but {sorry_count} sorry clauses remain.")
    else:
        st.error(f"❌ {error_count} errors need to be fixed first.")

    # Proof structure analysis
    have_clauses = [c for c in st.session_state.agent.editable_clauses.values() if c.clause_type == 'have']
    tactic_clauses = [c for c in st.session_state.agent.editable_clauses.values() if c.clause_type in ['tactic', 'tactic_block']]

    st.write("**Proof Structure:**")
    st.write(f"- Have statements: {len(have_clauses)}")
    st.write(f"- Tactic blocks: {len(tactic_clauses)}")

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
    st.title("🤖 Lean 4 Agentic Editor")
    st.markdown("*Interactive editing with real-time feedback (LLM-style workflow)*")

    # Sidebar for theorem loading
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
            "Nested Structure": """theorem nested_proof (P Q R S : Prop) : (P ∧ Q) ∧ (R ∧ S) → (P ∧ R) ∧ (Q ∧ S) := by
  have h1 : (P ∧ Q) ∧ (R ∧ S) → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ (R ∧ S) → R ∧ S := by sorry
  have h3 : P ∧ Q → P := by sorry
  have h4 : P ∧ Q → Q := by sorry
  have h5 : R ∧ S → R := by sorry
  have h6 : R ∧ S → S := by sorry
  intro h
  exact ⟨⟨h3 (h1 h), h5 (h2 h)⟩, ⟨h4 (h1 h), h6 (h2 h)⟩⟩"""
        }

        selected_demo = st.selectbox("Choose demo theorem:", list(demo_options.keys()))

        if selected_demo != "None" and st.button("Load Demo"):
            theorem_code = demo_options[selected_demo]
            with st.spinner("Loading theorem..."):
                result = st.session_state.agent.load_theorem(theorem_code)
                st.session_state.current_theorem_name = selected_demo
                st.session_state.edit_history = []  # Reset history
            st.success(f"Loaded: {selected_demo}")
            st.rerun()

        # Custom theorem
        st.subheader("Custom Theorem")
        custom_theorem = st.text_area(
            "Enter your theorem:",
            height=150,
            placeholder="theorem my_theorem (P Q : Prop) : P ∧ Q → Q ∧ P := by\n  sorry"
        )

        if st.button("Load Custom"):
            if custom_theorem.strip():
                with st.spinner("Loading custom theorem..."):
                    result = st.session_state.agent.load_theorem(custom_theorem)
                    st.session_state.current_theorem_name = "Custom"
                    st.session_state.edit_history = []  # Reset history
                st.success("Custom theorem loaded!")
                st.rerun()

        # Current theorem info
        if st.session_state.agent.current_code:
            st.subheader("📊 Current Status")
            error_count = sum(1 for msg in st.session_state.agent.current_messages if msg.severity == 'error')
            sorry_count = len([c for c in st.session_state.agent.editable_clauses.values() if c.clause_type == 'sorry'])

            if error_count == 0 and sorry_count == 0:
                st.success("✅ Complete!")
            elif error_count == 0:
                st.warning(f"⚠️ {sorry_count} sorries left")
            else:
                st.error(f"❌ {error_count} errors")

    # Main content
    if st.session_state.agent.current_code:
        # Create tabs for different editing views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "✏️ Edit",
            "🤖 AI Assist",
            "🔧 Auto-Fix",
            "📜 History",
            "🔍 Analysis"
        ])

        with tab1:
            # Show current code
            st.subheader("📝 Current Theorem")
            st.code(st.session_state.agent.current_code, language="lean")

            # Edit interface
            display_edit_interface()

            # Real-time compilation feedback
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

        with tab2:
            display_ai_suggestions()

        with tab3:
            display_auto_fix()

        with tab4:
            display_edit_history()

        with tab5:
            display_proof_analysis()

            # Show editable clauses
            st.subheader("✏️ Editable Clauses Overview")
            clauses = st.session_state.agent.editable_clauses

            if clauses:
                for clause_id, clause in clauses.items():
                    col1, col2, col3 = st.columns([2, 1, 2])

                    with col1:
                        st.write(f"{get_clause_icon(clause.clause_type)} **{clause_id}**")

                    with col2:
                        st.caption(clause.clause_type)

                    with col3:
                        st.code(clause.content[:50] + ("..." if len(clause.content) > 50 else ""), language="lean")

    else:
        # Welcome screen
        st.info("👋 Welcome! Load a theorem from the sidebar to start editing.")

        st.markdown("""
        ### 🎯 How to use the Agentic Editor:

        1. **Load a theorem** from the sidebar (demo or custom)
        2. **Use the Edit tab** to make targeted changes to specific clauses
        3. **Get AI suggestions** for next steps and common tactics
        4. **Try auto-fixes** for obvious improvements
        5. **Monitor your progress** via the analysis tab

        ### 🤖 LLM-Style Workflow:
        This editor simulates how an LLM agent would work:
        - **Targeted edits** to specific clauses rather than whole-file changes
        - **Real-time feedback** after each edit
        - **AI suggestions** for next actions
        - **Auto-fix capabilities** for common patterns
        - **Edit history** tracking for learning and debugging

        ### 🔄 Perfect for:
        - **Learning Lean 4** with guided assistance
        - **Debugging proofs** with targeted edits
        - **Testing agent strategies** for automated proving
        - **Comparing with VS Code** to verify consistency
        """)

if __name__ == "__main__":
    main()
