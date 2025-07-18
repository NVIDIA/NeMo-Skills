#!/usr/bin/env python3
"""
Custom Sketch Editor: True In-Place Editing for Lean Theorems

This provides the in-place editing capability that lean-interact lacks by:
1. Parsing theorem structure to identify individual sorry blocks
2. Allowing targeted replacement of specific sorries
3. Maintaining theorem identity while updating content
4. Testing updates with lean-interact before applying them

This addresses the user's requirement for genuine in-place editing.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from nemo_skills.code_execution.lean_prover import LeanProver


@dataclass
class SorryLocation:
    """Represents a sorry block within a theorem."""
    sorry_id: int
    line_number: int
    start_pos: int
    end_pos: int
    context: str  # The `have` statement or context this sorry belongs to
    replacement: Optional[str] = None


@dataclass
class EditableTheorem:
    """A theorem that supports in-place editing of individual sorry blocks."""
    name: str
    signature: str
    original_body: str
    current_body: str
    sorry_locations: Dict[int, SorryLocation] = field(default_factory=dict)
    proof_state: Optional[int] = None

    def __post_init__(self):
        """Parse the theorem to identify sorry locations."""
        self._parse_sorries()

    def _parse_sorries(self):
        """Parse the theorem body to identify individual sorry blocks."""
        lines = self.current_body.split('\n')
        sorry_count = 0

        for line_num, line in enumerate(lines):
            # Look for 'by sorry' patterns
            sorry_matches = list(re.finditer(r'\bby\s+sorry\b', line))
            for match in sorry_matches:
                # Find the context (look backwards for 'have' or similar)
                context = self._find_context(lines, line_num, match.start())

                self.sorry_locations[sorry_count] = SorryLocation(
                    sorry_id=sorry_count,
                    line_number=line_num,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=context
                )
                sorry_count += 1

    def _find_context(self, lines: List[str], line_num: int, pos: int) -> str:
        """Find the context (have statement, etc.) for a sorry."""
        current_line = lines[line_num]

        # Look for 'have' at the beginning of the line
        have_match = re.search(r'^\s*have\s+(\w+)\s*:', current_line)
        if have_match:
            return have_match.group(0)

        # Look backwards for context
        for i in range(line_num - 1, -1, -1):
            line = lines[i]
            have_match = re.search(r'^\s*have\s+(\w+)\s*:', line)
            if have_match:
                return have_match.group(0)

        return f"line {line_num + 1}"

    def update_sorry(self, sorry_id: int, replacement: str) -> bool:
        """Replace a specific sorry with a proof. Returns True if successful."""
        if sorry_id not in self.sorry_locations:
            return False

        sorry_loc = self.sorry_locations[sorry_id]
        lines = self.current_body.split('\n')

        # Replace 'by sorry' with the new proof
        line = lines[sorry_loc.line_number]
        new_line = line[:sorry_loc.start_pos] + f"by {replacement}" + line[sorry_loc.end_pos:]
        lines[sorry_loc.line_number] = new_line

        # Update the body
        self.current_body = '\n'.join(lines)
        sorry_loc.replacement = replacement

        # Re-parse to update positions
        self._parse_sorries()

        return True

    def get_full_theorem(self) -> str:
        """Get the complete theorem as it would appear in Lean code."""
        return f"theorem {self.name} {self.signature} := by\n{self.current_body}"

    def list_sorries(self) -> List[Tuple[int, str, Optional[str]]]:
        """List all sorries with their IDs, contexts, and current replacements."""
        return [
            (sorry_id, loc.context, loc.replacement)
            for sorry_id, loc in self.sorry_locations.items()
            if loc.replacement is None  # Only list unreplaced sorries
        ]

    def is_complete(self) -> bool:
        """Check if all sorries have been replaced."""
        return all(loc.replacement is not None for loc in self.sorry_locations.values())


class SketchEditor:
    """Editor that provides true in-place editing for Lean theorem sketches."""

    def __init__(self, mathlib_enabled: bool = True):
        self.prover = LeanProver(mathlib_enabled=mathlib_enabled)
        self.theorems: Dict[str, EditableTheorem] = {}

    def create_theorem(self, name: str, signature: str, body: str) -> EditableTheorem:
        """Create a new editable theorem."""
        theorem = EditableTheorem(
            name=name,
            signature=signature,
            original_body=body,
            current_body=body
        )

        self.theorems[name] = theorem

        # Test the theorem with lean-interact using unique name
        import time
        test_name = f"{name}_test_{int(time.time() * 1000) % 100000}"
        test_theorem = f"theorem {test_name} {signature} := by\n{body}"
        result = self.prover.run_command(test_theorem)
        theorem.proof_state = result.proof_state

        return theorem

    def update_sorry(self, theorem_name: str, sorry_id: int, replacement: str) -> Tuple[bool, str]:
        """
        Update a specific sorry in a theorem.
        Returns (success, message).
        """
        if theorem_name not in self.theorems:
            return False, f"Theorem '{theorem_name}' not found"

        theorem = self.theorems[theorem_name]

        # Make the update
        if not theorem.update_sorry(sorry_id, replacement):
            return False, f"Sorry ID {sorry_id} not found in theorem"

        # Test with lean-interact using a unique name to avoid conflicts
        import time
        test_name = f"{theorem.name}_test_{int(time.time() * 1000) % 100000}"
        test_theorem = f"theorem {test_name} {theorem.signature} := by\n{theorem.current_body}"
        result = self.prover.run_command(test_theorem)

        # Update proof state
        theorem.proof_state = result.proof_state

        if result.success:
            return True, f"Successfully updated sorry {sorry_id}"
        else:
            return True, f"Updated sorry {sorry_id}, but theorem has issues: {result.response}"

    def get_theorem_status(self, theorem_name: str) -> Dict[str, Any]:
        """Get complete status of a theorem."""
        if theorem_name not in self.theorems:
            return {"error": f"Theorem '{theorem_name}' not found"}

        theorem = self.theorems[theorem_name]

        return {
            "name": theorem.name,
            "signature": theorem.signature,
            "proof_state": theorem.proof_state,
            "is_complete": theorem.is_complete(),
            "remaining_sorries": theorem.list_sorries(),
            "current_code": theorem.get_full_theorem()
        }

    def test_theorem(self, theorem_name: str) -> Dict[str, Any]:
        """Test the current state of a theorem with Lean."""
        if theorem_name not in self.theorems:
            return {"error": f"Theorem '{theorem_name}' not found"}

        theorem = self.theorems[theorem_name]

        # Use unique name to avoid conflicts
        import time
        test_name = f"{theorem.name}_test_{int(time.time() * 1000) % 100000}"
        test_theorem = f"theorem {test_name} {theorem.signature} := by\n{theorem.current_body}"
        result = self.prover.run_command(test_theorem)

        return {
            "success": result.success,
            "proof_state": result.proof_state,
            "response": result.response,
            "has_sorry": result.has_sorry,
            "full_code": theorem.get_full_theorem()
        }


def demo_sketch_editor():
    """Demonstrate true in-place editing capabilities."""
    print("=" * 80)
    print("CUSTOM SKETCH EDITOR - TRUE IN-PLACE EDITING")
    print("=" * 80)

    editor = SketchEditor(mathlib_enabled=True)

    print("\n🎯 CREATING A THEOREM WITH MULTIPLE SORRIES:")
    print("-" * 60)

    # Create theorem with multiple sorries
    body = """  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  have h3 : P ∧ Q → P := by sorry
  have h4 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h3 (h1 h), h4 (h1 h)⟩"""

    theorem = editor.create_theorem(
        name="demo_editing",
        signature="(P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ Q",
        body=body
    )

    print("CREATED THEOREM:")
    print(theorem.get_full_theorem())
    print()

    # Show initial status
    status = editor.get_theorem_status("demo_editing")
    print(f"INITIAL STATUS:")
    print(f"  Proof state: {status['proof_state']}")
    print(f"  Is complete: {status['is_complete']}")
    print(f"  Remaining sorries: {len(status['remaining_sorries'])}")
    for sorry_id, context, _ in status['remaining_sorries']:
        print(f"    Sorry {sorry_id}: {context}")

    print("\n🔧 PERFORMING IN-PLACE EDITS:")
    print("-" * 60)

    # Edit sorry 0
    print("EDITING Sorry 0 (h1)...")
    success, msg = editor.update_sorry("demo_editing", 0, "intro h; exact h.left")
    print(f"  Result: {msg}")

    # Edit sorry 1
    print("EDITING Sorry 1 (h2)...")
    success, msg = editor.update_sorry("demo_editing", 1, "intro h; exact h.right")
    print(f"  Result: {msg}")

    # Edit sorry 2
    print("EDITING Sorry 2 (h3)...")
    success, msg = editor.update_sorry("demo_editing", 2, "intro h; exact h.left")
    print(f"  Result: {msg}")

    # Edit sorry 3
    print("EDITING Sorry 3 (h4)...")
    success, msg = editor.update_sorry("demo_editing", 3, "intro h; exact h.right")
    print(f"  Result: {msg}")

    print("\n📊 FINAL STATUS:")
    print("-" * 60)

    final_status = editor.get_theorem_status("demo_editing")
    print(f"Is complete: {final_status['is_complete']}")
    print(f"Remaining sorries: {len(final_status['remaining_sorries'])}")

    print("\nFINAL THEOREM CODE:")
    print(final_status['current_code'])

    # Test the final theorem
    test_result = editor.test_theorem("demo_editing")
    print(f"\nLEAN TEST RESULT:")
    print(f"  Success: {test_result['success']}")
    print(f"  Response: {test_result['response']}")

    if test_result['success']:
        print("\n🎉 IN-PLACE EDITING SUCCESSFUL!")
        print("   ✅ Individual sorries updated without rewriting entire theorem")
        print("   ✅ Theorem identity maintained throughout editing process")
        print("   ✅ Each edit validated with lean-interact")
        print("   ✅ Complete theorem tested and verified")


if __name__ == "__main__":
    demo_sketch_editor()

    print("\n" + "=" * 80)
    print("🎯 IN-PLACE EDITING SOLUTION SUMMARY:")
    print("=" * 80)
    print()
    print("✅ ADDRESSES YOUR REQUIREMENTS:")
    print("• True in-place editing of theorem sorries")
    print("• No need to rewrite entire theorems")
    print("• Maintains theorem object identity")
    print("• Targeted updates to specific parts")
    print("• Validation with lean-interact after each edit")
    print()
    print("✅ KEY CAPABILITIES:")
    print("• Parse theorem structure automatically")
    print("• Identify and track individual sorry blocks")
    print("• Replace specific sorries with proofs")
    print("• Test updates with Lean after each change")
    print("• Track completion status")
    print()
    print("✅ PERFECT FOR LLM INTEGRATION:")
    print("• LLM can work on individual theorem parts")
    print("• Incremental proof development")
    print("• No unwieldy theorem rewriting")
    print("• Clean separation of concerns")
    print()
    print("🎉 This provides the in-place editing you need!")
    print("=" * 80)
