"""
Example client for the Lean Proof API
Demonstrates how AI agents can interact with the FastAPI wrapper
"""

import requests
import json
from typing import Dict, Any

class LeanProofApiClient:
    """Client class for interacting with the Lean Proof API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def create_session(self, theorem_statement: str, repo_path: str = None) -> Dict[str, Any]:
        """Create a new proof session"""
        payload = {"theorem_statement": theorem_statement}
        if repo_path:
            payload["repo_path"] = repo_path

        response = self.session.post(f"{self.base_url}/sessions", json=payload)
        response.raise_for_status()
        return response.json()

    def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions"""
        response = self.session.get(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()

    def get_proof_state(self, session_id: str, branch_id: str = None) -> Dict[str, Any]:
        """Get current proof state"""
        params = {"branch_id": branch_id} if branch_id else {}
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/state", params=params)
        response.raise_for_status()
        return response.json()

    def apply_tactic(self, session_id: str, tactic: str, branch_id: str = None) -> Dict[str, Any]:
        """Apply a tactic to the proof"""
        payload = {"tactic": tactic}
        if branch_id:
            payload["branch_id"] = branch_id

        response = self.session.post(f"{self.base_url}/sessions/{session_id}/apply-tactic", json=payload)
        response.raise_for_status()
        return response.json()

    def undo_step(self, session_id: str, step_id: str, branch_id: str = None) -> Dict[str, Any]:
        """Undo a specific step"""
        payload = {"step_id": step_id}
        if branch_id:
            payload["branch_id"] = branch_id

        response = self.session.post(f"{self.base_url}/sessions/{session_id}/undo-step", json=payload)
        response.raise_for_status()
        return response.json()

    def create_branch(self, session_id: str, branch_name: str, from_step: str = None) -> Dict[str, Any]:
        """Create a new proof branch"""
        payload = {"branch_name": branch_name}
        if from_step:
            payload["from_step"] = from_step

        response = self.session.post(f"{self.base_url}/sessions/{session_id}/create-branch", json=payload)
        response.raise_for_status()
        return response.json()

    def switch_branch(self, session_id: str, branch_id: str) -> Dict[str, Any]:
        """Switch to a different branch"""
        payload = {"branch_id": branch_id}
        response = self.session.post(f"{self.base_url}/sessions/{session_id}/switch-branch", json=payload)
        response.raise_for_status()
        return response.json()

    def get_proof_script(self, session_id: str, branch_id: str = None) -> Dict[str, Any]:
        """Get the generated proof script"""
        params = {"branch_id": branch_id} if branch_id else {}
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/proof-script", params=params)
        response.raise_for_status()
        return response.json()

    def analyze_proof_attempts(self, session_id: str) -> Dict[str, Any]:
        """Analyze all proof attempts"""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/analysis")
        response.raise_for_status()
        return response.json()

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export session data"""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/export")
        response.raise_for_status()
        return response.json()

    def list_branches(self, session_id: str) -> Dict[str, Any]:
        """List all branches in a session"""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/branches")
        response.raise_for_status()
        return response.json()

    def get_branch_steps(self, session_id: str, branch_id: str) -> Dict[str, Any]:
        """Get all steps in a specific branch"""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/branches/{branch_id}/steps")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

def example_usage():
    """Example demonstrating how to use the API client"""

    # Initialize client
    client = LeanProofApiClient()

    # Check if API is healthy
    print("=== Health Check ===")
    health = client.health_check()
    print(f"API Health: {health}")

    # Create a new proof session
    print("\n=== Creating Proof Session ===")
    theorem = "theorem amc12a_2009_p5 (x : ℝ) (h₀ : x ^ 3 - (x + 1) * (x - 1) * x = 5) : x ^ 3 = 125"
    session_response = client.create_session(theorem)
    session_id = session_response["session_id"]
    print(f"Created session: {session_id}")
    print(f"Initial goals: {session_response['current_goals']}")

    # Get initial proof state
    print("\n=== Initial Proof State ===")
    state = client.get_proof_state(session_id)
    print(f"Branch: {state['branch_id']}")
    print(f"Status: {state['status']}")
    print(f"Goals: {state['current_goals']}")

    # Apply first tactic
    print("\n=== Applying First Tactic ===")
    tactic1 = "have h1 : (x + 1) * (x - 1) * x = x ^ 3 - x := by ring"
    result1 = client.apply_tactic(session_id, tactic1)
    print(f"Tactic: {tactic1}")
    print(f"Success: {result1['success']}")
    print(f"New goals: {result1['new_goals']}")

    # Apply second tactic
    print("\n=== Applying Second Tactic ===")
    tactic2 = "have h2 : x = 5 := by linarith [h1, h₀]"
    result2 = client.apply_tactic(session_id, tactic2)
    print(f"Tactic: {tactic2}")
    print(f"Success: {result2['success']}")
    print(f"New goals: {result2['new_goals']}")

    # Apply final tactic
    print("\n=== Applying Final Tactic ===")
    tactic3 = "rw [h2]; norm_num"
    result3 = client.apply_tactic(session_id, tactic3)
    print(f"Tactic: {tactic3}")
    print(f"Success: {result3['success']}")
    print(f"Proof complete: {result3['proof_complete']}")

    # Get final proof state
    print("\n=== Final Proof State ===")
    final_state = client.get_proof_state(session_id)
    print(f"Status: {final_state['status']}")
    print(f"Proof complete: {final_state['proof_complete']}")

    # Get generated proof script
    print("\n=== Generated Proof Script ===")
    script = client.get_proof_script(session_id)
    print(f"Script:\n{script['script']}")

    # Analyze proof attempts
    print("\n=== Proof Analysis ===")
    analysis = client.analyze_proof_attempts(session_id)
    print(f"Total steps: {analysis['total_steps']}")
    print(f"Success rate: {analysis['success_rate']:.2f}")
    print(f"Completed branches: {analysis['completed_branches']}")

    # Create a new branch and demonstrate branching
    print("\n=== Creating New Branch ===")
    branch_response = client.create_branch(session_id, "alternative_approach")
    new_branch_id = branch_response["branch_id"]
    print(f"Created branch: {new_branch_id}")

    # Switch to new branch
    print("\n=== Switching to New Branch ===")
    switch_response = client.switch_branch(session_id, new_branch_id)
    print(f"Switched to branch: {new_branch_id}")
    print(f"Goals in new branch: {switch_response['current_goals']}")

    # List all branches
    print("\n=== All Branches ===")
    branches = client.list_branches(session_id)
    for branch in branches["branches"]:
        print(f"Branch: {branch['branch_id']}")
        print(f"  Status: {branch['status']}")
        print(f"  Steps: {branch['step_count']}")
        print(f"  Current: {branch['is_current']}")

    # List all sessions
    print("\n=== All Sessions ===")
    sessions = client.list_sessions()
    for session in sessions["sessions"]:
        print(f"Session: {session['session_id']}")
        print(f"  Theorem: {session['theorem_statement'][:50]}...")
        print(f"  Status: {session['status']}")
        print(f"  Complete: {session['proof_complete']}")

    print(f"\n=== Example completed successfully! ===")
    return session_id

if __name__ == "__main__":
    try:
        example_usage()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the FastAPI server is running.")
        print("Start the server with: python lean_proof_api.py")
    except Exception as e:
        print(f"Error: {e}")
