"""
FastAPI wrapper for Lean Proof Iteration API
Provides REST endpoints for AI agents to interact with Lean proofs.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

# Import the LeanProofSession from the existing module
from nemo_skills.code_execution.lean_dojo_api import LeanProofSession, ProofStatus

app = FastAPI(
    title="Lean Proof API",
    description="REST API for AI agents to work on Lean proofs iteratively",
    version="1.0.0"
)

# In-memory storage for proof sessions
proof_sessions: Dict[str, LeanProofSession] = {}

# Pydantic models for request/response validation

class CreateSessionRequest(BaseModel):
    theorem_statement: str = Field(..., description="The theorem statement to prove")
    repo_path: Optional[str] = Field(None, description="Optional path to Lean repository")

class CreateSessionResponse(BaseModel):
    session_id: str
    theorem_statement: str
    created_at: str
    current_goals: List[str]

class ApplyTacticRequest(BaseModel):
    tactic: str = Field(..., description="The tactic to apply")
    branch_id: Optional[str] = Field(None, description="Branch ID (uses current branch if not specified)")

class ApplyTacticResponse(BaseModel):
    success: bool
    step_id: Optional[str] = None
    new_goals: Optional[List[str]] = None
    error: Optional[str] = None
    proof_complete: bool

class UndoStepRequest(BaseModel):
    step_id: str = Field(..., description="ID of the step to undo")
    branch_id: Optional[str] = Field(None, description="Branch ID (uses current branch if not specified)")

class UndoStepResponse(BaseModel):
    success: bool
    removed_steps: Optional[int] = None
    current_goals: Optional[List[str]] = None
    error: Optional[str] = None

class CreateBranchRequest(BaseModel):
    branch_name: str = Field(..., description="Name for the new branch")
    from_step: Optional[str] = Field(None, description="Step ID to branch from (optional)")

class CreateBranchResponse(BaseModel):
    branch_id: str
    parent_branch: str
    created_at: str

class SwitchBranchRequest(BaseModel):
    branch_id: str = Field(..., description="ID of the branch to switch to")

class SwitchBranchResponse(BaseModel):
    success: bool
    current_goals: Optional[List[str]] = None
    step_count: Optional[int] = None
    error: Optional[str] = None

class ProofStateResponse(BaseModel):
    branch_id: str
    status: str
    current_goals: List[str]
    step_count: int
    proof_complete: bool
    last_step: Optional[Dict[str, Any]] = None

class ProofScriptResponse(BaseModel):
    branch_id: str
    script: str
    step_count: int

class AnalysisResponse(BaseModel):
    total_branches: int
    total_steps: int
    successful_steps: int
    success_rate: float
    completed_branches: int
    failed_tactics: List[Dict[str, Any]]
    most_common_failures: List[Dict[str, Any]]

class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]]

# Helper function to get session or raise 404
def get_session(session_id: str) -> LeanProofSession:
    if session_id not in proof_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return proof_sessions[session_id]

# API Endpoints

@app.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: CreateSessionRequest):
    """Create a new proof session"""
    try:
        session = LeanProofSession(
            theorem_statement=request.theorem_statement,
            repo_path=request.repo_path
        )
        proof_sessions[session.session_id] = session

        return CreateSessionResponse(
            session_id=session.session_id,
            theorem_statement=session.theorem_statement,
            created_at=session.created_at,
            current_goals=session.branches[session.current_branch_id].current_goals
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )

@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all active proof sessions"""
    sessions_info = []
    for session_id, session in proof_sessions.items():
        current_branch = session.branches[session.current_branch_id]
        sessions_info.append({
            "session_id": session_id,
            "theorem_statement": session.theorem_statement,
            "created_at": session.created_at,
            "current_branch": session.current_branch_id,
            "status": current_branch.status.value,
            "proof_complete": current_branch.status == ProofStatus.COMPLETED,
            "step_count": len(current_branch.steps)
        })

    return SessionListResponse(sessions=sessions_info)

@app.get("/sessions/{session_id}/state", response_model=ProofStateResponse)
async def get_proof_state(session_id: str, branch_id: Optional[str] = None):
    """Get the current proof state"""
    session = get_session(session_id)

    try:
        state = session.get_proof_state(branch_id)
        if "error" in state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=state["error"]
            )

        return ProofStateResponse(**state)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get proof state: {str(e)}"
        )

@app.post("/sessions/{session_id}/apply-tactic", response_model=ApplyTacticResponse)
async def apply_tactic(session_id: str, request: ApplyTacticRequest):
    """Apply a tactic to the proof"""
    session = get_session(session_id)

    try:
        result = session.apply_tactic(request.tactic, request.branch_id)

        return ApplyTacticResponse(
            success=result["success"],
            step_id=result.get("step_id"),
            new_goals=result.get("new_goals"),
            error=result.get("error"),
            proof_complete=result.get("proof_complete", False)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply tactic: {str(e)}"
        )

@app.post("/sessions/{session_id}/undo-step", response_model=UndoStepResponse)
async def undo_step(session_id: str, request: UndoStepRequest):
    """Undo a specific step"""
    session = get_session(session_id)

    try:
        result = session.undo_step(request.step_id, request.branch_id)

        return UndoStepResponse(
            success=result["success"],
            removed_steps=result.get("removed_steps"),
            current_goals=result.get("current_goals"),
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to undo step: {str(e)}"
        )

@app.post("/sessions/{session_id}/create-branch", response_model=CreateBranchResponse)
async def create_branch(session_id: str, request: CreateBranchRequest):
    """Create a new proof branch"""
    session = get_session(session_id)

    try:
        branch_id = session.create_branch(request.branch_name, request.from_step)

        return CreateBranchResponse(
            branch_id=branch_id,
            parent_branch=session.current_branch_id,
            created_at=session.branches[branch_id].created_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create branch: {str(e)}"
        )

@app.post("/sessions/{session_id}/switch-branch", response_model=SwitchBranchResponse)
async def switch_branch(session_id: str, request: SwitchBranchRequest):
    """Switch to a different branch"""
    session = get_session(session_id)

    try:
        result = session.switch_branch(request.branch_id)

        return SwitchBranchResponse(
            success=result["success"],
            current_goals=result.get("current_goals"),
            step_count=result.get("step_count"),
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch branch: {str(e)}"
        )

@app.get("/sessions/{session_id}/proof-script", response_model=ProofScriptResponse)
async def get_proof_script(session_id: str, branch_id: Optional[str] = None):
    """Get the generated proof script"""
    session = get_session(session_id)

    try:
        script = session.get_proof_script(branch_id)
        effective_branch_id = branch_id or session.current_branch_id
        step_count = len(session.branches[effective_branch_id].steps)

        return ProofScriptResponse(
            branch_id=effective_branch_id,
            script=script,
            step_count=step_count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get proof script: {str(e)}"
        )

@app.get("/sessions/{session_id}/analysis", response_model=AnalysisResponse)
async def analyze_proof_attempts(session_id: str):
    """Analyze all proof attempts and provide insights"""
    session = get_session(session_id)

    try:
        analysis = session.analyze_proof_attempts()

        return AnalysisResponse(**analysis)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze proof attempts: {str(e)}"
        )

@app.get("/sessions/{session_id}/export")
async def export_session(session_id: str):
    """Export the entire session data"""
    session = get_session(session_id)

    try:
        return session.export_session()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export session: {str(e)}"
        )

@app.post("/sessions/{session_id}/save")
async def save_session(session_id: str, filename: str):
    """Save session to file"""
    session = get_session(session_id)

    try:
        session.save_session(filename)
        return {"message": f"Session saved to {filename}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save session: {str(e)}"
        )

@app.post("/sessions/load")
async def load_session(filename: str):
    """Load session from file"""
    try:
        session = LeanProofSession.load_session(filename)
        proof_sessions[session.session_id] = session

        return {
            "session_id": session.session_id,
            "theorem_statement": session.theorem_statement,
            "message": f"Session loaded from {filename}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load session: {str(e)}"
        )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a proof session"""
    if session_id not in proof_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    del proof_sessions[session_id]
    return {"message": f"Session {session_id} deleted"}

@app.get("/sessions/{session_id}/branches")
async def list_branches(session_id: str):
    """List all branches in a session"""
    session = get_session(session_id)

    branches_info = []
    for branch_id, branch in session.branches.items():
        branches_info.append({
            "branch_id": branch_id,
            "status": branch.status.value,
            "step_count": len(branch.steps),
            "current_goals": branch.current_goals,
            "created_at": branch.created_at,
            "is_current": branch_id == session.current_branch_id
        })

    return {"branches": branches_info}

@app.get("/sessions/{session_id}/branches/{branch_id}/steps")
async def get_branch_steps(session_id: str, branch_id: str):
    """Get all steps in a specific branch"""
    session = get_session(session_id)

    if branch_id not in session.branches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Branch {branch_id} not found"
        )

    branch = session.branches[branch_id]
    steps = [step.to_dict() for step in branch.steps]

    return {
        "branch_id": branch_id,
        "steps": steps,
        "total_steps": len(steps)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(proof_sessions)
    }

# Error handler for generic exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
