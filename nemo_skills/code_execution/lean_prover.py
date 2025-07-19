"""
DEPRECATED: This module has been moved to nemo_skills.code_execution.lean4

This file provides backward compatibility imports from the new lean4 submodule.
New code should import directly from nemo_skills.code_execution.lean4 instead.

For example:
    # Old (still works but deprecated)
    from nemo_skills.code_execution.lean_prover import LeanProver

    # New (preferred)
    from nemo_skills.code_execution.lean4 import LeanProver
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "nemo_skills.code_execution.lean_prover is deprecated. "
    "Use nemo_skills.code_execution.lean4 instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new lean4 submodule for backward compatibility
from nemo_skills.code_execution.lean4 import (
    LeanProver,
    ProofResult,
    ProofInProgress,
)

# Legacy compatibility aliases (if needed)
__all__ = [
    'LeanProver',
    'ProofResult',
    'ProofInProgress',
]
