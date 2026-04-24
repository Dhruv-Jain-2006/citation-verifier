import operator
from typing import TypedDict, Annotated, List, Optional
from utils.schemas import (
    ExtractedClaim, 
    BroadScreeningResult,
    RetrievedEvidence, 
    VerificationScore, 
    IntegrityReport
)

class GraphState(TypedDict):
    """
    The strictly typed state machine payload passed between LangGraph nodes.
    Each node receives this state, and returns a dictionary with updates.
    LangGraph merges the returned dictionary into this global state.
    """
    
    # --- Input Variables ---
    pdf_path: str
    pdf_text: str
    
    # --- Accumulated Node State ---
    # `Annotated[List, operator.add]` tells LangGraph that when a node returns 
    # {"claims": [new_claim]}, it should APPEND it to the existing list, not overwrite it.
    # This is critical for parallel node execution and mapping.
    
    claims: Annotated[List[ExtractedClaim], operator.add]
    screening_results: Annotated[List[BroadScreeningResult], operator.add]
    triaged_claims: Annotated[List[ExtractedClaim], operator.add]
    evidence: Annotated[List[RetrievedEvidence], operator.add]
    verifications: Annotated[List[VerificationScore], operator.add]
    critic_overrides: Annotated[List[VerificationScore], operator.add]
    
    # --- Error Tracking ---
    # Allows agents to flag failures (e.g., "API Timeout") without crashing the pipeline.
    errors: Annotated[List[str], operator.add]
    
    # --- Final Synthesized Output ---
    report: Optional[IntegrityReport]
