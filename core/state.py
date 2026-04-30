import operator
from typing import TypedDict, Annotated, List, Optional
from utils.schemas import (
    ExtractedClaim, 
    BroadScreeningResult,
    RetrievedEvidence, 
    VerificationScore,
    IntegrityReport
)

def merge_call_counts(old_counts: dict, new_counts: dict) -> dict:
    """Combines two call count dictionaries by summing matching keys."""
    combined = old_counts.copy() if old_counts else {}
    if not new_counts:
        return combined
        
    for k, v in new_counts.items():
        if isinstance(v, dict):
            # Recurse for nested counts (e.g. per-agent counts)
            combined[k] = merge_call_counts(combined.get(k, {}), v)
        else:
            combined[k] = combined.get(k, 0) + v
    return combined


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
    errors: Annotated[List[str], operator.add]
    
    # --- Trust Score Output ---
    score_data: dict
    
    # --- Final Synthesized Output ---
    report: Optional[IntegrityReport]

    # --- Quota and Instrumentation ---
    llm_quota_exhausted: bool
    call_counts: Annotated[dict, merge_call_counts]
