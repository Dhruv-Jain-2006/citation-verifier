from typing import List, Dict
from utils.schemas import BroadScreeningResult, VerificationScore

def compute_trust_score(
    screening_results: List[BroadScreeningResult],
    verifications: List[VerificationScore],
    critic_overrides: List[VerificationScore]
) -> Dict:
    """
    Computes a deterministic, interpretable 0-100 Trust Score for the paper.
    No LLM is used. Relies purely on rigid heuristics from prior nodes.
    
    Returns:
        dict: {
            "trust_score": int,
            "penalties": list of dicts,
            "high_risk_claims": list of claim IDs
        }
    """
    base_score = 100
    penalties = []
    high_risk_claims = []
    
    # 1. Merge the Verification and Critic scores
    # If the critic reviewed a claim, its score completely overwrites the verifier.
    final_verifications = {v.claim_id: v for v in verifications}
    for override in critic_overrides:
        final_verifications[override.claim_id] = override
        
    # 2. Score the Broad Metadata (Stage 1)
    for screen in screening_results:
        if screen.retracted:
            penalties.append({
                "claim_id": screen.claim_id,
                "penalty": -40,
                "reason": "Cited a retracted paper."
            })
            if screen.claim_id not in high_risk_claims:
                high_risk_claims.append(screen.claim_id)
                
        elif screen.suspicious:
            penalties.append({
                "claim_id": screen.claim_id,
                "penalty": -20,
                "reason": "Citation appears hallucinated (e.g., found but 0 citations)."
            })
        
        # Mild uncertainty tax for unresolved API calls (not elif — can stack with retracted/suspicious)
        if screen.resolution_failed:
            penalties.append({
                "claim_id": screen.claim_id,
                "penalty": -2,
                "reason": "[System Uncertainty] Citation could not be verified due to API timeout/rate limit."
            })
            
    # 3. Score the Deep Verifications (Stage 2)
    for v in final_verifications.values():
        if v.contradiction_detected:
            penalties.append({
                "claim_id": v.claim_id,
                "penalty": -35,
                "reason": f"Abstract explicitly contradicts the claim. ({v.reasoning})"
            })
            if v.claim_id not in high_risk_claims:
                high_risk_claims.append(v.claim_id)
                
        elif v.support == "unsupported":
            penalties.append({
                "claim_id": v.claim_id,
                "penalty": -25,
                "reason": f"Claim unsupported by abstract. ({v.reasoning})"
            })
            if v.claim_id not in high_risk_claims:
                high_risk_claims.append(v.claim_id)
                
        elif v.support == "partially_supported":
            penalties.append({
                "claim_id": v.claim_id,
                "penalty": -10,
                "reason": f"Claim requires inference / only partially supported. ({v.reasoning})"
            })
            
        elif v.support == "unverifiable":
            penalties.append({
                "claim_id": v.claim_id,
                "penalty": -3,
                "reason": "[System Uncertainty] Abstract could not be retrieved for verification."
            })
            
        if v.evidence_strength == "weak" and not v.contradiction_detected and v.support == "supported":
            penalties.append({
                "claim_id": v.claim_id,
                "penalty": -5,
                "reason": "Evidence strength was flagged as weak."
            })
            
    # Compile final integer
    total_penalty = sum([p["penalty"] for p in penalties])
    final_score = max(0, base_score + total_penalty) # Floor at 0
    
    return {
        "trust_score": final_score,
        "penalties": penalties,
        "high_risk_claims": high_risk_claims
    }

if __name__ == "__main__":
    print("Deterministic Trust Scorer Module initialized.")
