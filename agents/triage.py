from typing import List, Tuple
from utils.schemas import ExtractedClaim, BroadScreeningResult

def select_top_risky_claims(claims: List[ExtractedClaim],
                            screening_results: List[BroadScreeningResult],
                            threshold: int = 40,
                            max_claims: int = 5) -> List[ExtractedClaim]:
    """
    Ranks claims based on the Stage 1 metadata screen and heuristics.
    Selects the most foundational/risky claims for Stage 2 Deep Verification.
    
    Args:
        claims: All claims parsed from the document.
        screening_results: The BroadScreeningResult objects for all claims.
        threshold: Minimum risk score for a claim to warrant deep verification.
        max_claims: Absolute ceiling on how many claims to pass to Stage 2.
        
    Returns:
        List of ExtractedClaims selected for Deep Verification.
    """
    
    # Map screening results by claim_id for O(1) lookup
    screen_map = {res.claim_id: res for res in screening_results}
    
    scored_claims: List[Tuple[int, ExtractedClaim]] = []
    
    for claim in claims:
        # Base logic: If no screening result found, assume moderate risk fallback
        screen_res = screen_map.get(claim.claim_id)
        if not screen_res:
            score = 30
        else:
            score = screen_res.risk_score
            
        # --- CLAIM IMPORTANCE HEURISTIC ---
        # Longer in-text claims often assert specific, testable methodologies/results.
        # Shorter claims are often throwaway intro facts ("Deep learning is popular [1]").
        word_count = len(claim.in_text_claim.split())
        if word_count > 10:
            score += 15
        elif word_count < 5:
            score -= 10
            
        scored_claims.append((score, claim))
        
    # Sort claims descending by their final triage score (highest risk first)
    scored_claims.sort(key=lambda x: x[0], reverse=True)
    
    triaged = []
    for score, claim in scored_claims:
        if score >= threshold and len(triaged) < max_claims:
            print(f"[*] Triaged Claim '{claim.claim_id}' (Score: {score})")
            triaged.append(claim)
            
    # Guarantee at least 1 claim makes it through if the document has any claims at all
    # (prevents the pipeline from returning a blank report if all claims are "safe")
    if not triaged and scored_claims:
        print(f"[*] Triaged fallback Claim '{scored_claims[0][1].claim_id}' (Score: {scored_claims[0][0]})")
        triaged.append(scored_claims[0][1])
        
    return triaged[:max_claims]

if __name__ == "__main__":
    # Smoke Test
    print("Triage module initialized.")
