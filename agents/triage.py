from typing import List, Tuple
from utils.schemas import ExtractedClaim, BroadScreeningResult

def select_top_risky_claims(claims: List[ExtractedClaim],
                            screening_results: List[BroadScreeningResult],
                            threshold: int = 10,
                            max_claims: int = 3) -> List[ExtractedClaim]:
    """
    Ranks claims based on the Stage 1 metadata screen and heuristics.
    Selects the most foundational/risky claims for Stage 2 Deep Verification.
    
    Args:
        claims: All claims parsed from the document.
        screening_results: The BroadScreeningResult objects for all claims.
        threshold: Minimum triage score for a claim to warrant deep verification.
                   Lowered from 40 to 10 to ensure adequate coverage.
        max_claims: Absolute ceiling on how many claims to pass to Stage 2.
        
    Returns:
        List of ExtractedClaims selected for Deep Verification.
    """
    
    # Map screening results by claim_id for O(1) lookup
    screen_map = {res.claim_id: res for res in screening_results}
    
    scored_claims: List[Tuple[int, ExtractedClaim]] = []
    seen_ids = set()  # Prevent duplicate promotion
    
    for claim in claims:
        # Skip duplicates
        if claim.claim_id in seen_ids:
            continue
        seen_ids.add(claim.claim_id)
        
        # Base logic: If no screening result found, assume moderate risk fallback
        screen_res = screen_map.get(claim.claim_id)
        if not screen_res:
            score = 30
        else:
            score = screen_res.risk_score
            
            # Boost claims whose citations couldn't be resolved (API failure)
            # so they get a fair chance at deep verification rather than being silently skipped
            if screen_res.resolution_failed:
                score += 10
            
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
    
    # Promote up to min(3, max_claims, available) to ensure score discrimination
    min_promote = min(3, max_claims, len(scored_claims))
    
    triaged = []
    for score, claim in scored_claims:
        if len(triaged) >= max_claims:
            break
        if score >= threshold or len(triaged) < min_promote:
            print(f"[*] Triaged Claim '{claim.claim_id}' (Score: {score})")
            triaged.append(claim)
            
    return triaged

if __name__ == "__main__":
    # Smoke Test
    print("Triage module initialized.")
