from typing import List
from utils.schemas import IntegrityReport, VerificationScore, BroadScreeningResult

def synthesize_report(
    score_data: dict,
    screening_results: List[BroadScreeningResult],
    verifications: List[VerificationScore],
    critic_overrides: List[VerificationScore]
) -> IntegrityReport:
    """
    Deterministically synthesizes the final executive IntegrityReport.
    No LLM is used. Relies purely on counting and structured data to build an audit trail.
    """
    
    # Base data
    trust_score = score_data.get("trust_score", 0)
    high_risk_claims = score_data.get("high_risk_claims", [])
    penalties = score_data.get("penalties", [])
    
    # 1. Tallying (Stage 1)
    total_screened = len(screening_results)
    retracted_count = sum(1 for s in screening_results if s.retracted)
    suspicious_count = sum(1 for s in screening_results if s.suspicious)
    
    # Merge critic overrides (Stage 2)
    final_verifications = {v.claim_id: v for v in verifications}
    for override in critic_overrides:
        final_verifications[override.claim_id] = override
        
    total_verified = len(final_verifications)
    supported_count = sum(1 for v in final_verifications.values() if v.support == "supported")
    partially_supported = sum(1 for v in final_verifications.values() if v.support == "partially_supported")
    unsupported_count = sum(1 for v in final_verifications.values() if v.support == "unsupported")
    contradictions = sum(1 for v in final_verifications.values() if v.contradiction_detected)
    
    # 2. Building the Deterministic Executive Summary
    summary_lines = []
    summary_lines.append(f"CITATION INTEGRITY AUDIT: OVERALL SCORE [{trust_score}/100]")
    summary_lines.append("-" * 50)
    summary_lines.append(f"STAGE 1: BROAD SCREENING ({total_screened} Citations Checked)")
    
    if retracted_count > 0:
        summary_lines.append(f"🚨 CRITICAL WARNING: {retracted_count} retracted paper(s) cited in this document.")
    else:
        summary_lines.append("✅ No retracted papers detected.")
        
    if suspicious_count > 0:
        summary_lines.append(f"⚠️ WARNING: {suspicious_count} citation(s) appear unresolvable or hallucinated.")
    else:
        summary_lines.append("✅ All citations successfully resolved.")
        
    summary_lines.append("")
    summary_lines.append(f"STAGE 2: DEEP VERIFICATION ({total_verified} High-Risk Claims Checked)")
    summary_lines.append(f"- Supported: {supported_count}")
    summary_lines.append(f"- Partially Supported: {partially_supported}")
    summary_lines.append(f"- Unsupported: {unsupported_count}")
    
    if contradictions > 0:
        summary_lines.append(f"🚨 MAJOR CONTRADICTION DETECTED: {contradictions} abstract(s) explicitly contradict the author's claims.")
        
    summary_lines.append("")
    summary_lines.append("PENALTY LOG:")
    if not penalties:
        summary_lines.append("None. Paper appears highly reliable.")
    else:
        for p in penalties:
            summary_lines.append(f"[-{abs(p['penalty'])}] {p['claim_id']}: {p['reason']}")
            
    final_summary = "\n".join(summary_lines)
    
    return IntegrityReport(
        trust_score=trust_score,
        summary=final_summary,
        high_risk_claims=high_risk_claims
    )

if __name__ == "__main__":
    print("Deterministic Reporter Module initialized.")
