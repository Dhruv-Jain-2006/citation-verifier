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
    
    audit_failed = score_data.get("audit_failed", False)
    if audit_failed:
        summary_lines = [
            "CITATION INTEGRITY AUDIT: FAILED / INCOMPLETE",
            "-" * 50,
            "🚨 CRITICAL ERROR: The system failed to extract any verifiable claims from the document.",
            "This may be due to LLM API rate limits, parsing failures, or a document containing zero citations.",
            "A Trust Score cannot be securely calculated."
        ]
        return IntegrityReport(
            trust_score=None,
            summary="\n".join(summary_lines),
            high_risk_claims=[]
        )
        
    # Base data
    trust_score = score_data.get("trust_score", 0)
    high_risk_claims = score_data.get("high_risk_claims", [])
    penalties = score_data.get("penalties", [])
    
    # 1. Tallying (Stage 1)
    total_screened = len(screening_results)
    retracted_count = sum(1 for s in screening_results if s.retracted)
    suspicious_count = sum(1 for s in screening_results if s.suspicious)
    unresolved_api_count = sum(1 for s in screening_results if getattr(s, "resolution_failed", False))
    not_found_count = sum(1 for s in screening_results if not s.exists and not getattr(s, "resolution_failed", False))
    
    # Merge critic overrides (Stage 2)
    final_verifications = {v.claim_id: v for v in verifications}
    for override in critic_overrides:
        final_verifications[override.claim_id] = override
        
    total_verified = len(final_verifications)
    supported_count = sum(1 for v in final_verifications.values() if v.support == "supported")
    partially_supported = sum(1 for v in final_verifications.values() if v.support == "partially_supported")
    unsupported_count = sum(1 for v in final_verifications.values() if v.support == "unsupported")
    unverifiable_count = sum(1 for v in final_verifications.values() if v.support == "unverifiable")
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
        summary_lines.append(f"⚠️ WARNING: {suspicious_count} citation(s) appear hallucinated.")
    else:
        summary_lines.append("✅ All resolved citations appear legitimate.")
        
    if unresolved_api_count > 0:
        summary_lines.append(f"ℹ️ NOTE: {unresolved_api_count} citation(s) hit API rate limits or timeouts.")
        
    if not_found_count > 0:
        summary_lines.append(f"ℹ️ NOTE: {not_found_count} citation(s) returned 0 search results but were not penalized.")
        
    summary_lines.append("")
    summary_lines.append(f"STAGE 2: DEEP VERIFICATION ({total_verified} High-Risk Claims Checked)")
    summary_lines.append(f"- Supported: {supported_count}")
    summary_lines.append(f"- Partially Supported: {partially_supported}")
    summary_lines.append(f"- Unsupported: {unsupported_count}")
    summary_lines.append(f"- Unverifiable (Missing Abstract): {unverifiable_count}")
    
    if contradictions > 0:
        summary_lines.append(f"🚨 MAJOR CONTRADICTION DETECTED: {contradictions} abstract(s) explicitly contradict the author's claims.")
        
    summary_lines.append("")

    integrity_penalties = [
        p for p in penalties if "[System Uncertainty]" not in p["reason"]
    ]

    uncertainty_penalties = [
        p for p in penalties if "[System Uncertainty]" in p["reason"]
    ]

    summary_lines.append("PENALTY LOG (INTEGRITY):")
    if not integrity_penalties:
        summary_lines.append("None. Paper appears highly reliable.")
    else:
        for p in integrity_penalties:
            summary_lines.append(
                f"[-{abs(p['penalty'])}] {p['claim_id']}: {p['reason']}"
            )

    summary_lines.append("")
    summary_lines.append("SYSTEM UNCERTAINTY LOG:")

    if not uncertainty_penalties:
        summary_lines.append("None.")
    else:
        summary_lines.append(
            f"{len(uncertainty_penalties)} uncertainty penalties applied."
        )

        for p in uncertainty_penalties:
            summary_lines.append(
                f"[-{abs(p['penalty'])}] {p['claim_id']}: {p['reason']}"
            )

    # Executive Verdict (always rendered)
    summary_lines.append("")
    
    # Check if high uncertainty dominates the audit
    total_claims = total_verified
    uncertainty_ratio = unverifiable_count / total_claims if total_claims > 0 else 0
    
    if trust_score >= 85 and uncertainty_ratio <= 0.5:
        summary_lines.append("VERDICT: High citation integrity.")
    elif trust_score >= 85 and uncertainty_ratio > 0.5:
        summary_lines.append(
            "VERDICT: Score is high, but system confidence is limited due to unresolved citations. "
            "Manual spot-checking recommended."
        )
    elif trust_score >= 65:
        summary_lines.append("VERDICT: Moderate integrity; caution advised.")
    else:
        summary_lines.append("VERDICT: Low integrity; significant citation risk detected.")
            
    final_summary = "\n".join(summary_lines)
    
    return IntegrityReport(
        trust_score=trust_score,
        summary=final_summary,
        high_risk_claims=high_risk_claims
    )

if __name__ == "__main__":
    print("Deterministic Reporter Module initialized.")
