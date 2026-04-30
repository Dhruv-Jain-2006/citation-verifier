import os
import json
import time
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from utils.schemas import ExtractedClaim, RetrievedEvidence, VerificationScore, BatchedVerificationOutput
from utils.quota_manager import classify_quota_error

# Ensure environment variables are loaded for the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def evaluate_claims(claims: List[ExtractedClaim], evidence_list: List[RetrievedEvidence], quota_exhausted: bool = False) -> Tuple[List[VerificationScore], dict]:
    """
    Ingests triaged claims and abstract evidence, verifying support levels natively with Gemini structured output.
    Uses Batched evaluation to prevent Quota Exhaustion.
    """
    updates = {"llm_quota_exhausted": quota_exhausted, "call_counts": {"verifier_calls": 0, "retry_calls": 0}}
    
    # Map evidence by claim_id for quick lookup
    evidence_map = {e.claim_id: e for e in evidence_list}
    verifications = []
    
    verifiable_claims = []
    
    for claim in claims:
        evidence = evidence_map.get(claim.claim_id)
        if not evidence or not evidence.found or not evidence.abstract:
            verifications.append(VerificationScore(
                claim_id=claim.claim_id,
                support="unverifiable",
                evidence_strength="weak",
                contradiction_detected=False,
                reasoning="Abstract could not be explicitly retrieved from external source.",
                confidence=0
            ))
        else:
            verifiable_claims.append((claim, evidence))
            
    if not verifiable_claims:
        return verifications, updates
        
    if quota_exhausted:
        print("[Quota Exhaustion Failsafe Activated] Bypassing Verifier LLM.")
        for claim, _ in verifiable_claims:
            verifications.append(VerificationScore(
                claim_id=claim.claim_id,
                support="unverifiable",
                evidence_strength="none",
                contradiction_detected=False,
                reasoning="[System Uncertainty] LLM Quota Exhausted",
                confidence=0
            ))
        return verifications, updates

    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Build Batched Prompt
    claims_text = ""
    for i, (c, e) in enumerate(verifiable_claims):
        claims_text += f"\n--- CLAIM ITEM {i+1} ---\n"
        claims_text += f"Claim ID: {c.claim_id}\n"
        claims_text += f"Claim Text: \"{c.in_text_claim}\"\n"
        claims_text += f"Cited Abstract Evidence: \"{e.abstract}\"\n"
        
    prompt = f"""
    You are a skeptical scientific citation verification judge.
    Evaluate each of the following {len(verifiable_claims)} claims INDEPENDENTLY.
    
    {claims_text}
    """
    
    max_retries = 3
    base_delay = 4
    
    for attempt in range(max_retries):
        if updates["llm_quota_exhausted"]:
            break
            
        try:
            updates["call_counts"]["verifier_calls"] += 1
            if attempt > 0:
                updates["call_counts"]["retry_calls"] += 1
                
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=BatchedVerificationOutput,
                    temperature=0.0
                )
            )
            
            data = json.loads(response.text)
            if "verifications" not in data:
                raise ValueError("Missing verifications field")
            batch_results = data.get("verifications", [])
            
            for i, (claim, evidence) in enumerate(verifiable_claims):
                try:
                    res_data = None
                    if i < len(batch_results):
                        if batch_results[i].get("claim_id") == claim.claim_id:
                            res_data = batch_results[i]
                            
                    if not res_data:
                        matched = next((r for r in batch_results if r.get("claim_id") == claim.claim_id), None)
                        if matched:
                            res_data = matched
                        else:
                            raise ValueError(f"Missing output for claim_id: {claim.claim_id}")
                            
                    verifications.append(VerificationScore(**res_data))
                except Exception as map_err:
                    print(f"    [!] Mapping error for {claim.claim_id}: {map_err}")
                    verifications.append(VerificationScore(
                        claim_id=claim.claim_id,
                        support="unverifiable",
                        evidence_strength="none",
                        contradiction_detected=False,
                        reasoning="[System Uncertainty] Structured output mapping failure.",
                        confidence=0
                    ))
            break # Success
            
        except Exception as e:
            if classify_quota_error(e) == "RPD":
                print(f"[Quota Exhaustion Failsafe Activated] Detected hard quota exhaustion in Verifier: {e}")
                updates["llm_quota_exhausted"] = True
                # Add uncertainty scores for all remaining claims in this batch
                for claim, _ in verifiable_claims:
                    verifications.append(VerificationScore(
                        claim_id=claim.claim_id,
                        support="unverifiable",
                        evidence_strength="none",
                        contradiction_detected=False,
                        reasoning="[System Uncertainty] LLM Quota Exhausted",
                        confidence=0
                    ))
                break
                
            if attempt < max_retries - 1:
                time.sleep(base_delay ** (attempt + 1))
                continue
            
            print(f"[Verifier Error] Failed LLM Batch execution: {e}")
            for claim, _ in verifiable_claims:
                verifications.append(VerificationScore(
                    claim_id=claim.claim_id,
                    support="unverifiable",
                    evidence_strength="none",
                    contradiction_detected=False,
                    reasoning=f"LLM semantic parsing failed: {e}",
                    confidence=0
                ))
            break

    return verifications, updates

if __name__ == "__main__":
    print("Semantic Verifier Module initialized.")