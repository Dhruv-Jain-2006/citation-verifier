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

def execute_critic_override(claims: List[ExtractedClaim], 
                            evidence_list: List[RetrievedEvidence], 
                            verifications: List[VerificationScore],
                            quota_exhausted: bool = False) -> Tuple[List[VerificationScore], dict]:
    """
    Second-pass override layer acting as a deterministic Appellate Court.
    Uses Batched evaluation to prevent Quota Exhaustion.
    """
    updates = {"llm_quota_exhausted": quota_exhausted, "call_counts": {"critic_calls": 0, "retry_calls": 0}}
    overrides = []

    if quota_exhausted:
        print("[Quota Exhaustion Failsafe Activated] Bypassing Critic LLM.")
        return overrides, updates

    model = genai.GenerativeModel("gemini-2.5-flash")
    
    claims_map = {c.claim_id: c for c in claims}
    evidence_map = {e.claim_id: e for e in evidence_list}
    
    disputed_items = []
    
    for v in verifications:
        needs_review = False
        if v.support == "unsupported":
            needs_review = True
        elif v.support == "partially_supported" and v.confidence < 80:
            needs_review = True
            
        if not needs_review:
            continue
            
        claim = claims_map.get(v.claim_id)
        evidence = evidence_map.get(v.claim_id)
        
        if not claim or not evidence or not evidence.abstract:
            continue
            
        disputed_items.append((claim, evidence, v))
        
    if not disputed_items:
        return overrides, updates
        
    # Build Batched Prompt
    claims_text = ""
    for i, (c, e, v) in enumerate(disputed_items):
        claims_text += f"\n--- DISPUTED ITEM {i+1} ---\n"
        claims_text += f"Claim ID: {c.claim_id}\n"
        claims_text += f"Claim Text: \"{c.in_text_claim}\"\n"
        claims_text += f"Cited Abstract Evidence: \"{e.abstract}\"\n"
        claims_text += f"Lower Verifier Decision: support={v.support}, reasoning=\"{v.reasoning}\"\n"

    prompt = f"""
    You are an appellate scientific verification judge.
    Evaluate each of the following {len(disputed_items)} disputed claims INDEPENDENTLY.
    
    {claims_text}
    """
    
    max_retries = 3
    base_delay = 4
    
    for attempt in range(max_retries):
        if updates["llm_quota_exhausted"]:
            break
            
        try:
            updates["call_counts"]["critic_calls"] += 1
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
            
            for i, (claim, evidence, v) in enumerate(disputed_items):
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
                            
                    override = VerificationScore(**res_data)
                    
                    if override.support == "supported" and v.support == "unsupported":
                        override.support = "partially_supported"

                    if v.contradiction_detected and not override.contradiction_detected:
                        override.support = "partially_supported"
                        override.contradiction_detected = True
                        
                    overrides.append(override)
                except Exception as map_err:
                    print(f"    [!] Critic Mapping error for {claim.claim_id}: {map_err}")
            
            break # Success
            
        except Exception as e:
            if classify_quota_error(e) == "RPD":
                print(f"[Quota Exhaustion Failsafe Activated] Detected hard quota exhaustion in Critic: {e}")
                updates["llm_quota_exhausted"] = True
                break

            if attempt < max_retries - 1:
                time.sleep(base_delay ** (attempt + 1))
                continue
            
            print(f"[Critic Error] Failed LLM Batch execution: {e}")
            break

    return overrides, updates

if __name__ == "__main__":
    print("Semantic Critic Module initialized.")
