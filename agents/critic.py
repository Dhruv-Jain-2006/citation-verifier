import os
import json
import time
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from utils.schemas import ExtractedClaim, RetrievedEvidence, VerificationScore, BatchedVerificationOutput

# Ensure environment variables are loaded for the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def execute_critic_override(claims: List[ExtractedClaim], 
                            evidence_list: List[RetrievedEvidence], 
                            verifications: List[VerificationScore]) -> List[VerificationScore]:
    """
    Second-pass override layer acting as a deterministic Appellate Court.
    Uses Batched evaluation to prevent Quota Exhaustion.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    overrides = []
    
    claims_map = {c.claim_id: c for c in claims}
    evidence_map = {e.claim_id: e for e in evidence_list}
    
    disputed_items = []
    
    for v in verifications:
        # C. Critic trigger restraint
        # Trigger only for unsupported claims, or low-confidence partially_supported claims.
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
        return overrides
        
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
    You are given a batch of {len(disputed_items)} disputed claims.
    Evaluate each claim INDEPENDENTLY. Do not let one claim's weakness influence judgments for others.
    
    Be citation-function aware. Consider the citation role (e.g., empirical support vs background/lineage).

    Rules:
    - For direct empirical claims, mark "supported" ONLY if support is explicit in the abstract.
    - For background/lineage citations: If the abstract establishes foundational relevance but does not fully substantiate the author's modern framing, default to "partially_supported". Reserve "supported" only when the abstract genuinely supports the exact framing.
    - CRITICAL DOWNGRADE RULE: If a claim depends on details absent from an abstract (equations, parameterization, fine-grained comparisons), but the abstract strongly matches the topic relevance, downgrade "unsupported" to "partially_supported".
    - Reserve "unsupported" strictly for explicit contradictions or completely irrelevant citations.
    - Do NOT hallucinate missing evidence or assume domain knowledge beyond the abstract.

    Goal:
    Correct overly strict or unclear reasoning. Do not falsely flag legitimate context/lineage citations as unsupported. Do not weaken justified contradiction findings.

    Return the assessment as structured JSON adhering to the constraints.
    CRITICAL: Ensure the `claim_id` in your output EXACTLY matches the Claim ID provided in the item block.
    
    {claims_text}
    """
    
    max_retries = 3
    base_delay = 4
    
    for attempt in range(max_retries):
        try:
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
            
            # A. Dual mapping protection (both ID and positional validation)
            if len(batch_results) != len(disputed_items):
                print(f"[Critic Warning] Length mismatch: Expected {len(disputed_items)}, got {len(batch_results)}")
                
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
                    
                    # Prevent overly aggressive reversal
                    if override.support == "supported" and v.support == "unsupported":
                        override.support = "partially_supported"

                    # Prevent critic from erasing contradictions too easily
                    if v.contradiction_detected and not override.contradiction_detected:
                        override.support = "partially_supported"
                        override.contradiction_detected = True
                        
                    overrides.append(override)
                except Exception as map_err:
                    print(f"    [!] Critic Mapping error for {claim.claim_id}: {map_err}")
            
            break # Success
            
        except ResourceExhausted as e:
            # D. Quota exhaustion handling
            if attempt < max_retries - 1:
                sleep_time = base_delay ** (attempt + 1)
                print(f"[Critic 429] Quota exhausted. Waiting {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                print(f"[Critic Error] Ultimate Quota Exhaustion.")
                break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay ** (attempt + 1))
                continue
            
            print(f"[Critic Error] Failed LLM Batch execution: {e}")
            break

    return overrides

if __name__ == "__main__":
    print("Semantic Critic Module initialized.")
