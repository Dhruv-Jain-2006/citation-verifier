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

def evaluate_claims(claims: List[ExtractedClaim], evidence_list: List[RetrievedEvidence]) -> List[VerificationScore]:
    """
    Ingests triaged claims and abstract evidence, verifying support levels natively with Gemini structured output.
    Uses Batched evaluation to prevent Quota Exhaustion.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    
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
        return verifications
        
    # Build Batched Prompt
    claims_text = ""
    for i, (c, e) in enumerate(verifiable_claims):
        claims_text += f"\n--- CLAIM ITEM {i+1} ---\n"
        claims_text += f"Claim ID: {c.claim_id}\n"
        claims_text += f"Claim Text: \"{c.in_text_claim}\"\n"
        claims_text += f"Cited Abstract Evidence: \"{e.abstract}\"\n"
        
    prompt = f"""
    You are a skeptical scientific citation verification judge.
    
    Task:
    You are given a batch of {len(verifiable_claims)} claims to evaluate. 
    Evaluate each claim INDEPENDENTLY. Do not let one claim's weakness influence judgments for others.
    For each claim, determine whether the provided abstract fundamentally supports the claim made by the authors.

    Be citation-function aware. First, identify the citation role:
    - direct empirical support
    - background/context citation
    - method lineage citation
    - comparative citation

    Evaluate conservatively using the following calibration rules:

    Rules:
    - For direct empirical claims, mark "supported" ONLY if support is explicit in the abstract.
    - For background/lineage citations: If the abstract establishes foundational relevance but does not fully substantiate the author's modern framing, default to "partially_supported". Reserve "supported" only when the abstract genuinely supports the exact framing.
    - CRITICAL DOWNGRADE RULE: If a claim depends on details absent from an abstract (equations, parameterization, fine-grained comparisons), but the abstract strongly matches the topic relevance, downgrade "unsupported" to "partially_supported".
    - Reserve "unsupported" strictly for explicit contradictions or completely irrelevant citations.

    Rubric:
    - unsupported: Claim is explicitly contradicted or the citation is completely irrelevant to the core topic.
    - partially_supported: Abstract establishes foundational topic relevance, but support is incomplete, inferred, or relies on details likely in the paper body.
    - supported: Every key concept and framing in the claim is explicitly supported by the abstract.

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
            if len(batch_results) != len(verifiable_claims):
                print(f"[Verifier Warning] Length mismatch: Expected {len(verifiable_claims)}, got {len(batch_results)}")
                
            for i, (claim, evidence) in enumerate(verifiable_claims):
                try:
                    res_data = None
                    if i < len(batch_results):
                        if batch_results[i].get("claim_id") == claim.claim_id:
                            res_data = batch_results[i]
                            
                    if not res_data:
                        # Positional mismatch. Search for it.
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
            
        except ResourceExhausted as e:
            # D. Quota exhaustion handling as System Uncertainty
            if attempt < max_retries - 1:
                sleep_time = base_delay ** (attempt + 1)
                print(f"[Verifier 429] Quota exhausted. Waiting {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                print(f"[Verifier Error] Ultimate Quota Exhaustion.")
                for claim, _ in verifiable_claims:
                    verifications.append(VerificationScore(
                        claim_id=claim.claim_id,
                        support="unverifiable",
                        evidence_strength="none",
                        contradiction_detected=False,
                        reasoning="LLM Quota Exhausted / System Uncertainty",
                        confidence=0
                    ))
                break
        except Exception as e:
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

    return verifications

if __name__ == "__main__":
    print("Semantic Verifier Module initialized.")