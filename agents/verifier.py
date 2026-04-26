import os
import json
import time
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

from utils.schemas import ExtractedClaim, RetrievedEvidence, VerificationScore

# Ensure environment variables are loaded for the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def evaluate_claims(claims: List[ExtractedClaim], evidence_list: List[RetrievedEvidence]) -> List[VerificationScore]:
    """
    Ingests triaged claims and abstract evidence, verifying support levels natively with Gemini structured output.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Map evidence by claim_id for quick lookup
    evidence_map = {e.claim_id: e for e in evidence_list}
    verifications = []
    
    for claim in claims:
        evidence = evidence_map.get(claim.claim_id)
        
        # If we failed to retrieve an abstract in the deep_retrieval node,
        # fallback to an unverifiable score deterministically rather than wasting an LLM call.
        if not evidence or not evidence.found or not evidence.abstract:
            verifications.append(VerificationScore(
                claim_id=claim.claim_id,
                support="unverifiable",
                evidence_strength="weak",
                contradiction_detected=False,
                reasoning="Abstract could not be explicitly retrieved from Semantic Scholar.",
                confidence=0
            ))
            continue
            
        prompt = f"""
        You are a skeptical scientific citation verification judge.

        Task:
        Determine whether the provided abstract fundamentally supports the claim made by the authors.

        Be citation-function aware. First, identify the citation role:
        - direct empirical support
        - background/context citation
        - method lineage citation
        - comparative citation

        Claim text:
        "{claim.in_text_claim}"

        Cited Abstract Evidence:
        "{evidence.abstract}"

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

        Do not be generous, but do not falsely flag legitimate context/lineage citations as unsupported.

        Return the assessment as structured JSON adhering to the constraints.
        """
        
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Enforce the strict Pydantic Generation Config
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=VerificationScore,
                        temperature=0.0
                    )
                )
                
                data = json.loads(response.text)
                
                # The LLM is generally good, but to ensure strict LangGraph mapping, 
                # we force-override the claim_id to guarantee graph integrity.
                data["claim_id"] = claim.claim_id
                
                verifications.append(VerificationScore(**data))
                break # Success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(base_delay ** (attempt + 1))
                    continue
                
                print(f"[Verifier Error] Failed LLM execution on {claim.claim_id}: {e}")
                verifications.append(VerificationScore(
                    claim_id=claim.claim_id,
                    support="unverifiable",
                    evidence_strength="none",
                    contradiction_detected=False,
                    reasoning=f"LLM semantic parsing failed: {e}",
                    confidence=0
                ))



    return verifications

if __name__ == "__main__":
    print("Semantic Verifier Module initialized.")