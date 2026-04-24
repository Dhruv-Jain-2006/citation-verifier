import os
import json
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
                evidence_strength="none",
                contradiction_detected=False,
                reasoning="Abstract could not be explicitly retrieved from Semantic Scholar.",
                confidence=0
            ))
            continue
            
        prompt = f"""
        You are a skeptical scientific citation verification judge.

        Task:
        Determine whether the provided abstract fundamentally supports the claim made by the authors.

        Claim text:
        "{claim.in_text_claim}"

        Cited Abstract Evidence:
        "{evidence.abstract}"

        Evaluate conservatively.

        Rules:
        - Only mark "supported" if support is explicit in the abstract.
        - If any inference, extrapolation, or indirect reasoning is required, return "partially_supported".
        - If the claim is absent or contradicted, return "unsupported".

        Rubric:
        - unsupported: Claim is explicitly contradicted or conceptually absent.
        - partially_supported: Related evidence exists but support is incomplete or inferred.
        - supported: Every key concept in the claim is explicitly supported.

        Do not be generous.
        Bias toward partially_supported when uncertain.

        Return the assessment as structured JSON adhering to the constraints.
        """
        
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
            
        except Exception as e:
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