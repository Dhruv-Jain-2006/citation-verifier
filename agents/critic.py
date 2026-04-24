import os
import json
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

from utils.schemas import ExtractedClaim, RetrievedEvidence, VerificationScore

# Ensure environment variables are loaded for the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def execute_critic_override(claims: List[ExtractedClaim], 
                            evidence_list: List[RetrievedEvidence], 
                            verifications: List[VerificationScore]) -> List[VerificationScore]:
    """
    Second-pass override layer acting as a deterministic Appellate Court.
    Only evaluates claims flagged by the conditional router.
    Extracts nuanced context that a simple structural verifier might miss.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    overrides = []
    
    claims_map = {c.claim_id: c for c in claims}
    evidence_map = {e.claim_id: e for e in evidence_list}
    
    for v in verifications:
        # Pass-through clean claims (Do not waste LLM cycles)
        if v.support == "supported" and v.evidence_strength == "strong" and not v.contradiction_detected:
            continue
            
        # Target only the weak ones for review
        claim = claims_map.get(v.claim_id)
        evidence = evidence_map.get(v.claim_id)
        
        # If the abstract itself is missing, the critic cannot do anything either.
        if not claim or not evidence or not evidence.abstract:
            continue
            
        prompt = f"""
        You are an appellate scientific verification judge.

        A prior strict verifier flagged this claim as weak or uncertain.
        You must reassess conservatively using only the abstract.

        Claim:
        "{claim.in_text_claim}"

        Abstract:
        "{evidence.abstract}"

        Lower Verifier Decision:
        support = {v.support}
        reasoning = "{v.reasoning}"

        Rules:
        - Only mark "supported" if support is explicit in the abstract.
        - If any inference or extrapolation is required, return "partially_supported".
        - If contradicted or absent, return "unsupported".
        - Do NOT hallucinate missing evidence.
        - Do NOT assume domain knowledge beyond the abstract.

        Goal:
        Correct overly strict or unclear reasoning, but do NOT be lenient.

        Return strict JSON only using the schema.
        """
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=VerificationScore,
                    temperature=0.0
                )
            )
            
            data = json.loads(response.text)
            data["claim_id"] = claim.claim_id

            override = VerificationScore(**data)

            # Prevent overly aggressive reversal
            if override.support == "supported" and v.support == "unsupported":
                override.support = "partially_supported"

            overrides.append(override)
            
        except Exception as e:
            print(f"[Critic Error] Failed to generate override for {claim.claim_id}: {e}")
            
    return overrides

if __name__ == "__main__":
    print("Semantic Critic Module initialized.")
