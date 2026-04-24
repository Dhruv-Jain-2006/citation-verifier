import os
import json
import uuid
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel

from utils.schemas import ExtractedClaim

# Ensure environment variables are loaded for the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ParserOutput(BaseModel):
    """Wrapper for the structured output list to ensure strict Gemini generation."""
    claims: list[ExtractedClaim]

def extract_critical_claims(citation_paragraphs: List[str], references_section: str) -> List[ExtractedClaim]:
    """
    Passes the deterministically extracted chunks to Gemini to
    intelligently map the citations to the bibliography.
    
    Args:
        citation_paragraphs: List of text blocks containing citations (e.g. from PyMuPDF).
        references_section: The raw string block of the bibliography.
        
    Returns:
        List of Pydantic ExtractedClaim objects (Max 5).
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Limit prompt size for context safety (MVP)
    citation_paragraphs = citation_paragraphs[:15]

    # Trim bibliography to avoid oversized prompts
    references_section = references_section[:12000]

    prompt = f"""
    You are an expert scientific fact-checker parsing a research paper.
    
    We have extracted paragraphs from the paper that contain citations, as well as the paper's bibliography.
    
    Your task:
    1. Identify up to the 5 most critical, load-bearing claims made by the authors that rely on a citation.
    2. For each claim, extract the exact text or a direct paraphrase (in_text_claim).
    3. Identify the inline citation marker used for that claim, e.g. "[12]" or "(Smith, 2023)" (citation_reference).
    4. Search the provided bibliography to find the exact full reference string corresponding to that marker.
    
    CONSTRAINTS:
    - Never invent citations. If the marker isn't found in the bibliography, leave `full_bibliography_entry` empty.
    - Treat claims as factual statements that can be verified against an abstract.
    - Assign a unique string for `claim_id` (e.g., "claim_1").
    - Return exactly 5 claims maximum.
    - Prefer foundational claims over minor supporting claims.
    
    === CITATION PARAGRAPHS ===
    {" ".join(citation_paragraphs)}
    
    === BIBLIOGRAPHY SECTION ===
    {references_section}
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ParserOutput,
                temperature=0.0 # Deterministic
            )
        )
        
        # Parse the structured JSON output
        data = json.loads(response.text)
        claims = [ExtractedClaim(**c) for c in data.get("claims", [])]
        
        # Ensure fallback generation of claim_ids if the LLM struggled
        for i, claim in enumerate(claims):
            if not claim.claim_id or claim.claim_id.strip() == "":
                claim.claim_id = f"claim_{uuid.uuid4().hex[:6]}"
                
        return claims
        
    except Exception as e:
        print(f"[Parser Agent Error] Failed to extract claims: {e}")
        return []

if __name__ == "__main__":
    # Smoke Test
    print("Parser Agent initialized. Call extract_critical_claims(...) with PyMuPDF output.")
