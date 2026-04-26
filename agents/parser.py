import os
import json
import uuid
import time
import re
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

def _fallback_extract(citation_paragraphs: List[str]) -> List[ExtractedClaim]:
    """Domain-agnostic hybrid claim miner for fallback extraction."""
    claims = []
    
    # Multi-strategy regex patterns
    citation_pattern = re.compile(r'(\[\d+(?:,\s*\d+)*\]|\([A-Za-z]+(?: et al\.)?, \d{4}\))')
    novelty_pattern = re.compile(r'\b(we propose|introduce|present|develop|our method|novel|first to)\b', re.IGNORECASE)
    result_pattern = re.compile(r'\b(results show|indicate|demonstrate|achieve|outperform|reveal|suggest)\b', re.IGNORECASE)
    
    # Expanded quantitative patterns split to avoid \b boundary errors on non-word chars (like %)
    quant_values = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|x|times|-fold|p\s*<\s*0\.\d+|±)', re.IGNORECASE)
    quant_metrics = re.compile(r'\b(?:RMSE|MAE|MSE|AUC|R²|R2|accuracy|F1|precision|recall|sensitivity|specificity)\b', re.IGNORECASE)

    scored_sentences = []
    
    for paragraph in citation_paragraphs:
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        for sentence in sentences:
            if len(sentence.split()) < 8: # Ignore overly short/fragmented sentences
                continue
                
            score = 0
            citations = citation_pattern.findall(sentence)
            
            # 1. Citation-backed (Base requirement for standard verification, +1)
            if citations:
                score += 1
                
            # 2. Novelty claims (+2)
            if novelty_pattern.search(sentence):
                score += 2
                
            # 3. Evidence/Result claims (+2)
            if result_pattern.search(sentence):
                score += 2
                
            # 4. Quantitative result sentences (+2)
            if quant_values.search(sentence) or quant_metrics.search(sentence):
                score += 2
                
            # Extract anything hitting at least one heuristic (e.g. just a citation)
            if score >= 1:
                ref = citations[0] if citations else "N/A"
                scored_sentences.append((score, sentence.strip(), ref))

    # Sort descending by score to prioritize the densest, most defensible claims
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate and extract top 5
    seen = set()
    for score, text, ref in scored_sentences:
        if text not in seen:
            seen.add(text)
            claims.append(ExtractedClaim(
                claim_id=f"fallback_{uuid.uuid4().hex[:6]}",
                in_text_claim=text,
                citation_reference=ref,
                full_bibliography_entry="",
                cited_paper_title=""
            ))
        if len(claims) >= 5:
            break
            
    return claims

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
    5. Extract the exact title of the cited paper from that bibliography entry (cited_paper_title).
    
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
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
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
                    
            if claims:
                return claims
            else:
                raise ValueError("LLM returned 0 claims.")
                
        except Exception as e:
            print(f"[Parser Agent Error] Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay ** (attempt + 1))
            else:
                print("[!] LLM extraction failed completely. Engaging regex fallback...")
                
    fallback_claims = _fallback_extract(citation_paragraphs)
    if not fallback_claims:
        print("[!] Fallback regex also yielded 0 claims.")
    else:
        print(f"[*] Fallback regex successfully extracted {len(fallback_claims)} claims.")
        
    return fallback_claims

if __name__ == "__main__":
    # Smoke Test
    print("Parser Agent initialized. Call extract_critical_claims(...) with PyMuPDF output.")
