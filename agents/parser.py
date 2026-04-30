import os
import json
import uuid
import time
import re
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel

from utils.schemas import ExtractedClaim
from utils.quota_manager import classify_quota_error

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
    
    # Expanded quantitative patterns
    quant_values = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|x|times|-fold|p\s*<\s*0\.\d+|±)', re.IGNORECASE)
    quant_metrics = re.compile(r'\b(?:RMSE|MAE|MSE|AUC|R²|R2|accuracy|F1|precision|recall|sensitivity|specificity)\b', re.IGNORECASE)

    scored_sentences = []
    
    for paragraph in citation_paragraphs:
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        for sentence in sentences:
            if len(sentence.split()) < 8: 
                continue
                
            score = 0
            citations = citation_pattern.findall(sentence)
            
            if citations:
                score += 1
            if novelty_pattern.search(sentence):
                score += 2
            if result_pattern.search(sentence):
                score += 2
            if quant_values.search(sentence) or quant_metrics.search(sentence):
                score += 2
                
            if score >= 1:
                ref = citations[0] if citations else "N/A"
                scored_sentences.append((score, sentence.strip(), ref))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
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

def extract_critical_claims(citation_paragraphs: List[str], references_section: str, quota_exhausted: bool = False) -> Tuple[List[ExtractedClaim], dict]:
    """
    Passes the deterministically extracted chunks to Gemini to
    intelligently map the citations to the bibliography.
    """
    updates = {"llm_quota_exhausted": quota_exhausted, "call_counts": {"parser_calls": 0, "retry_calls": 0}}
    
    if quota_exhausted:
        print("[Quota Exhaustion Failsafe Activated] Bypassing Parser LLM.")
        return _fallback_extract(citation_paragraphs), updates

    model = genai.GenerativeModel("gemini-2.5-flash")
    citation_paragraphs = citation_paragraphs[:15]
    references_section = references_section[:12000]

    prompt = f"""
    You are an expert scientific fact-checker parsing a research paper.
    Identify up to the 5 most critical, load-bearing claims made by the authors that rely on a citation.
    
    === CITATION PARAGRAPHS ===
    {" ".join(citation_paragraphs)}
    
    === BIBLIOGRAPHY SECTION ===
    {references_section}
    """
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        if updates["llm_quota_exhausted"]:
            break
            
        try:
            updates["call_counts"]["parser_calls"] += 1
            if attempt > 0:
                updates["call_counts"]["retry_calls"] += 1

                
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ParserOutput,
                    temperature=0.0
                )
            )
            
            data = json.loads(response.text)
            claims = [ExtractedClaim(**c) for c in data.get("claims", [])]
            for i, claim in enumerate(claims):
                if not claim.claim_id or claim.claim_id.strip() == "":
                    claim.claim_id = f"claim_{uuid.uuid4().hex[:6]}"
                    
            if claims:
                return claims, updates
            else:
                raise ValueError("LLM returned 0 claims.")
                
        except Exception as e:
            if classify_quota_error(e) == "RPD":

                print(f"[Quota Exhaustion Failsafe Activated] Detected hard quota exhaustion in Parser: {e}")
                updates["llm_quota_exhausted"] = True
                break
                
            print(f"[Parser Agent Error] Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay ** (attempt + 1))
            else:
                print("[!] LLM extraction failed completely. Engaging regex fallback...")
                
    return _fallback_extract(citation_paragraphs), updates

if __name__ == "__main__":
    print("Parser Agent initialized.")
