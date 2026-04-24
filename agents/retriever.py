import time
import requests
import re
from typing import List
from typing import List

from utils.schemas import ExtractedClaim, BroadScreeningResult, RetrievedEvidence

def screen_citations(claims: List[ExtractedClaim]) -> List[BroadScreeningResult]:
    """
    Stage 1 Lightweight Screener.
    Takes all extracted claims and pings the Semantic Scholar API to check metadata.
    Does NOT use LLMs. Strictly deterministic REST execution.
    
    Args:
        claims: The list of ExtractedClaims from the Parser node.
        
    Returns:
        List of BroadScreeningResult Pydantic objects.
    """
    results = []
    
    # Base endpoint for Semantic Scholar search
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    for claim in claims:
        # Prevent huge queries from breaking the GET request
        # First 120 chars generally capture Authors + Title + Year seamlessly
        query_text = claim.full_bibliography_entry.strip()
        if not query_text:
            query_text = claim.in_text_claim # Fallback to claim text if bib is totally missing
            
        search_query = query_text[:120]
        # Clean the query to prevent 400 Bad Requests on punctuation
        search_query = re.sub(r'[^\w\s]', ' ', search_query)
        search_query = ' '.join(search_query.split())
        
        params = {
            "query": search_query,
            "limit": 1,
            "fields": "title,isRetracted,citationCount"
        }
        
        try:
            # Respect Semantic Scholar public tier rate limit
            time.sleep(0.25)
            
            headers = {
                "User-Agent": "citation-verifier-agent/0.1"
            }

            response = requests.get(BASE_URL, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if SS returned any paper matched
                if "data" in data and len(data["data"]) > 0:
                    paper = data["data"][0]
                    
                    is_retracted = paper.get("isRetracted", False)
                    title = paper.get("title", "")
                    citation_count = paper.get("citationCount", 0)
                    
                    # Compute heuristic risk score
                    # High risk = Retracted paper or 0 citations (highly suspicious/hallucinated)
                    risk_score = 100 if is_retracted else 0
                    if citation_count == 0 and not is_retracted:
                        risk_score += 40
                        
                    result = BroadScreeningResult(
                        claim_id=claim.claim_id,
                        exists=True,
                        retracted=is_retracted,
                        suspicious=(risk_score >= 40 and not is_retracted),
                        resolution_failed=False,
                        api_title_match=title,
                        api_citation_count=citation_count,
                        risk_score=risk_score
                    )
                else:
                    # Semantic Scholar couldn't find any paper matching this text
                    result = BroadScreeningResult(
                        claim_id=claim.claim_id,
                        exists=False,
                        retracted=False,
                        suspicious=False,
                        resolution_failed=False, # Search executed fine, just 0 matches
                        api_title_match="NOT_FOUND",
                        api_citation_count=0,
                        risk_score=0 # Reduced from 60 to 0. Not an integrity risk, just unresolved API.
                    )
            elif response.status_code == 429:
                print(f"[!] Rate-limited by Semantic Scholar on claim {claim.claim_id}")
                result = _generate_fallback(claim.claim_id, risk_score=0)
            else:
                print(f"[!] API Error {response.status_code} on claim {claim.claim_id}")
                result = _generate_fallback(claim.claim_id, risk_score=0)
                
        except Exception as e:
            print(f"[!] Exception fetching metadata for {claim.claim_id}: {str(e)}")
            result = _generate_fallback(claim.claim_id, risk_score=0)
            
        results.append(result)
        
    return results


def _generate_fallback(claim_id: str, risk_score: int) -> BroadScreeningResult:
    """Helper to generate a safe default when API calls fail."""
    return BroadScreeningResult(
        claim_id=claim_id,
        exists=False, 
        retracted=False,
        suspicious=False,
        resolution_failed=True,
        api_title_match="API_FAILURE",
        api_citation_count=0,
        risk_score=risk_score
    )

def fetch_deep_abstracts(claims: List[ExtractedClaim]) -> List[RetrievedEvidence]:
    """
    Stage 2 Deep Retrieval.
    Fetches the actual string content of the abstract for the triaged claims.
    """
    results = []
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    for claim in claims:
        query_text = claim.full_bibliography_entry.strip()
        if not query_text:
            query_text = claim.in_text_claim
            
        search_query = query_text[:120]
        search_query = re.sub(r'[^\w\s]', ' ', search_query)
        search_query = ' '.join(search_query.split())
        
        params = {
            "query": search_query,
            "limit": 1,
            "fields": "title,abstract,externalIds"
        }
        
        time.sleep(0.25) # Avoid aggressive rate-limiting

        headers = {
            "User-Agent": "citation-verifier-agent/0.1"
        }

        try:
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    paper = data["data"][0]
                    abstract = paper.get("abstract")
                    
                    if abstract:
                        results.append(RetrievedEvidence(
                            claim_id=claim.claim_id,
                            found=True,
                            abstract=abstract,
                            doi=paper.get("externalIds", {}).get("DOI")
                        ))
                    else:
                        results.append(RetrievedEvidence(
                            claim_id=claim.claim_id,
                            found=False,
                            doi=paper.get(
                                "externalIds", {}
                            ).get("DOI")
                        ))
                else:
                    results.append(RetrievedEvidence(claim_id=claim.claim_id, found=False))
            else:
                results.append(RetrievedEvidence(claim_id=claim.claim_id, found=False))
        except Exception:
            results.append(RetrievedEvidence(claim_id=claim.claim_id, found=False))
            
    return results

if __name__ == "__main__":
    # Smoke Test
    test_claim = ExtractedClaim(
        claim_id="mock_1",
        in_text_claim="Transformers work well",
        citation_reference="[1]",
        full_bibliography_entry="Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30."
    )
    
    print("Testing Semantic Scholar Broad Screener...")
    res = screen_citations([test_claim])
    print(res[0].model_dump_json(indent=2))
