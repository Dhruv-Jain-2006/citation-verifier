"""
agents/retriever.py
Multi-source API client optimized for Semantic Scholar with an OpenAlex fallback.
Implements:
  - Jittered exponential backoff for 429 rate limits
  - In-memory NormalizedPaper cache to prevent redundant API calls
  - Title-centric query construction
  - OpenAlex abstract_inverted_index reconstruction
"""
import time
import random
import re
from typing import List, Optional, Dict, Any

import requests

from utils.schemas import ExtractedClaim, BroadScreeningResult, RetrievedEvidence

# ────────────────────────────────────────────────────────────────
# Constants & Unified Cache
# ────────────────────────────────────────────────────────────────

SS_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OA_BASE_URL = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "citation-verifier-agent/0.3; mailto:bot@example.com"}

# Normalized cache keyed by query string. 
# Stores dict: {'title': str, 'retracted': bool, 'citations': int, 'abstract': str, 'doi': str, 'source': str}
_QUERY_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

INTER_REQUEST_DELAY = 1.2
_429_BACKOFF = [1.0, 3.0, 7.0]

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _build_query(claim: ExtractedClaim) -> str:
    """
    Prefer the LLM-extracted title. If missing or empty, gracefully fallback 
    to the highly focused in_text_claim. Avoid throwing raw bibliography strings 
    at the API to reduce noise and missed hits.
    """
    raw = getattr(claim, "cited_paper_title", "").strip()
    if not raw:
        raw = claim.in_text_claim.strip()
        
    raw = raw[:120]
    raw = re.sub(r"[^\w\s]", " ", raw)
    return " ".join(raw.split())

def _reconstruct_oa_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstructs OpenAlex's abstract_inverted_index into a flat string."""
    if not inverted_index:
        return None
    
    # Find the maximum index to size our array
    try:
        max_idx = max(pos for positions in inverted_index.values() for pos in positions)
        words = [""] * (max_idx + 1)
        
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
                
        return " ".join(words)
    except Exception as e:
        print(f"    [OpenAlex] Failed to reconstruct abstract: {e}")
        return None

def _fetch_semantic_scholar(query: str) -> Optional[Dict[str, Any]]:
    """Attempts to fetch from Semantic Scholar with 429 backoff."""
    params = {"query": query, "limit": 1, "fields": "title,isRetracted,citationCount,abstract,externalIds"}
    
    for attempt, wait in enumerate(_429_BACKOFF):
        try:
            response = requests.get(SS_BASE_URL, params=params, headers=HEADERS, timeout=12)
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    return None
                    
                paper = data[0]
                return {
                    "title": paper.get("title", ""),
                    "retracted": paper.get("isRetracted", False),
                    "citations": paper.get("citationCount", 0),
                    "abstract": paper.get("abstract"),
                    "doi": paper.get("externalIds", {}).get("DOI"),
                    "source": "SemanticScholar"
                }
                
            elif response.status_code == 429:
                sleep_s = float(response.headers.get("Retry-After", wait)) * random.uniform(0.8, 1.2)
                print(f"    [SS 429] Waiting {sleep_s:.1f}s (attempt {attempt + 1})...")
                time.sleep(sleep_s)
                continue
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.Timeout):
                time.sleep(wait * random.uniform(0.8, 1.2))
                continue
            return None
            
    return None

def _fetch_openalex(query: str) -> Optional[Dict[str, Any]]:
    """Fallback fetch from OpenAlex."""
    params = {"search": query, "per-page": 3}
    
    try:
        response = requests.get(OA_BASE_URL, params=params, headers=HEADERS, timeout=12)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if not results:
                return None
                
            # Minimal safety gate: prefer the result whose title has the best word overlap with the query
            query_words = set(query.lower().split())
            best_paper = results[0]
            best_overlap = 0
            
            for p in results:
                title_words = set((p.get("title") or "").lower().split())
                if not title_words:
                    continue
                overlap = len(query_words.intersection(title_words)) / max(1, len(query_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_paper = p
                    
                # If we find a near-perfect match, stop looking
                if overlap > 0.8:
                    break
                    
            paper = best_paper
            abstract_index = paper.get("abstract_inverted_index")
            abstract = _reconstruct_oa_abstract(abstract_index)
            
            return {
                "title": paper.get("title", ""),
                "retracted": paper.get("is_retracted", False),
                "citations": paper.get("cited_by_count", 0),
                "abstract": abstract,
                "doi": paper.get("doi"),
                "source": "OpenAlex"
            }
    except requests.exceptions.RequestException as e:
        print(f"    [OpenAlex Error] {e}")
        
    return None

def _fetch_paper_normalized(query: str) -> Optional[Dict[str, Any]]:
    """
    Unified multi-source fetch. Uses cache heavily.
    Tries SS, falls back to OpenAlex.
    """
    if query in _QUERY_CACHE:
        return _QUERY_CACHE[query]
        
    time.sleep(INTER_REQUEST_DELAY)
    
    # 1. Primary: Semantic Scholar
    paper = _fetch_semantic_scholar(query)
    
    # 2. Fallback: OpenAlex (if SS failed or returned 0 results)
    if not paper:
        print(f"    [!] SS failed/empty for '{query[:40]}...'. Falling back to OpenAlex.")
        time.sleep(INTER_REQUEST_DELAY)
        paper = _fetch_openalex(query)
        
    # Cache the result (even if None, to prevent redundant re-fetching of bad queries)
    _QUERY_CACHE[query] = paper
    return paper

# ────────────────────────────────────────────────────────────────
# Graph Nodes
# ────────────────────────────────────────────────────────────────

def screen_citations(claims: List[ExtractedClaim]) -> List[BroadScreeningResult]:
    """Stage 1 lightweight screener."""
    results: List[BroadScreeningResult] = []
    
    for claim in claims:
        query = _build_query(claim)
        paper = _fetch_paper_normalized(query)
        
        if paper is None:
            # Complete API failure across both sources
            results.append(BroadScreeningResult(
                claim_id=claim.claim_id,
                exists=False,
                retracted=False,
                suspicious=False,
                resolution_failed=True,
                api_title_match="API_FAILURE",
                api_citation_count=0,
                risk_score=0
            ))
            continue
            
        # Calculate heuristic risk
        risk_score = 100 if paper["retracted"] else 0
        if paper["citations"] == 0 and not paper["retracted"]:
            risk_score += 40
            
        results.append(BroadScreeningResult(
            claim_id=claim.claim_id,
            exists=True,
            retracted=paper["retracted"],
            suspicious=(risk_score >= 40 and not paper["retracted"]),
            resolution_failed=False,
            api_title_match=paper["title"][:50] + "..." if paper["title"] else "UNKNOWN",
            api_citation_count=paper["citations"],
            risk_score=risk_score
        ))
        
    return results

def fetch_deep_abstracts(claims: List[ExtractedClaim]) -> List[RetrievedEvidence]:
    """
    Stage 2 deep retrieval.
    Reuses the _QUERY_CACHE populated during Stage 1.
    """
    results: List[RetrievedEvidence] = []
    
    for claim in claims:
        query = _build_query(claim)
        
        # This will hit cache immediately if Stage 1 saw this query
        paper = _fetch_paper_normalized(query)
        
        if paper is None:
            results.append(RetrievedEvidence(claim_id=claim.claim_id, found=False))
            continue
            
        abstract = paper.get("abstract")
        doi = paper.get("doi")
        
        if abstract:
            results.append(RetrievedEvidence(
                claim_id=claim.claim_id,
                found=True,
                abstract=abstract,
                doi=doi,
                title=paper.get("title")
            ))
        else:
            results.append(RetrievedEvidence(
                claim_id=claim.claim_id,
                found=False,
                doi=doi,
                title=paper.get("title")
            ))
            
    return results

if __name__ == "__main__":
    # Smoke Test
    test = ExtractedClaim(
        claim_id="mock_1",
        in_text_claim="Transformers outperform RNNs on translation tasks.",
        citation_reference="[1]",
        full_bibliography_entry="Vaswani, A., Shazeer, N., Parmar, N. (2017). Attention is all you need.",
        cited_paper_title="Attention is all you need"
    )
    print("Testing Normalized Multi-Source Retriever...")
    res = screen_citations([test])
    print(res[0].model_dump_json(indent=2))
    
    abs_res = fetch_deep_abstracts([test])
    print(abs_res[0].model_dump_json(indent=2))
