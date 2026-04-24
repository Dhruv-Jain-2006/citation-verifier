from langgraph.graph import StateGraph, END
from typing import Literal

from core.state import GraphState
from utils.schemas import (
    ExtractedClaim,
    RetrievedEvidence,
    VerificationScore,
    IntegrityReport
)
from tools.pdf_extract import extract_pdf_data
from agents.parser import extract_critical_claims
from agents.retriever import screen_citations, fetch_deep_abstracts
from agents.triage import select_top_risky_claims
from agents.verifier import evaluate_claims

# ==========================================
# 1. NODE SKELETONS (The "Brains" of the pipeline)
# Each node takes the GraphState as input and returns a dictionary
# containing ONLY the keys it wants to update/append to.
# ==========================================

def parse_pdf(state: GraphState) -> dict:
    """
    Extracts the most critical claims and their reference contexts.
    Updates: `claims` and (optionally) `errors`.
    """
    pdf_path = state.get("pdf_path", "")
    print(f"-> Node [parse_pdf]: Extracting claims from {pdf_path}...")
    
    if not pdf_path:
        return {"errors": ["No PDF path provided."]}
        
    try:
        # Step 1: Deterministic PyMuPDF extraction
        extracted_data = extract_pdf_data(pdf_path)
        citation_paragraphs = extracted_data.get("citation_paragraphs", [])
        references_section = extracted_data.get("references_section", "")
        
        # Step 2: Gemini structured claim generation
        if not citation_paragraphs:
            return {
                "errors":["No citation paragraphs extracted."]
            }
        claims = extract_critical_claims(citation_paragraphs, references_section)

        print(f"Extracted {len(claims)} critical claims.")
        
        return {"claims": claims}
        
    except Exception as e:
        return {"errors": [f"parse_pdf failed: {str(e)}"]}

def broad_screen_citations(state: GraphState) -> dict:
    """
    Stage 1: Pings Semantic Scholar for ALL extracted claims.
    Updates: `screening_results`.
    """
    claims = state.get("claims", [])
    print(f"-> Node [broad_screen]: Screening {len(claims)} citations via API...")
    
    results = screen_citations(claims)
    return {"screening_results": results}

def triage_node(state: GraphState) -> dict:
    """
    Ranks claims and promotes the top 5 riskiest to Stage 2.
    Updates: `triaged_claims`.
    """
    claims = state.get("claims", [])
    screens = state.get("screening_results", [])
    
    triaged = select_top_risky_claims(claims, screens)
    print(f"-> Node [triage_node]: Promoted {len(triaged)} claims to Stage 2.")
    return {"triaged_claims": triaged}

def deep_retrieve_evidence(state: GraphState) -> dict:
    """
    Stage 2: Fetches real abstract text ONLY for the triaged claims.
    Updates: `evidence`.
    """
    triaged_claims = state.get("triaged_claims", [])
    print(f"-> Node [deep_retrieve_evidence]: Fetching abstracts for {len(triaged_claims)} claims...")
    
    evidence = fetch_deep_abstracts(triaged_claims)
    return {"evidence": evidence}

def verify_claims(state: GraphState) -> dict:
    """
    Evaluates Claim vs Abstract.
    Updates: `verifications` and (optionally) `errors`.
    """
    claims = state.get("triaged_claims", [])
    evidence = state.get("evidence", [])
    
    print(f"-> Node [verify_claims]: Evaluating ({len(claims)}) claims via Gemini...")
    verifications = evaluate_claims(claims, evidence)
    return {"verifications": verifications}

def critic_review(state: GraphState) -> dict:
    """
    Second-pass override layer. Only runs if conditional routing flags weak claims.
    Updates: `critic_overrides`.
    """
    verifications = state.get("verifications", [])
    # TODO: Invoke gemini-3.1-pro to audit the failing/weak verifications
    
    print("-> Node [critic_review]: Performing 3.1 Pro override review...")
    return {"critic_overrides": []}

def trust_scorer(state: GraphState) -> dict:
    """
    Deterministic python node to calculate the math for the Trust Score.
    No LLM used here.
    Updates: (We will ultimately bundle this into the integrity report or track it separately).
    """
    print("-> Node [trust_scorer]: Calculating deterministic penalties...")
    # TODO: Loop through verifications/critic_overrides and deduct points from 100
    
    return {} # Returning empty dict to prevent breaking GraphState schema before full implementation

def generate_report(state: GraphState) -> dict:
    """
    Final synthesis node compiling the Executive Risk Report.
    Updates: `report`.
    """
    print("-> Node [generate_report]: Synthesizing the final Integrity Report payload...")
    # TODO: Call Gemini to write the `summary` grounded in verified facts.
    # Instantiate final IntegrityReport Pydantic object here.
    
    return {}


# ==========================================
# 2. CONDITIONAL ROUTING LOGIC
# ==========================================

def conditional_critic_routing(state: GraphState) -> Literal["critic_review", "trust_scorer"]:
    """
    Inspects Verifier output. Routes to the Critic ONLY if:
    - support is not "supported"
    - AND/OR evidence is not "strong"
    - AND/OR the LLM flagged a contradiction.
    """
    verifications = state.get("verifications", [])
    
    # If no verifications exist (e.g., error pipeline), default to scorer
    if not verifications:
        return "trust_scorer"
        
    for verification in verifications:
        # Check the conditions based on schemas.VerificationScore
        if (verification.support != "supported" or 
            verification.evidence_strength != "strong" or 
            verification.contradiction_detected == True):
            
            print("=> Route: Weakness detected by Verifier. Routing to Critic...")
            return "critic_review"
            
    print("=> Route: All claims supported strongly. Bypassing Critic...")
    return "trust_scorer"


# ==========================================
# 3. GRAPH COMPILATION (The LangGraph Engine)
# ==========================================

# Initialize the StateGraph with our rigid GraphState schema
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("parse_pdf", parse_pdf)
workflow.add_node("broad_screen", broad_screen_citations)
workflow.add_node("triage", triage_node)
workflow.add_node("deep_retrieve", deep_retrieve_evidence)
workflow.add_node("verify_claims", verify_claims)
workflow.add_node("critic_review", critic_review)
workflow.add_node("trust_scorer", trust_scorer)
workflow.add_node("generate_report", generate_report)

# Define the standard acyclic edges
workflow.add_edge("parse_pdf", "broad_screen")
workflow.add_edge("broad_screen", "triage")
workflow.add_edge("triage", "deep_retrieve")
workflow.add_edge("deep_retrieve", "verify_claims")

# Conditional Edge from Verifier
workflow.add_conditional_edges(
    "verify_claims", # Source Node
    conditional_critic_routing, # Routing function
    { # Mapping the returned string to the specific Node name
        "critic_review": "critic_review",
        "trust_scorer": "trust_scorer"
    }
)

# Close the graph logic
workflow.add_edge("critic_review", "trust_scorer")
workflow.add_edge("trust_scorer", "generate_report")
workflow.add_edge("generate_report", END)

# Set the entry point 
workflow.set_entry_point("parse_pdf")

# Compile the graph into a runnable Langchain architecture
citation_graph = workflow.compile()

if __name__ == "__main__":
    print("\n--- Testing LangGraph Skeleton Initialization ---")
    mock_input = {
        "pdf_path":"mocked.pdf",
        "pdf_text":"Sample text [1]",
        "claims":[],
        "evidence":[],
        "verifications":[],
        "critic_overrides":[],
        "errors":[],
        "report":None
    }
    
    # Running the skeleton
    # It should hit parse -> retrieve -> verify -> conditional_route (returns scorer because list is empty) -> scorer -> report -> end
    for event in citation_graph.stream(mock_input):
        for k, v in event.items():
            if k != "__end__":
                print(f"[{k}] returned state updates")
