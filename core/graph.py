from langgraph.graph import StateGraph, END
from typing import Literal

from core.state import GraphState
from utils.schemas import (
    ExtractedClaim,
    RetrievedEvidence,
    VerificationScore,
    IntegrityReport
)

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
    pdf_text = state.get("pdf_text", "")
    # TODO: Implement PyMuPDF regex logic
    # TODO: Call gemini-2.5-flash with structured output to get claims
    
    print("-> Node [parse_pdf]: Extracting claims...")
    return {"claims": []} # Returns empty list structure for now

def retrieve_evidence(state: GraphState) -> dict:
    """
    Fetches real abstracts for the citations mapped to the extracted claims.
    Updates: `evidence` and (optionally) `errors`.
    """
    claims = state.get("claims", [])
    # TODO: Map over claims and hit Semantic Scholar / Crossref APIs
    
    print(f"-> Node [retrieve_evidence]: Fetching DOIs for {len(claims)} claims...")
    return {"evidence": []}

def verify_claims(state: GraphState) -> dict:
    """
    Evaluates Claim vs Abstract.
    Updates: `verifications` and (optionally) `errors`.
    """
    claims = state.get("claims", [])
    evidence = state.get("evidence", [])
    # TODO: Invoke gemini-2.5-flash Verifier prompt leveraging Pydantic VerificationScore
    
    print("-> Node [verify_claims]: Evaluating evidence...")
    return {"verifications": []}

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
workflow.add_node("retrieve_evidence", retrieve_evidence)
workflow.add_node("verify_claims", verify_claims)
workflow.add_node("critic_review", critic_review)
workflow.add_node("trust_scorer", trust_scorer)
workflow.add_node("generate_report", generate_report)

# Define the standard acyclic edges
workflow.add_edge("parse_pdf", "retrieve_evidence")
workflow.add_edge("retrieve_evidence", "verify_claims")

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
