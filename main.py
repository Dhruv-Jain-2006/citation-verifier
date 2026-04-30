import os
import urllib.request
from core.graph import citation_graph
from agents.summarizer import generate_paper_summary
from tools.pdf_extract import extract_pdf_data

ENABLE_SUMMARIZER = False

def download_sample_paper(filepath: str):
    print(f"Downloading sample paper to {filepath}...")
    url = "https://arxiv.org/pdf/1706.03762.pdf" # Attention is All You Need
    urllib.request.urlretrieve(url, filepath)
    print("Download complete.")

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Get the first pdf in data/
    pdfs = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not pdfs:
        target_pdf = os.path.join(data_dir, "sample_paper.pdf")
        download_sample_paper(target_pdf)
        pdfs = ["sample_paper.pdf"]
        
    pdf_path = os.path.join(data_dir, pdfs[0])
    
    print(f"\n{'='*50}\nSTARTING END-TO-END CITATION AUDIT ON:\n{pdf_path}\n{'='*50}\n")
    
    initial_state = {
        "pdf_path": pdf_path,
        "pdf_text": "",
        "claims": [],
        "screening_results": [],
        "triaged_claims": [],
        "evidence": [],
        "verifications": [],
        "critic_overrides": [],
        "errors": [],
        "score_data": {},
        "report": None,
        "llm_quota_exhausted": False,
        "call_counts": {
            "parser_calls": 0,
            "verifier_calls": 0,
            "critic_calls": 0,
            "retry_calls": 0
        }
    }
    
    # LangGraph returns a stream of dictionaries mapping NodeName -> StateUpdates
    final_report = None
    
    for event in citation_graph.stream(initial_state, {"recursion_limit": 50}):
        for node_name, updates in event.items():
            print(f"\n[Stream Update] -> Finished Node: {node_name}")
            if "report" in updates:
                final_report = updates["report"]
            if "pdf_text" in updates:
                initial_state["pdf_text"] = updates["pdf_text"]
            if "errors" in updates and updates["errors"]:
                print("[Node Errors]", updates["errors"])
            if "call_counts" in updates:
                for k, v in updates["call_counts"].items():
                    initial_state["call_counts"][k] = initial_state["call_counts"].get(k, 0) + v
            if "llm_quota_exhausted" in updates:
                initial_state["llm_quota_exhausted"] = updates["llm_quota_exhausted"]

            
    print(f"\n\n{'='*50}\nFINAL SYSTEM OUTPUT\n{'='*50}")
    
    if final_report:
        score_str = f"{final_report.trust_score}/100" if final_report.trust_score is not None else "N/A (Audit Failed)"
        print(f"TRUST SCORE: {score_str}")
        
        print("\n--- EXECUTIVE SUMMARY ---")
        print(final_report.summary)
        
        if final_report.high_risk_claims:
            print("\n--- HIGH RISK CLAIMS FLAGGED ---")
            for c in final_report.high_risk_claims:
                print(f"- {c}")

        print("\n--- LLM CALL ACCOUNTING ---")
        counts = initial_state["call_counts"]
        print(f"parser_calls:   {counts.get('parser_calls', 0)}")
        print(f"verifier_calls: {counts.get('verifier_calls', 0)}")
        print(f"critic_calls:   {counts.get('critic_calls', 0)}")
        print(f"retry_calls:    {counts.get('retry_calls', 0)}")
        total = sum(counts.values())
        print(f"total_calls:    {total}")


        if ENABLE_SUMMARIZER:        
            # --- TEST NEW AUXILIARY SUMMARIZER ---
            print("\n\n==================================================")
            print("GENERATING AUXILIARY PAPER SUMMARY")
            print("==================================================")
            
            # Extract text directly for the auxiliary module
            raw_data = extract_pdf_data(pdf_path)
            pdf_text = raw_data.get("full_text", "")
            
            if pdf_text:
                print("This may take a few seconds...")
                summary = generate_paper_summary(pdf_text)
                print("\n[PROBLEM STATEMENT]")
                print(summary.problem_statement)
                print("\n[CORE METHODOLOGY]")
                print(summary.core_methodology)
                print("\n[KEY CONTRIBUTIONS]")
                for c in summary.key_contributions:
                    print(f"- {c}")
                print("\n[MAIN RESULTS]")
                print(summary.main_results)
                print("\n[LIMITATIONS]")
                print(summary.limitations)
            else:
                print("[!] Could not extract PDF text for summarization.")
            
    else:
        print("\n[!] Pipeline failed to generate a final report.")

if __name__ == "__main__":
    main()