import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

from utils.schemas import PaperSummary

# Ensure environment variables are loaded for the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_paper_summary(pdf_text: str) -> PaperSummary:
    """
    Auxiliary module that generates a concise, technical, and structured summary
    of the entire research paper using Gemini 2.5 Flash.
    Operates independently from the citation trust pipeline.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # We truncate the input text to roughly 20,000 chars to save tokens and latency,
    # as the core problem, methodology, and results are almost always in the first few pages.
    # The abstract and intro alone usually provide everything needed for this high-level summary.
    safe_text = pdf_text[:12000] + "\n\n...[TRUNCATED]...\n\n" + pdf_text[-8000:]
    
    prompt = f"""
    You are an expert AI researcher and technical summarizer.
    Read the following excerpt from a research paper and extract a structured summary.
    Keep the summary highly technical, concise, and useful for a senior scientist.
    
    Paper Text Excerpt:
    \"\"\"
    {safe_text}
    \"\"\"
    
    Requirements:
    1. Problem Statement: 1-2 sentences on what gap this paper solves.
    2. Core Methodology: Brief summary of the architecture, experiment, or math introduced.
    3. Key Contributions: Exactly 3 bullet points.
    4. Main Results: Quantitative or qualitative outcome.
    5. Limitations: Any constraints, caveats, or future work mentioned. If none, write 'None mentioned'.
    
    Return the result strictly conforming to the JSON schema.
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=PaperSummary,
                temperature=0.0
            )
        )
        
        data = json.loads(response.text)
        return PaperSummary(**data)
        
    except Exception as e:
        print(f"[Summarizer Error] Failed to generate paper summary: {e}")
        return PaperSummary(
            problem_statement="Generation failed.",
            core_methodology="Generation failed.",
            key_contributions=["Error", "Error", "Error"],
            main_results="Generation failed.",
            limitations="Generation failed."
        )

if __name__ == "__main__":
    print("Auxiliary Paper Summarizer initialized.")
