import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def verify_claim_support(claim, abstract):

    prompt = f"""
You are an academic citation verification agent.

Determine whether the abstract supports the claim.

Claim:
{claim}

Abstract:
{abstract}

Return ONLY valid JSON:

{{
 "support":"supported|partially_supported|unsupported",
 "confidence":0-100,
 "reasoning":"one sentence explanation"
}}
"""

    response = model.generate_content(prompt)

    return response.text