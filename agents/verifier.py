import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Use Flash for development.
# Later you can switch to gemini-3.1-pro-preview
model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


def verify_claim_support(claim, abstract):

    prompt = f"""
You are a skeptical scientific citation verification judge.

Task:
Determine whether the cited abstract supports the claim.

Use this rubric:

supported:
Every key concept in the claim is explicitly supported
by the abstract, with no inference required.

partially_supported:
Any claim requiring inference, semantic interpretation,
or containing unsupported concepts belongs here.

unsupported:
The claim is absent or contradicted.

Rules:
- Be conservative.
- Only mark supported if claim wording is explicitly supported.
- If support depends on inference or interpretation,
  return partially_supported.
- Do not infer unstated concepts such as costs,
  efficiency, or performance unless explicitly stated.
- If claim contradicts abstract, return:
  support="unsupported"
  contradiction_detected=true

Claim:
{claim}

Abstract:
{abstract}

Return ONLY valid JSON.
No markdown.
No explanations outside JSON.

Use EXACT schema:

{{
"support":"supported|partially_supported|unsupported",
"confidence": integer from 0 to 100 only,
"evidence_strength":"strong|moderate|weak",
"contradiction_detected": true/false,
"reasoning":"one concise sentence"
}}
"""

    response = model.generate_content(prompt)

    text = response.text.strip()

    # Remove accidental markdown fences
    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text)

    except Exception:
        return {
            "support":"parse_error",
            "confidence":0,
            "evidence_strength":"weak",
            "contradiction_detected":False,
            "reasoning":text
        }


def compute_trust_score(result):

    score = result["confidence"]

    if result["evidence_strength"] == "weak":
        score -= 20

    elif result["evidence_strength"] == "moderate":
        score -= 10

    if result["contradiction_detected"]:
        score -= 30

    return max(score,0)