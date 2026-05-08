from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import google.generativeai as genai

from core.graph import citation_graph
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------
# App Initialization
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------------
# Helper: Transform Output
# -------------------------
def to_dashboard(result):
    # Helper to safely read attributes or dict keys
    def read(x, key, default=None):
        if isinstance(x, dict):
            return x.get(key, default)
        return getattr(x, key, default)

    verifications = read(result, "verifications", [])

    # Verification counts
    verification_counts = {
        "supported": sum(1 for v in verifications if read(v, "support") == "supported"),
        "partial": sum(1 for v in verifications if read(v, "support") == "partially_supported"),
        "unsupported": sum(1 for v in verifications if read(v, "support") == "unsupported"),
        "unverifiable": sum(1 for v in verifications if read(v, "support") == "unverifiable"),
    }

    score_data = read(result, "score_data")
    report = read(result, "report")

    score = read(score_data, "trust_score")

    if score >= 90:
        verdict = "High Integrity"
    elif score >= 70:
        verdict = "Moderate Integrity"
    else:
        verdict = "Low Integrity"

    penalties = read(score_data, "penalties", [])

    # Uncertainty extraction
    seen = set()
    uncertainty = []

    for p in penalties:
        if "[System Uncertainty]" in read(p, "reason", ""):
            key = (read(p, "claim_id"), read(p, "reason"))
            if key not in seen:
                seen.add(key)
                uncertainty.append(p)

    pdf_path = read(result, "pdf_path", "")
    filename = os.path.basename(pdf_path).replace(".pdf", "")

    # Remove UUID prefix
    if "_" in filename:
        title = filename.split("_", 1)[1]
    else:
        title = filename
    domain = "Unknown"

    return {
        "paper": {
            "title": title,
            "domain": domain
        },
        "trust": {
            "score": score,
            "verdict": verdict
        },
        "verification": verification_counts,
        "penalties": penalties,
        "uncertainty": uncertainty,
        "high_risk_claims": read(score_data, "high_risk_claims", []),
        "summary": read(report, "summary"),
        "call_counts": read(result, "call_counts", {})
    }


# -------------------------
# API Endpoint
# -------------------------
@app.post("/audit")
async def run_audit(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            return {
                "status": "error",
                "message": "Only PDF files are allowed"
            }

        # Save uploaded file
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run pipeline
        result = citation_graph.invoke({
            "pdf_path": file_path
        })

        # Transform for frontend
        dashboard_data = to_dashboard(result)

        return {
            "status": "success",
            "data": dashboard_data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
