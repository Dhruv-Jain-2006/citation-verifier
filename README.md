# 📄 Citation Integrity Auditor

An AI-powered system designed to **analyze research papers and verify citation reliability**.
The system detects weak, unverifiable, or potentially misleading references and assigns a **trust score** based on evidence-backed validation.

---

## 🚀 Overview

This project implements a **multi-stage AI pipeline** that:

* Extracts key claims from a research paper
* Retrieves supporting evidence from academic sources
* Verifies claims using LLM-based reasoning
* Applies deterministic scoring with penalties
* Outputs a structured **Citation Integrity Report**

---

## 🧠 Key Features

* 🔍 **Multi-Agent Architecture**

  * Parser → Screening → Triage → Verifier → Critic → Scorer

* 📊 **Trust Score (0–100)**

  * Quantifies overall citation reliability

* ⚠️ **Penalty-Based Scoring**

  * Penalizes weak, unsupported, or unverifiable claims

* ❓ **Uncertainty Detection**

  * Flags API failures, missing abstracts, or incomplete evidence

* 📄 **PDF Upload Support**

  * Analyze any research paper via API

* 🌐 **Frontend-Ready Output**

  * Clean JSON designed for dashboard visualization

---

## 🏗️ System Architecture

```
PDF Input
   ↓
[ Parser ] → Extract Claims
   ↓
[ Broad Screening ] → Validate citations
   ↓
[ Triage ] → Select high-risk claims
   ↓
[ Evidence Retrieval ]
   ↓
[ Verifier (LLM) ]
   ↓
[ Critic (Override Logic) ]
   ↓
[ Trust Scorer ]
   ↓
[ Final Report ]
```

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **AI/LLM:** Gemini (Google Generative AI)
* **Orchestration:** LangGraph
* **PDF Processing:** PyMuPDF
* **Data Handling:** Python, Pydantic

---

## Live Site
The site is live at https://scholarlearn-ai.netlify.app/
---

## 🛠️ Installation

```bash
git clone https://github.com/Dhruv-Jain-2006/citation-verifier.git
cd citation-verifier

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Running the Backend

```bash
uvicorn app.api:app --reload
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Usage

### Endpoint

```
POST /audit
```

### Request

* Type: `multipart/form-data`
* Field: `file`
* Upload: PDF research paper

---

### Example Response

```json
{
  "status": "success",
  "data": {
    "paper": {
      "title": "Attention Is All You Need",
      "domain": "Machine Learning"
    },
    "trust": {
      "score": 92,
      "verdict": "High Integrity"
    },
    "verification": {
      "supported": 1,
      "partial": 1,
      "unsupported": 0,
      "unverifiable": 1
    },
    "penalties": [...],
    "uncertainty": [...],
    "high_risk_claims": [],
    "summary": "..."
  }
}
```

---

## 📊 Output Explanation

* **Trust Score** → Overall citation reliability
* **Verification Breakdown** → Claim support levels
* **Penalties** → Reasons for score reduction
* **Uncertainty Log** → Missing or unverifiable evidence
* **High-Risk Claims** → Potentially problematic references

---

## ⚠️ Limitations

* Depends on availability of external abstracts/APIs
* LLM-based reasoning may introduce minor variability
* Some citations may remain unverifiable due to missing data

---

## 📌 Future Improvements

* Domain detection (automatic classification)
* Support for additional academic databases
* Improved retrieval robustness
* UI dashboard integration

---

## 👨‍💻 Author

**Dhruv Jain**

---

## ⭐ Final Note

This project focuses on **verifiability, transparency, and reliability** in AI systems—
moving beyond generation to **evidence-backed reasoning**.
