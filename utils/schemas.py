from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedClaim(BaseModel):
    """Represents a discrete, testable claim extracted from the document."""
    claim_id: str = Field(
        description="Unique identifier for the claim (e.g., 'claim_1')"
    )
    in_text_claim: str = Field(
        description="The exact text or direct paraphrase of the claim made in the paper."
    )
    citation_reference: str = Field(
        description="The inline citation marker, e.g., '[12]' or '(Smith, 2023)'."
    )
    full_bibliography_entry: str = Field(
        description="The complete bibliographic entry mapped to this citation."
    )

class RetrievedEvidence(BaseModel):
    """Outputs from the Retrieval Node representing external abstract data."""
    claim_id: str = Field(
        description="The ID of the claim this evidence supports."
    )
    found: bool = Field(
        description="Whether the external DB successfully found the abstract."
    )
    abstract: Optional[str] = Field(
        default=None, 
        description="The raw abstract text fetched from the external literature database."
    )
    doi: Optional[str] = Field(
        default=None, 
        description="The DOI of the fetched paper."
    )
    title: Optional[str] = Field(
        default=None,
        description="Retrieved paper title."
    )
    authors: Optional[str] = Field(
        default=None,
        description="Retrieved paper authors."
    )

class VerificationScore(BaseModel):
    """Output from the Verifier/Critic Nodes natively returned by Gemini."""
    claim_id: str = Field(
        description="The ID of the evaluated claim."
    )
    support: str = Field(
        description="The categorical alignment of the claim vs the abstract.",
        pattern="^(supported|partially_supported|unsupported|unverifiable)$"
    )
    evidence_strength: str = Field(
        description="Qualitative assessment of the cited abstract's relevance.",
        pattern="^(strong|moderate|weak|none)$"
    )
    contradiction_detected: bool = Field(
        description="True if the abstract explicitly contradicts the author's claim."
    )
    reasoning: str = Field(
        description="A concise one-sentence justification for the output scores."
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score from 0 to 100."
    )

class IntegrityReport(BaseModel):
    """Final output compilation representing the system's payload."""
    trust_score: int = Field(
        description="0-100 integer score calculating the overall paper reliability."
    )
    summary: str = Field(
        description="A trusted executive summary grounded only in verified evidence."
    )
    high_risk_claims: List[str] = Field(
        description="List of claim IDs that failed verification or contradicted."
    )
