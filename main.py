from agents.verifier import (
    verify_claim_support,
    compute_trust_score
)

claim = """
Transformers reduce recurrence costs
compared to recurrent networks.
"""

abstract = """
The Transformer removes recurrence and
allows significantly greater parallelization.
"""

result = verify_claim_support(
    claim,
    abstract
)

score = compute_trust_score(result)

print(result)
print("Trust Score:", score)