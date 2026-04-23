from agents.verifier import verify_claim_support

claim = """
Transformers reduce recurrence costs compared to RNNs.
"""

abstract = """
The Transformer removes recurrence and allows
significantly more parallelization.
"""

result = verify_claim_support(
    claim,
    abstract
)

print(result)