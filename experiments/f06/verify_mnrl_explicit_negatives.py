"""API smoke check for explicit negatives in the same Sentence-Transformers MNRL."""
import json
from pathlib import Path

import torch
from torch import nn
from sentence_transformers import losses


class DummyModel(nn.Module):
    pass


torch.manual_seed(13)
loss = losses.MultipleNegativesRankingLoss(model=DummyModel())
batch = 4
dim = 16
query = torch.randn(batch, dim, requires_grad=True)
positive = torch.randn(batch, dim, requires_grad=True)
hard_negative = torch.randn(batch, dim, requires_grad=True)

# Control: in-batch negatives only; treatment: one explicit hard-negative column.
# Both calls use the identical loss object/configuration.
control_loss = loss.compute_loss_from_embeddings([query, positive], labels=None)
treatment_loss = loss.compute_loss_from_embeddings(
    [query, positive, hard_negative], labels=None
)
assert torch.isfinite(control_loss).item()
assert torch.isfinite(treatment_loss).item()
treatment_loss.backward()
assert query.grad is not None and torch.isfinite(query.grad).all().item()
assert positive.grad is not None and torch.isfinite(positive.grad).all().item()
assert hard_negative.grad is not None and torch.isfinite(hard_negative.grad).all().item()

result = {
    "status": "PASS",
    "sentence_transformers": __import__("sentence_transformers").__version__,
    "loss_class": type(loss).__name__,
    "control_inputs": 2,
    "treatment_inputs": 3,
    "control_finite_loss": float(control_loss.detach()),
    "treatment_finite_loss": float(treatment_loss.detach()),
    "treatment_gradients_finite": True,
    "scope": "API smoke test only; no ESAW data, model training, or performance claim",
}
out = Path(__file__).with_name("mnrl_api_smoke.json")
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
