from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from model.contrast import BertModel
from model.graph import GraphEncoder


class Hgclr(nn.Module):
    """HGCLR using the authors' Graphormer-guided positive construction."""

    def __init__(
        self,
        base_model: str,
        num_labels: int,
        graph_data_path: str,
        gamma: float = 0.02,
        contrastive_weight: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        pretrained = AutoModel.from_pretrained(base_model)
        self.encoder = BertModel.from_pretrained(base_model)
        self.encoder.load_state_dict(pretrained.state_dict(), strict=False)
        del pretrained
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.graph_encoder = GraphEncoder(
            self.encoder.config,
            graph=True,
            layer=1,
            data_path=graph_data_path,
            threshold=gamma,
            tau=temperature,
        )
        self.projector = nn.Sequential(
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.encoder.config.hidden_dropout_prob),
            nn.Linear(hidden, hidden),
        )
        # ``ContrastModel.from_pretrained`` initializes every newly attached
        # module with BERT's normal(0, initializer_range) rule before loading
        # the pretrained encoder.  These modules live outside the
        # ``BertPreTrainedModel`` wrapper in this compatibility port, so apply
        # that same rule explicitly rather than retaining PyTorch's default
        # Kaiming initialization.
        self.classifier.apply(self.encoder._init_weights)
        self.graph_encoder.apply(self.encoder._init_weights)
        self.projector.apply(self.encoder._init_weights)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def positive_weights(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # The authors reuse raw word embeddings from the original view.
        # A second full embedding call draws an unused dropout mask and shifts
        # the stochastic positive construction. Preserve the author RNG order.
        word_embeddings = self.encoder.embeddings.word_embeddings(input_ids)

        def embed_label_ids(label_ids: torch.Tensor) -> torch.Tensor:
            return self.encoder.embeddings(input_ids=label_ids)[0]

        return self.graph_encoder(word_embeddings, attention_mask, labels, embed_label_ids)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        embedding_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            embedding_weight=embedding_weight,
            return_dict=True,
        )
        return self.dropout(output.last_hidden_state[:, 0])

    def nt_xent(self, original: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.projector(torch.cat([original, positive], dim=0)), dim=-1)
        logits = z @ z.T / self.temperature
        logits.fill_diagonal_(-1e9)
        batch = original.shape[0]
        target = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
        return F.cross_entropy(logits, target)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self.encode(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled)
        result = {"logits": logits}
        if labels is None:
            return result
        weights = self.positive_weights(input_ids, labels, attention_mask)
        positive = self.encode(input_ids, attention_mask, token_type_ids, weights)
        positive_logits = self.classifier(positive)
        bce = nn.BCEWithLogitsLoss()
        result["loss"] = (
            bce(logits, labels)
            + bce(positive_logits, labels)
            + self.contrastive_weight * self.nt_xent(pooled, positive)
        )
        return result
