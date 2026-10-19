from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from model.graph import GraphEncoder
from model.contrast import BertModel


class PathGuidedClassifier(nn.Module):
    """Executable reconstruction of HTC-CLIP Equations 3--8.

    The paper applies ReLU to every class-level activation and then sigmoid.
    Taken literally, that makes every class probability at least 0.5 and the
    all-ones root boundary makes every root probability at least sigmoid(1).
    Those equations cannot produce the paper's threshold-0.5 F1.  The default
    therefore treats class-level outputs as logits and retains ReLU only in the
    hidden parent-to-child transform.  ``literal_paper=True`` is provided to
    reproduce and test the printed inconsistency explicitly.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        depth2label: dict[int, list[int]],
        intermediate: int = 128,
        literal_paper: bool = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.depth2label = depth2label
        self.literal_paper = literal_paper
        self.direct = nn.ModuleDict({str(d): nn.Linear(hidden_size, len(nodes)) for d, nodes in depth2label.items()})
        self.previous = nn.ModuleDict(
            {
                str(d): nn.Sequential(nn.Linear(len(depth2label[d - 1]), intermediate), nn.ReLU(), nn.Linear(intermediate, len(nodes)))
                for d, nodes in depth2label.items()
                if d > 0
            }
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        per_depth: list[torch.Tensor] = []
        previous_pooled: torch.Tensor | None = None
        for depth in sorted(self.depth2label):
            pooled_activation = self.direct[str(depth)](pooled)
            if previous_pooled is None:
                activation = F.relu(pooled_activation) + 1.0 if self.literal_paper else pooled_activation
            else:
                inherited = self.previous[str(depth)](previous_pooled)
                if self.literal_paper:
                    activation = F.relu(pooled_activation) + F.relu(inherited)
                else:
                    activation = pooled_activation + F.relu(inherited)
            per_depth.append(activation)
            previous_pooled = F.relu(pooled_activation)
        output = pooled.new_zeros((pooled.shape[0], self.num_labels))
        for depth, values in zip(sorted(self.depth2label), per_depth):
            output[:, self.depth2label[depth]] = values
        return output


class HtcClip(nn.Module):
    """HTC-CLIP: HGCLR positive construction + linear/path classifiers."""

    def __init__(
        self,
        base_model: str,
        num_labels: int,
        depth2label: dict[int, list[int]],
        label_names: list[str],
        graph_data_path: str,
        gamma: float = 0.02,
        contrastive_weight: float = 0.05,
        temperature: float = 1.0,
        path_hidden: int = 128,
        literal_paper_path: bool = False,
    ):
        super().__init__()
        pretrained = AutoModel.from_pretrained(base_model)
        self.encoder = BertModel.from_pretrained(base_model)
        # ``model.contrast.BertModel`` is the HGCLR encoder implementation.  It
        # exposes the pre-LayerNorm word embeddings needed by the differentiable
        # token selector, while retaining the published BERT weights.
        self.encoder.load_state_dict(pretrained.state_dict(), strict=False)
        del pretrained
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.linear = nn.Linear(hidden, num_labels)
        self.path = PathGuidedClassifier(hidden, num_labels, depth2label, path_hidden, literal_paper_path)
        self.graph_encoder = GraphEncoder(
            self.encoder.config,
            graph=True,
            layer=1,
            data_path=graph_data_path,
            threshold=gamma,
            tau=temperature,
        )
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        # Match the BERT initialization that the HGCLR ``from_pretrained``
        # implementation applies to all task-specific modules.
        self.linear.apply(self.encoder._init_weights)
        self.path.apply(self.encoder._init_weights)
        self.graph_encoder.apply(self.encoder._init_weights)
        self.projector.apply(self.encoder._init_weights)
        self.gamma = gamma
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.label_names = label_names

    def positive_weights(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, word_embeddings = self.encoder.embeddings(input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids))

        def embed_label_ids(label_ids: torch.Tensor) -> torch.Tensor:
            return self.encoder.embeddings(input_ids=label_ids)[0]

        return self.graph_encoder(word_embeddings, attention_mask, labels, embed_label_ids)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        embedding_weight: torch.Tensor | None = None,
    ):
        if embedding_weight is None:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        else:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                embedding_weight=embedding_weight,
                return_dict=True,
            )
        return self.dropout(output.last_hidden_state[:, 0]), output.last_hidden_state

    def heads(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(pooled), self.path(pooled)

    def nt_xent(self, original: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.projector(torch.cat([original, positive], dim=0)), dim=-1)
        logits = z @ z.T / self.temperature
        logits.fill_diagonal_(-1e9)
        batch = original.shape[0]
        target = torch.arange(2 * batch, device=z.device)
        target = (target + batch) % (2 * batch)
        return F.cross_entropy(logits, target)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        pooled, _ = self.encode(input_ids, attention_mask, token_type_ids)
        linear_logits, path_logits = self.heads(pooled)
        result = {"linear_logits": linear_logits, "path_logits": path_logits, "logits": torch.maximum(linear_logits, path_logits)}
        if labels is None:
            return result
        weights = self.positive_weights(input_ids, labels, attention_mask)
        positive, _ = self.encode(input_ids, attention_mask, token_type_ids, weights)
        positive_linear, positive_path = self.heads(positive)
        bce = nn.BCEWithLogitsLoss()
        result["loss"] = (
            bce(linear_logits, labels)
            + bce(path_logits, labels)
            + bce(positive_linear, labels)
            + bce(positive_path, labels)
            + self.contrastive_weight * self.nt_xent(pooled, positive)
        )
        return result
