from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


def _global_logits(per_depth: list[torch.Tensor], depth2label: dict[int, list[int]], num_labels: int) -> torch.Tensor:
    output = per_depth[0].new_zeros((per_depth[0].shape[0], num_labels))
    for depth, values in zip(sorted(depth2label), per_depth):
        output[:, depth2label[depth]] = values
    return output


class HydraModel(nn.Module):
    """Official HYDRA architecture, with global indexing restored for evaluation."""

    def __init__(
        self,
        base_model: str,
        num_labels: int,
        depth2label: dict[int, list[int]],
        architecture: str = "local",
        project_embedding: bool = True,
    ) -> None:
        super().__init__()
        if architecture not in {"local", "local_global", "local_nested"}:
            raise ValueError(architecture)
        self.encoder = AutoModel.from_pretrained(base_model)
        self.num_labels = num_labels
        self.depth2label = depth2label
        self.architecture = architecture
        self.num_levels = len(depth2label)
        hidden = self.encoder.config.hidden_size
        self.projector = nn.Linear(hidden, hidden * self.num_levels) if project_embedding else nn.Identity()
        embedding_size = hidden * self.num_levels if project_embedding else hidden
        self.local_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_size, embedding_size * 2),
                nn.LayerNorm(embedding_size * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embedding_size * 2, len(depth2label[depth])),
            )
            for depth in sorted(depth2label)
        )
        if architecture == "local_global":
            self.auxiliary = nn.Sequential(
                nn.Linear(embedding_size, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden * 2, num_labels),
            )
        elif architecture == "local_nested":
            self.auxiliary = nn.Sequential(
                nn.Linear(num_labels, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden * 2, num_labels),
            )
        else:
            self.auxiliary = None

    def forward(self, input_ids, attention_mask, labels=None):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.projector(encoded)[:, 0]
        local_values = [head(pooled) for head in self.local_heads]
        local_logits = _global_logits(local_values, self.depth2label, self.num_labels)
        auxiliary_logits = None
        if self.architecture == "local_global":
            auxiliary_logits = self.auxiliary(pooled)
        elif self.architecture == "local_nested":
            auxiliary_logits = self.auxiliary(torch.cat(local_values, dim=1))
        logits = auxiliary_logits if auxiliary_logits is not None else local_logits
        result = {"logits": logits, "local_logits": local_logits}
        if auxiliary_logits is not None:
            result["auxiliary_logits"] = auxiliary_logits
        if labels is not None:
            loss = local_logits.new_zeros(())
            for depth, values in zip(sorted(self.depth2label), local_values):
                loss = loss + F.binary_cross_entropy_with_logits(values, labels[:, self.depth2label[depth]])
            if auxiliary_logits is not None:
                loss = loss + F.binary_cross_entropy_with_logits(auxiliary_logits, labels)
            result["loss"] = loss
        return result


def exp_map0(x: torch.Tensor, curvature: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scaled_norm = curvature.sqrt() * torch.norm(x, dim=-1, keepdim=True)
    argument = torch.clamp(scaled_norm, min=eps, max=np.arcsinh(2**15))
    return torch.sinh(argument) * x / torch.clamp(scaled_norm, min=eps)


def pairwise_lorentz_distance(x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_time = torch.sqrt(1 / curvature + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curvature + torch.sum(y**2, dim=-1, keepdim=True))
    inner = x @ y.T - x_time @ y_time.T
    return torch.acosh(torch.clamp(-curvature * inner, min=1 + eps)) / curvature.sqrt()


class HyilrContrastiveLoss(nn.Module):
    """HyILR's released Lorentz loss and hierarchy-aware negative rule."""

    def __init__(self, depth2label, children, hidden_size: int, temperature: float = 0.07):
        super().__init__()
        self.depth2label = depth2label
        self.child_map = children
        self.temperature = temperature
        self.curvature_log = nn.Parameter(torch.tensor(1.0).log())
        self.text_alpha_log = nn.Parameter(torch.tensor(hidden_size**-0.5).log())
        self.label_alpha_log = nn.Parameter(torch.tensor(hidden_size**-0.5).log())
        self.label_depth = {node: depth for depth, nodes in depth2label.items() for node in nodes}

    def _collect_negative_descendants(self, node: int, positive_set: set[int]) -> set[int]:
        # Match the released recursive sampler exactly: a positive child is
        # removed before recursion, so the traversal does not continue through
        # that positive node into its subtree.
        result = {
            child for child in self.child_map.get(node, [])
            if child not in positive_set
        }
        for child in list(result):
            result.update(self._collect_negative_descendants(child, positive_set))
        return result

    def forward(self, text_embeddings, label_embeddings, targets):
        curvature = self.curvature_log.exp()
        text = exp_map0(text_embeddings * self.text_alpha_log.exp(), curvature)
        labels = exp_map0(label_embeddings * self.label_alpha_log.exp(), curvature)
        distances = pairwise_lorentz_distance(text, labels, curvature)
        # Negative identities are nondifferentiable argmin decisions.  The
        # released loop compared individual CUDA scalars from Python, forcing
        # thousands of device synchronizations per batch.  Copy the detached
        # decision inputs once, reproduce the same ordered selection on CPU,
        # and keep all selected distance values and gradients on the GPU.
        decision_distances = distances.detach().cpu()
        decision_targets = targets.detach().cpu()
        losses = []
        for row, target in enumerate(decision_targets):
            positives = torch.where(target > 0)[0].tolist()
            positive_set = set(positives)
            negatives: list[int] = []
            for positive in positives:
                peers = [node for node in self.depth2label[self.label_depth[positive]] if node not in positive_set]
                if peers:
                    unused = [node for node in peers if node not in negatives]
                    pool = unused or peers
                    negatives.append(min(pool, key=lambda node: decision_distances[row, node]))
                descendants = list(self._collect_negative_descendants(positive, positive_set))
                if descendants:
                    negatives.append(min(descendants, key=lambda node: decision_distances[row, node]))
            negatives = list(dict.fromkeys(negatives))
            positive_scores = -distances[row, positives] / self.temperature
            negative_scores = -distances[row, negatives] / self.temperature if negatives else positive_scores.new_empty(0)
            denominator = torch.exp(torch.cat((positive_scores, negative_scores))).sum()
            losses.append((-torch.log(torch.exp(positive_scores) / denominator)).mean())
        return torch.stack(losses).mean()


class HyilrModel(nn.Module):
    def __init__(self, base_model, num_labels, depth2label, children, contrastive_weight=0.3, temperature=0.07):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.classifier = nn.Linear(num_labels * config.hidden_size, num_labels)
        self.contrast = HyilrContrastiveLoss(depth2label, children, config.hidden_size, temperature)
        self.num_labels = num_labels
        self.contrastive_weight = contrastive_weight

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        token_values = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        mask = attention_mask[:, None, :].bool()
        attention = self.attention(token_values).transpose(1, 2).masked_fill(~mask, -torch.inf)
        representation = torch.softmax(attention, dim=-1) @ token_values
        logits = self.classifier(representation.flatten(1))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = result["loss"] + self.contrastive_weight * self.contrast(
                token_values[:, 0], self.attention.weight, labels
            )
        return result


class HierarchyGcnModule(nn.Module):
    """Released HiSR/HiAGM propagation layer, expressed from edge lists."""

    def __init__(self, num_labels, children, edge_probabilities, hidden_size, dropout=0.05):
        super().__init__()
        child_matrix = torch.zeros(num_labels, num_labels)
        parent_matrix = torch.zeros(num_labels, num_labels)
        for parent, child_ids in children.items():
            for child in child_ids:
                child_matrix[parent, child] = 1.0
                parent_matrix[child, parent] = edge_probabilities.get((parent, child), 1.0)
        self.adjacency = nn.Parameter(child_matrix.clone())
        self.register_buffer("adjacency_mask", child_matrix.bool())
        self.reverse_adjacency = nn.Parameter(parent_matrix.clone())
        self.edge_bias = nn.Parameter(torch.zeros(num_labels, hidden_size))
        self.reverse_edge_bias = nn.Parameter(torch.zeros(num_labels, hidden_size))
        self.gate_weight = nn.Parameter(torch.empty(hidden_size, 1))
        self.reverse_gate_weight = nn.Parameter(torch.empty(hidden_size, 1))
        self.loop_gate = nn.Parameter(torch.empty(hidden_size, 1))
        self.gate_bias = nn.Parameter(torch.zeros(num_labels, 1))
        self.reverse_gate_bias = nn.Parameter(torch.zeros(num_labels, 1))
        self.dropout = nn.Dropout(dropout)
        for value in (self.gate_weight, self.reverse_gate_weight, self.loop_gate):
            nn.init.xavier_uniform_(value)

    def forward(self, values):
        forward = torch.matmul(self.adjacency.masked_fill(~self.adjacency_mask, 0), values) + self.edge_bias
        forward = forward * torch.sigmoid(torch.matmul(values, self.gate_weight) + self.gate_bias)
        reverse = torch.matmul(
            self.reverse_adjacency.masked_fill(~self.adjacency_mask.T, 0), values
        ) + self.reverse_edge_bias
        reverse = reverse * torch.sigmoid(torch.matmul(values, self.reverse_gate_weight) + self.reverse_gate_bias)
        loop = values * torch.sigmoid(torch.matmul(values, self.loop_gate))
        return F.relu(self.dropout(forward) + self.dropout(reverse) + self.dropout(loop))


class HisrSelfAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.scale = hidden**0.5

    def forward(self, values):
        weights = torch.softmax(self.query(values) @ self.key(values).transpose(-2, -1) / self.scale, dim=-1)
        return (weights @ self.value(values)).mean(dim=1)


@dataclass(frozen=True)
class HisrPaths:
    membership: torch.Tensor
    lengths: torch.Tensor


class HisrModel(nn.Module):
    def __init__(
        self,
        base_model,
        num_labels,
        children,
        parent,
        edge_probabilities,
        node_loss_weight=0.01,
        sequence_loss_weight=0.01,
        margin=0.15,
        gold_margin=0.5,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model)
        config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(base_model, config=config, add_pooling_layer=True)
        hidden = config.hidden_size
        self.self_attention = HisrSelfAttention(hidden)
        self.transform = nn.Linear(hidden, num_labels * hidden)
        self.graph = HierarchyGcnModule(num_labels, children, edge_probabilities, hidden)
        self.classifier = nn.Linear(num_labels * hidden, num_labels)
        self.dropout = nn.Dropout(0.2)
        self.num_labels = num_labels
        self.node_loss_weight = node_loss_weight
        self.sequence_loss_weight = sequence_loss_weight
        self.margin = margin
        self.gold_margin = gold_margin
        # ``Berthtc.from_pretrained`` applies BERT initialization to newly
        # attached task modules before loading the pretrained encoder.
        for module in (self.self_attention, self.transform, self.classifier):
            module.apply(self.encoder._init_weights)
        path_matrix = torch.zeros(num_labels, num_labels)
        path_lengths = torch.zeros(num_labels)
        for node in range(num_labels):
            path = [node]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path_matrix[node, path] = 1.0
            path_lengths[node] = len(path)
        self.register_buffer("path_membership", path_matrix)
        self.register_buffer("path_lengths", path_lengths)

    @staticmethod
    def ranking_loss(scores, positive_score, margin, gold_margin, compare_candidates=True):
        loss = scores.new_zeros(())
        if compare_candidates:
            for distance in range(1, scores.numel()):
                loss = loss + F.margin_ranking_loss(
                    scores[:-distance], scores[distance:], torch.ones_like(scores[:-distance]), margin * distance
                )
        loss = loss + F.margin_ranking_loss(
            positive_score.expand_as(scores), scores, torch.ones_like(scores), gold_margin
        )
        return loss

    def hierarchy_ranking_loss(self, logits, labels):
        probabilities = torch.sigmoid(logits)
        losses = []
        for scores, target in zip(probabilities, labels):
            positive = target.bool()
            node_gold_score = scores[positive].min()
            sequence_gold_score = scores[positive].mean()
            masked = scores.masked_fill(positive, 0)
            node_scores = torch.topk(masked, min(10, self.num_labels)).values
            node_loss = self.ranking_loss(node_scores, node_gold_score, self.margin, self.gold_margin, False)
            path_scores = (self.path_membership @ scores) / self.path_lengths
            overlap = self.path_membership @ target
            f1 = 2 * overlap / (self.path_lengths + target.sum()).clamp_min(1)
            groups = []
            for value in torch.unique(f1[f1 < 1 - 1e-7], sorted=True).flip(0):
                groups.append(path_scores[torch.isclose(f1, value)].max())
            if groups:
                negative_scores = torch.stack(groups)
                sequence_loss = self.ranking_loss(
                    negative_scores, sequence_gold_score, self.margin, self.gold_margin, True
                ) / len(groups)
            else:
                sequence_loss = node_loss.new_zeros(())
            losses.append(self.node_loss_weight * node_loss + self.sequence_loss_weight * sequence_loss)
        return torch.stack(losses).mean()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        layer_cls = torch.stack([layer[:, 0] for layer in output.hidden_states[-12:]], dim=1)
        text = self.self_attention(layer_cls)
        propagated = self.graph(self.transform(text).view(text.shape[0], self.num_labels, -1))
        logits = self.dropout(self.classifier(propagated.flatten(1)))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = result["loss"] + self.hierarchy_ranking_loss(logits, labels)
        return result
