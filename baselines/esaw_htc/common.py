from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Hierarchy:
    labels: list[str]
    display: dict[str, str]
    children: dict[int, list[int]]
    parent: dict[int, int]
    depth2label: dict[int, list[int]]
    field_leaves: dict[str, list[int]]
    output_by_leaf: dict[str, dict[int, str]]


def load_hierarchy(data_dir: str | Path) -> Hierarchy:
    payload = json.loads((Path(data_dir) / "metadata.json").read_text(encoding="utf-8"))
    labels = payload["labels"]
    ids = {name: i for i, name in enumerate(labels)}
    children = {ids[parent]: [ids[child] for child in values] for parent, values in payload["children"].items()}
    parent = {child: root for root, values in children.items() for child in values}

    def depth(node: int) -> int:
        result = 0
        while node in parent:
            result += 1
            node = parent[node]
        return result

    depth2label: dict[int, list[int]] = {}
    for node in range(len(labels)):
        depth2label.setdefault(depth(node), []).append(node)
    return Hierarchy(
        labels=labels,
        display=payload["display"],
        children=children,
        parent=parent,
        depth2label=depth2label,
        field_leaves={field: [ids[name] for name in values] for field, values in payload["field_leaves"].items()},
        output_by_leaf={
            field: {ids[name]: code for name, code in values.items()} for field, values in payload["output_by_leaf"].items()
        },
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class EsawTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, num_labels: int, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length
        self.encoded: dict[str, torch.Tensor] | None = None
        self.encoded_labels: torch.Tensor | None = None

    def pretokenize(self, chunk_size: int = 2048) -> None:
        """Cache deterministic padded tokenization for repeated training epochs."""
        if self.encoded is not None:
            return
        chunks: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(self.rows), chunk_size):
            encoded = self.tokenizer(
                [row["token"] for row in self.rows[start:start + chunk_size]],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            for key, value in encoded.items():
                chunks.setdefault(key, []).append(value)
        self.encoded = {
            key: torch.cat(values, dim=0) if values else torch.empty((0, self.max_length), dtype=torch.long)
            for key, values in chunks.items()
        }
        self.encoded_labels = torch.zeros((len(self.rows), self.num_labels), dtype=torch.float32)
        for index, row in enumerate(self.rows):
            self.encoded_labels[index, row["label"]] = 1.0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.encoded is not None:
            return {"_pretokenized_index": index}
        return self.rows[index]

    def collate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self.encoded is not None and self.encoded_labels is not None:
            indices = torch.tensor([row["_pretokenized_index"] for row in rows], dtype=torch.long)
            tokens = {key: value.index_select(0, indices) for key, value in self.encoded.items()}
            tokens["labels"] = self.encoded_labels.index_select(0, indices)
            tokens["record_key"] = [self.rows[index]["record_key"] for index in indices.tolist()]
            return tokens
        tokens = self.tokenizer(
            [row["token"] for row in rows],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.zeros((len(rows), self.num_labels), dtype=torch.float32)
        for i, row in enumerate(rows):
            labels[i, row["label"]] = 1.0
        tokens["labels"] = labels
        tokens["record_key"] = [row["record_key"] for row in rows]
        return tokens


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def topk_by_field(scores: torch.Tensor, hierarchy: Hierarchy, top_k: int = 10) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for field, ids in hierarchy.field_leaves.items():
        k = min(top_k, len(ids))
        local_scores = scores[ids]
        local_indices = torch.topk(local_scores, k=k).indices.tolist()
        result[field] = [hierarchy.output_by_leaf[field][ids[index]] for index in local_indices]
    return result


def evaluate_topk(
    record_keys: Iterable[str],
    gold_ids: Iterable[Iterable[int]],
    score_rows: Iterable[torch.Tensor],
    hierarchy: Hierarchy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counters = {field: {"records": 0, "top1": 0, "top5": 0, "top10": 0} for field in hierarchy.field_leaves}
    predictions: list[dict[str, Any]] = []
    exact = 0
    recall5 = 0
    recall10 = 0
    for key, gold, scores in zip(record_keys, gold_ids, score_rows):
        gold_set = set(gold)
        top = topk_by_field(scores, hierarchy)
        gold_by_field: dict[str, str] = {}
        for field, leaf_ids in hierarchy.field_leaves.items():
            # A fine-grained gold path contains both its taxonomic parent and
            # its leaf. Parent codes are also admissible ESAW outputs, so a
            # first-match lookup can silently replace a child gold with its
            # ancestor. Select the deepest admissible node on the path.
            candidates = [node for node in leaf_ids if node in gold_set]
            if not candidates:
                continue
            hit = max(candidates, key=lambda node: next(depth for depth, nodes in hierarchy.depth2label.items() if node in nodes))
            gold_code = hierarchy.output_by_leaf[field][hit]
            gold_by_field[field] = gold_code
            counters[field]["records"] += 1
            counters[field]["top1"] += int(gold_code in top[field][:1])
            counters[field]["top5"] += int(gold_code in top[field][:5])
            counters[field]["top10"] += int(gold_code in top[field][:10])
        fields = list(gold_by_field)
        exact += int(all(gold_by_field[field] in top[field][:1] for field in fields))
        recall5 += int(all(gold_by_field[field] in top[field][:5] for field in fields))
        recall10 += int(all(gold_by_field[field] in top[field][:10] for field in fields))
        predictions.append({"record_key": key, "gold": gold_by_field, "top10": top})
    field_metrics: dict[str, Any] = {}
    for field, counts in counters.items():
        n = max(1, counts["records"])
        field_metrics[field] = {
            "records": counts["records"],
            "top1_accuracy": counts["top1"] / n,
            "top5_accuracy": counts["top5"] / n,
            "top10_accuracy": counts["top10"] / n,
        }
    total = max(1, len(predictions))
    metrics = {
        "fields": field_metrics,
        "mean": {
            name: sum(values[name] for values in field_metrics.values()) / len(field_metrics)
            for name in ("top1_accuracy", "top5_accuracy", "top10_accuracy")
        },
        "record_exact@1": exact / total,
        "record_recall@5": recall5 / total,
        "record_recall@10": recall10 / total,
    }
    return metrics, predictions


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
