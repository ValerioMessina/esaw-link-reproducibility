#!/usr/bin/env python3
"""Demonstrate the internal contradiction in printed HTC-CLIP Equations 3--8."""

from __future__ import annotations

import json

import torch

from esaw_htc.htc_clip_model import PathGuidedClassifier


def main() -> None:
    classifier = PathGuidedClassifier(4, 5, {0: [0, 1], 1: [2, 3, 4]}, 3, literal_paper=True)
    logits = classifier(torch.randn(7, 4))
    probabilities = torch.sigmoid(logits)
    roots = probabilities[:, :2]
    payload = {
        "minimum_all_class_probability": probabilities.min().item(),
        "minimum_root_probability": roots.min().item(),
        "all_class_probabilities_at_least_0.5": bool((probabilities >= 0.5).all()),
        "root_lower_bound": float(torch.sigmoid(torch.tensor(1.0))),
        "wos_micro_f1_upper_bound_from_forced_roots": 0.4,
    }
    assert payload["all_class_probabilities_at_least_0.5"]
    assert payload["minimum_root_probability"] >= payload["root_lower_bound"]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
