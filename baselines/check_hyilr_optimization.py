#!/usr/bin/env python3
from __future__ import annotations

import copy
import argparse
import json
import time

import torch

from esaw_htc.recent_models import HyilrContrastiveLoss, exp_map0, pairwise_lorentz_distance
from esaw_htc.common import load_hierarchy


def reference_forward(module, text_embeddings, label_embeddings, targets):
    curvature = module.curvature_log.exp()
    text = exp_map0(text_embeddings * module.text_alpha_log.exp(), curvature)
    labels = exp_map0(label_embeddings * module.label_alpha_log.exp(), curvature)
    distances = pairwise_lorentz_distance(text, labels, curvature)
    losses = []
    for row, target in enumerate(targets):
        positives = torch.where(target > 0)[0].tolist()
        positive_set = set(positives)
        negatives = []
        for positive in positives:
            peers = [
                node for node in module.depth2label[module.label_depth[positive]]
                if node not in positive_set
            ]
            if peers:
                unused = [node for node in peers if node not in negatives]
                pool = unused or peers
                negatives.append(min(pool, key=lambda node: distances[row, node]))
            descendants = list(module._collect_negative_descendants(positive, positive_set))
            if descendants:
                negatives.append(min(descendants, key=lambda node: distances[row, node]))
        negatives = list(dict.fromkeys(negatives))
        positive_scores = -distances[row, positives] / module.temperature
        negative_scores = (
            -distances[row, negatives] / module.temperature
            if negatives else positive_scores.new_empty(0)
        )
        denominator = torch.exp(torch.cat((positive_scores, negative_scores))).sum()
        losses.append((-torch.log(torch.exp(positive_scores) / denominator)).mean())
    return torch.stack(losses).mean()


def run_equivalence_check():
    torch.manual_seed(13)
    depth2label = {0: [0, 1], 1: [2, 3, 4, 5], 2: [6, 7, 8, 9]}
    children = {0: [2, 3], 1: [4, 5], 2: [6, 7], 3: [8], 4: [9]}
    reference = HyilrContrastiveLoss(depth2label, children, hidden_size=12)
    optimized = copy.deepcopy(reference)
    targets = torch.zeros(4, 10)
    targets[0, [0, 2, 6]] = 1
    targets[1, [0, 3, 8]] = 1
    targets[2, [1, 4, 9]] = 1
    targets[3, [1, 5]] = 1
    reference_text = torch.randn(4, 12, requires_grad=True)
    reference_labels = torch.randn(10, 12, requires_grad=True)
    optimized_text = reference_text.detach().clone().requires_grad_(True)
    optimized_labels = reference_labels.detach().clone().requires_grad_(True)
    old_loss = reference_forward(reference, reference_text, reference_labels, targets)
    new_loss = optimized(optimized_text, optimized_labels, targets)
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    torch.testing.assert_close(optimized_text.grad, reference_text.grad, rtol=0, atol=0)
    torch.testing.assert_close(optimized_labels.grad, reference_labels.grad, rtol=0, atol=0)
    for old_parameter, new_parameter in zip(reference.parameters(), optimized.parameters()):
        torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=0, atol=0)
    print("HyILR optimization equivalence: PASS")


def run_cuda_benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the synchronization benchmark")
    hierarchy = load_hierarchy("data/esaw")
    with open("data/esaw/esaw_train.json", encoding="utf-8") as handle:
        rows = [json.loads(next(handle)) for _ in range(10)]
    targets = torch.zeros(10, len(hierarchy.labels), device="cuda")
    for row_index, row in enumerate(rows):
        targets[row_index, row["label"]] = 1
    text = torch.randn(10, 768, device="cuda")
    labels = torch.randn(len(hierarchy.labels), 768, device="cuda")
    module = HyilrContrastiveLoss(hierarchy.depth2label, hierarchy.children, 768).cuda()

    def timed(function, repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            function()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / repeats

    optimized_seconds = timed(lambda: module(text, labels, targets), 3)
    legacy_seconds = timed(lambda: reference_forward(module, text, labels, targets), 1)
    print(json.dumps({
        "legacy_loss_seconds": legacy_seconds,
        "optimized_loss_seconds": optimized_seconds,
        "loss_speedup": legacy_seconds / optimized_seconds,
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    arguments = parser.parse_args()
    run_equivalence_check()
    if arguments.benchmark:
        run_cuda_benchmark()
