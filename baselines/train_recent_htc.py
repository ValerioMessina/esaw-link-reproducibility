#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from esaw_htc.common import EsawTextDataset, evaluate_topk, load_hierarchy, read_jsonl, seed_everything, write_json
from esaw_htc.recent_models import HydraModel, HyilrModel, HisrModel


def label_f1(predictions: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    predictions = predictions.bool()
    truth = truth.bool()
    tp = (predictions & truth).sum(0).float()
    fp = (predictions & ~truth).sum(0).float()
    fn = (~predictions & truth).sum(0).float()
    per_label = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    micro = (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1)).item()
    return micro, per_label.mean().item()


@torch.no_grad()
def infer(model, loader, device, score_head="selected"):
    model.eval()
    keys, gold, scores = [], [], []
    for batch in tqdm(loader, desc="eval"):
        keys.extend(batch.pop("record_key"))
        labels = batch.pop("labels")
        output = model(**{
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        })
        logits = output["local_logits"] if score_head == "local" else output["logits"]
        gold.extend([torch.where(row > 0)[0].tolist() for row in labels])
        scores.extend(torch.sigmoid(logits).cpu())
    return keys, gold, scores


def truth_tensor(gold, num_labels):
    truth = torch.zeros((len(gold), num_labels))
    for row, labels in enumerate(gold):
        truth[row, labels] = 1
    return truth


def training_edge_probabilities(rows, hierarchy):
    counts = Counter()
    for row in rows:
        counts.update(row["label"])
    probabilities = {}
    for parent, children in hierarchy.children.items():
        # HiAGM/HiSR normalizes the observed child co-occurrence counts over
        # the parent's direct children.  This equals P(child|parent) when every
        # positive path continues to a child, but remains source-faithful when
        # an admissible prediction stops at an internal ESAW node.
        denominator = sum(counts[child] for child in children)
        for child in children:
            probabilities[(parent, child)] = (
                counts[child] / denominator if denominator else 0.0
            )
    return probabilities


def json_arguments(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main():
    parser = argparse.ArgumentParser(description="Faithful recent HTC reproduction and ESAW adaptation.")
    parser.add_argument("--method", choices=("hydra", "hisr", "hyilr"), required=True)
    parser.add_argument("--hydra-architecture", choices=("local", "local_global", "local_nested"), default="local")
    parser.add_argument("--hydra-score-head", choices=("selected", "local"), default="selected")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-name", default="esaw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--early-stop", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--contrastive-weight", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--source-metric-specific-checkpoints", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-dev-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretokenize", action="store_true")
    args = parser.parse_args()

    defaults = {
        "hydra": dict(base_model="FacebookAI/roberta-base", batch_size=32, learning_rate=3.5e-5, max_epochs=50,
                      early_stop=5, warmup_steps=500, contrastive_weight=0.0, temperature=1.0),
        "hisr": dict(base_model="bert-base-uncased", batch_size=8, learning_rate=1e-5, max_epochs=1000,
                     early_stop=6, warmup_steps=0, contrastive_weight=0.01, temperature=1.0),
        "hyilr": dict(base_model="bert-base-uncased", batch_size=10, learning_rate=1e-5, max_epochs=1000,
                      early_stop=6, warmup_steps=0, contrastive_weight=0.3, temperature=0.07),
    }[args.method]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy = load_hierarchy(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_rows = read_jsonl(args.data_dir / f"{args.data_name}_train.json")
    dev_rows = read_jsonl(args.data_dir / f"{args.data_name}_dev.json")
    test_rows = [] if args.skip_test else read_jsonl(args.data_dir / f"{args.data_name}_test.json")
    if args.max_train_examples:
        train_rows = train_rows[:args.max_train_examples]
    if args.max_dev_examples:
        dev_rows = dev_rows[:args.max_dev_examples]
    if args.max_test_examples:
        test_rows = test_rows[:args.max_test_examples]
    train_set = EsawTextDataset(train_rows, tokenizer, len(hierarchy.labels), args.max_length)
    dev_set = EsawTextDataset(dev_rows, tokenizer, len(hierarchy.labels), args.max_length)
    test_set = EsawTextDataset(test_rows, tokenizer, len(hierarchy.labels), args.max_length)
    if args.pretokenize:
        train_set.pretokenize()
        dev_set.pretokenize()
        test_set.pretokenize()
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, collate_fn=train_set.collate,
        generator=generator, **loader_options,
    )
    dev_loader = DataLoader(dev_set, args.eval_batch_size, collate_fn=dev_set.collate, **loader_options)
    test_loader = None if args.skip_test else DataLoader(
        test_set, args.eval_batch_size, collate_fn=test_set.collate, **loader_options
    )
    # The HiAGM statistic inherited by the released HiSR preprocessing builds
    # its fixed edge-prior file from train+validation.  Preserve that behavior
    # for the source-domain reproduction.  On ESAW, keep validation genuinely
    # held out because it selects architecture/checkpoint configurations.
    edge_probability_rows = train_rows + dev_rows if args.data_name == "wos" else train_rows
    edge_probabilities = training_edge_probabilities(edge_probability_rows, hierarchy)
    if args.method == "hydra":
        model = HydraModel(args.base_model, len(hierarchy.labels), hierarchy.depth2label, args.hydra_architecture)
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if not any(x in n for x in ("bias", "LayerNorm.weight"))], "weight_decay": 0.01},
                {"params": [p for n, p in model.named_parameters() if any(x in n for x in ("bias", "LayerNorm.weight"))], "weight_decay": 0.0},
            ], lr=args.learning_rate
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_steps, len(train_loader) * args.max_epochs
        )
    elif args.method == "hyilr":
        model = HyilrModel(args.base_model, len(hierarchy.labels), hierarchy.depth2label, hierarchy.children,
                           args.contrastive_weight, args.temperature)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = None
    else:
        model = HisrModel(args.base_model, len(hierarchy.labels), hierarchy.children, hierarchy.parent, edge_probabilities,
                          node_loss_weight=args.contrastive_weight, sequence_loss_weight=args.contrastive_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, max(1, int(0.5 * len(train_loader))), eta_min=1e-6, T_mult=2
        )
    model.to(device)

    best_macro = best_micro = -math.inf
    patience = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        loss_count = 0
        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            batch.pop("record_key")
            output = model(**{
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            })
            output["loss"].backward()
            if args.method == "hydra":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum.add_(output["loss"].detach())
            loss_count += 1
        _, gold, scores = infer(model, dev_loader, device, args.hydra_score_head)
        micro, macro = label_f1(torch.stack(scores) > 0.5, truth_tensor(gold, len(hierarchy.labels)))
        row = {"epoch": epoch, "loss": (loss_sum / loss_count).item(), "micro_f1": micro, "macro_f1": macro}
        history.append(row)
        print(json.dumps(row), flush=True)
        patience += 1
        if micro > best_micro:
            best_micro = micro
            if args.source_metric_specific_checkpoints:
                torch.save({"model": model.state_dict(), "args": json_arguments(args), "epoch": epoch}, args.output_dir / "best_micro.pt")
            if args.method in {"hisr", "hyilr"}:
                patience = 0
        if macro > best_macro:
            best_macro = macro
            torch.save({"model": model.state_dict(), "args": json_arguments(args), "epoch": epoch}, args.output_dir / "best_macro.pt")
            patience = 0
        if patience >= args.early_stop:
            break
    validation = {
        "method": args.method.upper(), "selection_split": "validation", "best_macro_f1": best_macro,
        "best_micro_f1": best_micro, "history": history, "hyperparameters": json_arguments(args),
    }
    write_json(args.output_dir / "validation.json", validation)
    if args.skip_test:
        return

    checkpoint = torch.load(args.output_dir / "best_macro.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    keys, gold, scores = infer(model, test_loader, device, args.hydra_score_head)
    metrics, predictions = evaluate_topk(keys, gold, scores, hierarchy)
    macro_micro, macro_macro = label_f1(torch.stack(scores) > 0.5, truth_tensor(gold, len(hierarchy.labels)))
    paper_micro = macro_micro
    if args.source_metric_specific_checkpoints:
        micro_checkpoint = torch.load(args.output_dir / "best_micro.pt", map_location=device, weights_only=False)
        model.load_state_dict(micro_checkpoint["model"])
        _, micro_gold, micro_scores = infer(model, test_loader, device, args.hydra_score_head)
        paper_micro, _ = label_f1(
            torch.stack(micro_scores) > 0.5, truth_tensor(micro_gold, len(hierarchy.labels))
        )
    metrics.update({
        "method": args.method.upper(), "seed": args.seed, "checkpoint": checkpoint["epoch"],
        "test_micro_f1": paper_micro, "test_macro_f1": macro_macro,
        "common_macro_checkpoint_micro_f1": macro_micro,
        "source_checkpoint_policy": "metric-specific" if args.source_metric_specific_checkpoints else "best validation Macro-F1",
        "history": history,
    })
    write_json(args.output_dir / "benchmark.json", metrics)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
