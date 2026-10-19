#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from esaw_htc.common import EsawTextDataset, evaluate_topk, load_hierarchy, read_jsonl, seed_everything, write_json
from esaw_htc.hgclr_model import Hgclr
from esaw_htc.htc_clip_model import HtcClip
from model.optim import Adam as ReleasedHgclrAdam


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
def infer(model, loader, device):
    model.eval()
    keys: list[str] = []
    gold: list[list[int]] = []
    scores: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="eval"):
        record_keys = batch.pop("record_key")
        labels = batch.pop("labels")
        output = model(**{
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        })
        keys.extend(record_keys)
        gold.extend([torch.where(row > 0)[0].tolist() for row in labels])
        scores.extend(torch.sigmoid(output["logits"]).cpu())
    return keys, gold, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithful HGCLR or HTC-CLIP adaptation.")
    parser.add_argument("--architecture", choices=("htc-clip", "hgclr"), default="htc-clip")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-name", default="esaw", help="Prefix of *_train/dev/test.json files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--path-hidden", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--early-stop", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-dev-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretokenize", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hierarchy = load_hierarchy(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_rows = read_jsonl(args.data_dir / f"{args.data_name}_train.json")
    dev_rows = read_jsonl(args.data_dir / f"{args.data_name}_dev.json")
    test_rows = [] if args.skip_test else read_jsonl(args.data_dir / f"{args.data_name}_test.json")
    if args.max_train_examples:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_dev_examples:
        dev_rows = dev_rows[: args.max_dev_examples]
    if args.max_test_examples:
        test_rows = test_rows[: args.max_test_examples]
    train_set = EsawTextDataset(train_rows, tokenizer, len(hierarchy.labels), args.max_length)
    dev_set = EsawTextDataset(dev_rows, tokenizer, len(hierarchy.labels), args.max_length)
    test_set = EsawTextDataset(test_rows, tokenizer, len(hierarchy.labels), args.max_length)
    if args.pretokenize:
        train_set.pretokenize()
        dev_set.pretokenize()
        test_set.pretokenize()
    loader_options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, collate_fn=train_set.collate, **loader_options
    )
    dev_loader = DataLoader(
        dev_set, args.eval_batch_size, collate_fn=dev_set.collate, **loader_options
    )
    test_loader = None if args.skip_test else DataLoader(
        test_set, args.eval_batch_size, collate_fn=test_set.collate, **loader_options
    )
    if args.architecture == "htc-clip":
        model = HtcClip(
            args.base_model,
            len(hierarchy.labels),
            hierarchy.depth2label,
            [hierarchy.display[name] for name in hierarchy.labels],
            str(args.data_dir),
            args.gamma,
            args.contrastive_weight,
            args.temperature,
            args.path_hidden,
        ).to(device)
    else:
        model = Hgclr(
            args.base_model,
            len(hierarchy.labels),
            str(args.data_dir),
            args.gamma,
            args.contrastive_weight,
            args.temperature,
        ).to(device)
    # HTC-CLIP states that it follows HGCLR's Adam setup. The released HGCLR
    # optimizer is numerically Adam at zero weight decay and is used here for
    # both architectures to preserve that source-declared implementation.
    optimizer = ReleasedHgclrAdam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.warmup_steps:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / args.warmup_steps, 1.0),
        )
    best_macro = -math.inf
    best_micro = -math.inf
    patience = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        loss_count = 0
        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            batch.pop("record_key")
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            output = model(**batch)
            output["loss"].backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum.add_(output["loss"].detach())
            loss_count += 1
        _, gold, scores = infer(model, dev_loader, device)
        truth = torch.zeros((len(gold), len(hierarchy.labels)))
        for i, labels in enumerate(gold):
            truth[i, labels] = 1
        micro, macro = label_f1(torch.stack(scores) > 0.5, truth)
        row = {"epoch": epoch, "loss": (loss_sum / loss_count).item(), "micro_f1": micro, "macro_f1": macro}
        history.append(row)
        print(json.dumps(row))
        patience += 1
        improved_micro = micro > best_micro
        if improved_micro:
            best_micro = micro
            # The released HGCLR runner resets patience on either validation
            # metric, although its published test command loads the
            # best-Macro checkpoint. HTC-CLIP explicitly stops on Macro only.
            if args.architecture == "hgclr":
                patience = 0
        if macro > best_macro:
            best_macro = macro
            patience = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "macro_f1": macro}, args.output_dir / "best.pt")
        if patience >= args.early_stop:
            break
    validation_payload = {
        "method": "HTC-CLIP" if args.architecture == "htc-clip" else "HGCLR",
        "selection_split": "validation",
        "best_macro_f1": best_macro,
        "best_micro_f1": best_micro,
        "best_epoch": next(row["epoch"] for row in history if row["macro_f1"] == best_macro),
        "history": history,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "gamma": args.gamma,
            "contrastive_weight": args.contrastive_weight,
            "temperature": args.temperature,
            "path_hidden": args.path_hidden,
            "max_length": args.max_length,
            "seed": args.seed,
        },
    }
    validation_output = args.validation_output or (args.output_dir / "validation.json")
    write_json(validation_output, validation_payload)
    if args.skip_test:
        return
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    keys, gold, scores = infer(model, test_loader, device)
    metrics, predictions = evaluate_topk(keys, gold, scores, hierarchy)
    truth = torch.zeros((len(gold), len(hierarchy.labels)))
    for i, labels in enumerate(gold):
        truth[i, labels] = 1
    test_micro, test_macro = label_f1(torch.stack(scores) > 0.5, truth)
    metrics.update({
        "method": "HTC-CLIP" if args.architecture == "htc-clip" else "HGCLR",
        "paper_faithful_components": True,
        "fidelity_status": (
            "official HGCLR implementation ported to the pinned modern environment"
            if args.architecture == "hgclr"
            else "paper-based HTC-CLIP reconstruction; executable class-logit erratum; original code unavailable"
        ),
        "seed": args.seed,
        "checkpoint": checkpoint["epoch"],
        "test_micro_f1": test_micro,
        "test_macro_f1": test_macro,
        "history": history,
    })
    write_json(args.output_dir / "benchmark.json", metrics)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
