#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default="bert-base-uncased")
    args = parser.parse_args()
    meta = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = meta["labels"]
    ids = {name: i for i, name in enumerate(labels)}
    children = {ids[parent]: {ids[child] for child in values} for parent, values in meta["children"].items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # HGCLR/HTC-CLIP uses real label names in its Graphormer label encoder.
    # HGCLR/HTC-CLIP expects only the lexical label tokens here: the label
    # encoder adds its own sequence-level special tokens afterwards.
    bert_value_dict = {
        i: tokenizer.encode(meta["display"][name], add_special_tokens=False) for i, name in enumerate(labels)
    }
    torch.save(bert_value_dict, args.data_dir / "bert_value_dict.pt")
    torch.save(children, args.data_dir / "slot.pt")

    # DPT replaces BERT's [unused] inventory with atomic fabricated symbols.
    # BERT-base has enough unused entries for the 826 ESAW nodes plus the
    # level separator, so preserve their original token IDs and embeddings.
    symbols = {i: f"[unused{i + 1}]" for i in range(len(labels))}
    torch.save(symbols, args.data_dir / "value_dict.pt")
    torch.save({name: symbols[i] for i, name in enumerate(labels)}, args.data_dir / "label_symbol_dict.pt")
    with (args.data_dir / "special_tokens.txt").open("w", encoding="utf-8") as handle:
        handle.write("[unused0]\n")
        for symbol in symbols.values():
            handle.write(symbol + "\n")
    print(json.dumps({"labels": len(labels), "edges": sum(map(len, children.values())), "model": args.model}, indent=2))


if __name__ == "__main__":
    main()
