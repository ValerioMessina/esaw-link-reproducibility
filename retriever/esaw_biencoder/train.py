from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers.optimization import Adafactor

from .data import (
    Candidate,
    TEXT_FORMATS,
    build_candidates,
    build_explicit_taxonomy,
    candidate_lookup,
    dataclass_to_dict,
    iter_labeled_records,
    label_to_output_code,
    label_to_taxonomy_code,
    load_dataset,
    load_taxonomy,
    query_text,
    record_value,
    sample_negative,
    write_json,
)
from .label_holdout import (
    audit_hard_negative_lookup,
    audit_training_candidates,
    filter_candidates_for_holdout,
    load_holdout_fold,
    sha256_file,
    validate_manifest_sources,
)
from .schema import TARGET_FIELDS


DEFAULT_MODEL_PRESET = "minilm-multilingual"
MODEL_PRESETS = {
    DEFAULT_MODEL_PRESET: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Addestra un bi-encoder SentenceTransformer per classificazione ESAW.")
    parser.add_argument("--dataset", default="dataset.json", help="Path del dataset JSON.")
    parser.add_argument("--taxonomy", default="tassonomia.json", help="Path della tassonomia JSON.")
    parser.add_argument("--output-dir", default="models/esaw-biencoder", help="Cartella in cui salvare modello e metadati.")
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default=DEFAULT_MODEL_PRESET,
        help="Preset SentenceTransformer di partenza. Usa qwen3-embedding-0.6b per Qwen 0.6B.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="SentenceTransformer di partenza. Se impostato, sovrascrive --model-preset.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Commit/revisione Hugging Face del modello base; utile per run esattamente riproducibili.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Device per il modello. Su Apple Silicon usa 'mps'. Default: rilevamento automatico.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adafactor"],
        default="adamw",
        help="Optimizer per il training. 'adafactor' usa meno memoria ed e' utile con modelli grandi come Qwen.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=0,
        help="Lunghezza massima tokenizzata del SentenceTransformer. 0 = default del modello.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="dtype usato per caricare il modello transformer. Utile con Qwen su GPU.",
    )
    parser.add_argument("--use-amp", action="store_true", help="Usa mixed precision AMP durante il training.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Abilita gradient checkpointing sul transformer se supportato.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument(
        "--split-file",
        default=None,
        help="JSON con split fisso train/validation/test. Se non esiste viene creato.",
    )
    parser.add_argument(
        "--label-holdout-file",
        default=None,
        help=(
            "Manifest pre-registrato per generalized label-wise zero-shot. Le classi del fold "
            "sono escluse da positivi, candidati di training e hard negative."
        ),
    )
    parser.add_argument(
        "--label-holdout-fold",
        type=int,
        default=0,
        help="Indice del fold nel --label-holdout-file.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Limita i record usati. 0 = tutti.")
    parser.add_argument("--max-train-examples", type=int, default=0, help="Limita gli esempi/coppie di training. 0 = tutti.")
    parser.add_argument("--max-validation-examples", type=int, default=0, help="Limita gli esempi/coppie di validation. 0 = tutti.")
    parser.add_argument(
        "--trainable-last-n-layers",
        type=int,
        default=0,
        help="Se > 0 congela il transformer e allena solo gli ultimi N layer. 0 = allena tutto.",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument(
        "--loss",
        choices=["cosine", "mnrl", "cached-mnrl"],
        default="cosine",
        help=(
            "'cosine' usa coppie positive/negative e CosineSimilarityLoss; "
            "'mnrl' usa solo coppie positive con MultipleNegativesRankingLoss; "
            "'cached-mnrl' usa la variante memory-efficient."
        ),
    )
    parser.add_argument(
        "--cached-mini-batch-size",
        type=int,
        default=32,
        help="Mini batch interno per CachedMultipleNegativesRankingLoss.",
    )
    parser.add_argument(
        "--mnrl-directions",
        choices=["query-to-doc", "symmetric", "gte"],
        default="query-to-doc",
        help=(
            "Direzioni usate da MultipleNegativesRankingLoss. "
            "'symmetric' aggiunge doc->query; 'gte' aggiunge anche negativi query-query e doc-doc."
        ),
    )
    parser.add_argument(
        "--mnrl-hardness-mode",
        choices=["none", "in-batch-negatives", "hard-negatives", "all-negatives"],
        default="none",
        help="Modalita' hardness supportata da SentenceTransformers per MNRL.",
    )
    parser.add_argument(
        "--mnrl-hardness-strength",
        type=float,
        default=0.0,
        help="Forza della hardness in MNRL quando --mnrl-hardness-mode non e' 'none'.",
    )
    parser.add_argument(
        "--hard-negative-file",
        default=None,
        help="JSON prodotto da esaw_biencoder.mine_hard_negatives con negativi minati dal retriever precedente.",
    )
    parser.add_argument(
        "--mnrl-hard-negatives-per-positive",
        type=int,
        default=0,
        help="Numero di hard negatives espliciti da aggiungere a ogni tupla MNRL.",
    )
    parser.add_argument(
        "--negative-strategy",
        choices=["random", "parent", "hard"],
        default="parent",
        help=(
            "'random' pesca da tutte le classi sbagliate dello stesso campo; "
            "'parent' replica il comportamento storico; "
            "'hard' usa stesso parent/macro e codici vicini."
        ),
    )
    parser.add_argument("--hard-top-n", type=int, default=8, help="Numero di candidati piu' vicini da cui pescare hard negative.")
    parser.add_argument(
        "--text-format",
        choices=TEXT_FORMATS,
        default="legacy",
        help=(
            "'legacy' usa il formato storico; 'v5' mette il target a inizio query; "
            "'v6' aggiunge instruction Qwen e candidate piu' contrastive."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--evaluation-steps", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Cartella in cui salvare checkpoint SentenceTransformers durante il training.",
    )
    parser.add_argument(
        "--checkpoint-save-steps",
        type=int,
        default=500,
        help="Numero di step tra un checkpoint e il successivo se --checkpoint-path e' impostato.",
    )
    parser.add_argument(
        "--checkpoint-save-total-limit",
        type=int,
        default=3,
        help="Numero massimo di checkpoint da conservare. 0 = nessun limite.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Riprende dal checkpoint piu' recente in --checkpoint-path se disponibile.",
    )
    return parser.parse_args()


def resolve_base_model(args: argparse.Namespace) -> str:
    if args.base_model:
        return args.base_model
    return MODEL_PRESETS[args.model_preset]


def candidate_by_any_code(field_candidates: Sequence[Candidate]) -> Dict[str, Candidate]:
    lookup: Dict[str, Candidate] = {}
    for candidate in field_candidates:
        lookup[candidate.code] = candidate
        lookup[candidate.taxonomy_code] = candidate
    return lookup


def load_hard_negative_lookup(path: str | Path | None) -> Dict[str, Dict[str, List[str]]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = payload.get("fields", {})
    lookup: Dict[str, Dict[str, List[str]]] = {}
    for output_field, field_payload in fields.items():
        field_items = field_payload.get("items", {})
        lookup[output_field] = {}
        for split_key, item_payload in field_items.items():
            codes: List[str] = []
            for item in item_payload.get("negatives", []):
                if isinstance(item, str):
                    codes.append(item)
                elif isinstance(item, Mapping) and (item.get("taxonomy_code") or item.get("code")):
                    codes.append(str(item.get("taxonomy_code") or item.get("code")))
            lookup[output_field][split_key] = codes
    return lookup


def mined_negative_candidates(
    split_key: str,
    target_output_field: str,
    positive: Candidate,
    field_candidates: Sequence[Candidate],
    hard_negative_lookup: Mapping[str, Mapping[str, Sequence[str]]],
    limit: int,
) -> List[Candidate]:
    if limit <= 0:
        return []
    by_code = candidate_by_any_code(field_candidates)
    negatives: List[Candidate] = []
    seen = {positive.code, positive.taxonomy_code}
    for code in hard_negative_lookup.get(target_output_field, {}).get(split_key, []):
        candidate = by_code.get(code)
        if candidate is None or candidate.code in seen or candidate.taxonomy_code in seen:
            continue
        negatives.append(candidate)
        seen.add(candidate.code)
        seen.add(candidate.taxonomy_code)
        if len(negatives) >= limit:
            break
    return negatives


def make_examples(
    records: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Candidate]],
    seed: int,
    negatives_per_positive: int,
    negative_strategy: str,
    hard_top_n: int,
    text_format: str,
    hard_negative_lookup: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    mnrl_hard_negatives_per_positive: int = 0,
    include_negatives: bool = True,
) -> List[InputExample]:
    rng = random.Random(seed)
    lookup = candidate_lookup(candidates)
    hard_negative_lookup = hard_negative_lookup or {}
    examples: List[InputExample] = []

    for record_index, record in enumerate(records):
        for target in TARGET_FIELDS:
            raw_label = record_value(record, target.field)
            target_code = label_to_taxonomy_code(target, raw_label)
            if not target_code:
                continue

            positive_pool = [candidate for candidate in candidates[target.field] if candidate.taxonomy_code == target_code]
            if not positive_pool:
                continue
            positive = rng.choice(positive_pool)

            query = query_text(record, target, text_format)
            texts = [query, positive.text]
            if mnrl_hard_negatives_per_positive > 0:
                split_key = record_split_key(record, record_index)
                negatives = mined_negative_candidates(
                    split_key,
                    target.output_field,
                    positive,
                    candidates[target.field],
                    hard_negative_lookup,
                    mnrl_hard_negatives_per_positive,
                )
                seen_codes = {positive.code, positive.taxonomy_code}
                for negative in negatives:
                    seen_codes.add(negative.code)
                    seen_codes.add(negative.taxonomy_code)
                fallback_attempts = 0
                while len(negatives) < mnrl_hard_negatives_per_positive:
                    fallback_attempts += 1
                    if fallback_attempts <= 50:
                        fallback = sample_negative(
                            rng,
                            target_code,
                            candidates[target.field],
                            strategy=negative_strategy,
                            hard_top_n=hard_top_n,
                        )
                    else:
                        fallback_pool = [
                            candidate
                            for candidate in candidates[target.field]
                            if candidate.code not in seen_codes and candidate.taxonomy_code not in seen_codes
                        ]
                        if not fallback_pool:
                            break
                        fallback = rng.choice(fallback_pool)
                    if fallback.code in seen_codes or fallback.taxonomy_code in seen_codes:
                        continue
                    negatives.append(fallback)
                    seen_codes.add(fallback.code)
                    seen_codes.add(fallback.taxonomy_code)
                texts.extend(negative.text for negative in negatives)
            examples.append(InputExample(texts=texts, label=1.0))

            if not include_negatives:
                continue

            for _ in range(negatives_per_positive):
                negative = sample_negative(
                    rng,
                    target_code,
                    candidates[target.field],
                    strategy=negative_strategy,
                    hard_top_n=hard_top_n,
                )
                examples.append(InputExample(texts=[query, negative.text], label=0.0))

    rng.shuffle(examples)
    return examples


def split_records(records: List[Mapping[str, Any]], validation_ratio: float, seed: int) -> tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio)) if validation_ratio > 0 else 0
    return shuffled[validation_size:], shuffled[:validation_size]


def record_split_key(record: Mapping[str, Any], index: int) -> str:
    for field in ("id", "numero_caso"):
        value = record.get(field) or record.get(field.upper())
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"index:{index}"


def records_by_keys(records: Sequence[Mapping[str, Any]], keys: Set[str]) -> List[Mapping[str, Any]]:
    return [
        record
        for index, record in enumerate(records)
        if record_split_key(record, index) in keys
    ]


def fixed_split_records(
    records: List[Mapping[str, Any]],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    split_file: str | Path | None,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], List[Mapping[str, Any]], Dict[str, Any] | None]:
    if split_file is None:
        train_records, validation_records = split_records(records, validation_ratio, seed)
        return train_records, validation_records, [], None

    split_path = Path(split_file)
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        splits = payload.get("splits", payload)
    else:
        rng = random.Random(seed)
        keys = [record_split_key(record, index) for index, record in enumerate(records)]
        rng.shuffle(keys)
        test_size = max(1, int(len(keys) * test_ratio)) if test_ratio > 0 else 0
        validation_size = max(1, int(len(keys) * validation_ratio)) if validation_ratio > 0 else 0
        splits = {
            "test": keys[:test_size],
            "validation": keys[test_size : test_size + validation_size],
            "train": keys[test_size + validation_size :],
        }
        payload = {
            "seed": seed,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "split_key": "id, fallback numero_caso, fallback index",
            "splits": splits,
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(split_path, payload)

    train_records = records_by_keys(records, set(splits.get("train", [])))
    validation_records = records_by_keys(records, set(splits.get("validation", [])))
    test_records = records_by_keys(records, set(splits.get("test", [])))
    return train_records, validation_records, test_records, payload


def mnrl_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    directions_by_name = {
        "query-to-doc": ("query_to_doc",),
        "symmetric": ("query_to_doc", "doc_to_query"),
        "gte": ("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"),
    }
    kwargs: Dict[str, Any] = {
        "directions": directions_by_name[args.mnrl_directions],
        "partition_mode": "per_direction" if args.mnrl_directions == "symmetric" else "joint",
    }
    if args.mnrl_hardness_mode != "none":
        kwargs["hardness_mode"] = args.mnrl_hardness_mode.replace("-", "_")
        kwargs["hardness_strength"] = args.mnrl_hardness_strength
    return kwargs


def build_train_loss(args: argparse.Namespace, model: SentenceTransformer) -> losses.SentenceTransformerLoss:
    if args.loss == "mnrl":
        return losses.MultipleNegativesRankingLoss(model=model, **mnrl_kwargs(args))
    if args.loss == "cached-mnrl":
        return losses.CachedMultipleNegativesRankingLoss(
            model=model,
            mini_batch_size=args.cached_mini_batch_size,
            **mnrl_kwargs(args),
        )
    return losses.CosineSimilarityLoss(model=model)


def optimizer_kwargs(args: argparse.Namespace) -> tuple[type, Dict[str, Any]]:
    if args.optimizer == "adafactor":
        return Adafactor, {
            "lr": args.learning_rate,
            "relative_step": False,
            "scale_parameter": False,
            "warmup_init": False,
        }
    return AdamW, {"lr": args.learning_rate}


def apply_trainable_layer_limit(model: SentenceTransformer, last_n_layers: int) -> int | None:
    if last_n_layers <= 0:
        return None

    transformer = model[0]
    auto_model = getattr(transformer, "auto_model", None)
    layers = getattr(auto_model, "layers", None)
    if layers is None:
        encoder = getattr(auto_model, "encoder", None)
        layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("--trainable-last-n-layers non supportato per questa architettura transformer.")

    for parameter in auto_model.parameters():
        parameter.requires_grad = False

    trainable_layers = list(layers)[-last_n_layers:]
    for layer in trainable_layers:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    for name in ("norm", "ln_f", "final_layernorm"):
        module = getattr(auto_model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def enable_gradient_checkpointing(model: SentenceTransformer) -> bool:
    transformer = model[0]
    auto_model = getattr(transformer, "auto_model", None)
    if auto_model is None or not hasattr(auto_model, "gradient_checkpointing_enable"):
        return False
    auto_model.gradient_checkpointing_enable()
    if hasattr(auto_model.config, "use_cache"):
        auto_model.config.use_cache = False
    return True


def model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.torch_dtype == "auto":
        return {}
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return {"torch_dtype": dtype_by_name[args.torch_dtype]}


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    base_model: str,
    candidates: Mapping[str, Sequence[Candidate]],
    train_records: int,
    validation_records: int,
    test_records: int,
    label_holdout_metadata: Mapping[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "candidates.json",
        {field: [dataclass_to_dict(candidate) for candidate in field_candidates] for field, field_candidates in candidates.items()},
    )
    write_json(
        output_dir / "training_config.json",
        {
            "model_preset": None if args.base_model else args.model_preset,
            "base_model": base_model,
            "model_revision": args.model_revision,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "max_seq_length": args.max_seq_length,
            "torch_dtype": args.torch_dtype,
            "use_amp": args.use_amp,
            "gradient_checkpointing": args.gradient_checkpointing,
            "warmup_ratio": args.warmup_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "split_file": args.split_file,
            "label_holdout_file": args.label_holdout_file,
            "label_holdout_fold": args.label_holdout_fold if args.label_holdout_file else None,
            "label_holdout": dict(label_holdout_metadata or {}),
            "max_records": args.max_records,
            "max_train_examples": args.max_train_examples,
            "max_validation_examples": args.max_validation_examples,
            "trainable_last_n_layers": args.trainable_last_n_layers,
            "loss": args.loss,
            "cached_mini_batch_size": args.cached_mini_batch_size,
            "mnrl_directions": args.mnrl_directions,
            "mnrl_hardness_mode": args.mnrl_hardness_mode,
            "mnrl_hardness_strength": args.mnrl_hardness_strength,
            "hard_negative_file": args.hard_negative_file,
            "mnrl_hard_negatives_per_positive": args.mnrl_hard_negatives_per_positive,
            "negatives_per_positive": args.negatives_per_positive,
            "negative_strategy": args.negative_strategy,
            "hard_top_n": args.hard_top_n,
            "text_format": args.text_format,
            "seed": args.seed,
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_save_steps": args.checkpoint_save_steps,
            "checkpoint_save_total_limit": args.checkpoint_save_total_limit,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "train_records": train_records,
            "validation_records": validation_records,
            "test_records": test_records,
            "fields": [target.field for target in TARGET_FIELDS],
        },
    )


def main() -> None:
    args = parse_args()
    # The original pipeline seeded only example construction.  Seed every
    # training RNG as well so repeated runs are independent and reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    base_model = resolve_base_model(args)

    raw_records = list(iter_labeled_records(load_dataset(args.dataset)))
    if args.max_records and args.max_records > 0:
        raw_records = raw_records[: args.max_records]
    if len(raw_records) < 2:
        raise ValueError("Servono almeno 2 record etichettati per addestrare il bi-encoder.")

    raw_taxonomy = load_taxonomy(args.taxonomy)
    explicit_taxonomy = build_explicit_taxonomy(raw_taxonomy)
    candidates = build_candidates(explicit_taxonomy, text_format=args.text_format)
    train_records, validation_records, test_records, _ = fixed_split_records(
        raw_records,
        args.validation_ratio,
        args.test_ratio,
        args.seed,
        args.split_file,
    )
    hard_negative_lookup = load_hard_negative_lookup(args.hard_negative_file)

    heldout_by_output_field: Dict[str, set[str]] | None = None
    label_holdout_metadata: Dict[str, Any] | None = None
    training_candidates = candidates
    if args.label_holdout_file:
        if not args.split_file:
            raise ValueError("--label-holdout-file richiede --split-file per verificare la pre-registrazione.")
        heldout_by_output_field, manifest = load_holdout_fold(
            args.label_holdout_file, args.label_holdout_fold
        )
        validate_manifest_sources(manifest, args.dataset, args.taxonomy, args.split_file)
        training_candidates = filter_candidates_for_holdout(candidates, heldout_by_output_field)
        candidate_audit = audit_training_candidates(training_candidates, heldout_by_output_field)
        hard_negative_audit = audit_hard_negative_lookup(
            hard_negative_lookup, heldout_by_output_field
        )
        if not candidate_audit["passed"]:
            raise ValueError(f"Leakage nei candidati di training: {candidate_audit['leaked_heldout_codes']}")
        if not hard_negative_audit["passed"]:
            raise ValueError(
                "Il file hard-negative contiene classi held-out e non puo' essere usato: "
                f"{hard_negative_audit['leaked_heldout_codes']}"
            )
        for target in TARGET_FIELDS:
            if len(training_candidates[target.field]) < 2:
                raise ValueError(f"Pool visto insufficiente per {target.output_field}.")

        excluded_train_pairs = {
            target.output_field: sum(
                label_to_taxonomy_code(target, record_value(record, target.field))
                in heldout_by_output_field[target.output_field]
                for record in train_records
            )
            for target in TARGET_FIELDS
        }
        included_train_pairs = {
            target.output_field: sum(
                bool(label_to_taxonomy_code(target, record_value(record, target.field)))
                and label_to_taxonomy_code(target, record_value(record, target.field))
                not in heldout_by_output_field[target.output_field]
                for record in train_records
            )
            for target in TARGET_FIELDS
        }
        label_holdout_metadata = {
            "protocol": manifest["protocol"],
            "manifest_sha256": sha256_file(args.label_holdout_file),
            "fold": args.label_holdout_fold,
            "heldout_taxonomy_codes_by_output_field": {
                field: sorted(codes) for field, codes in heldout_by_output_field.items()
            },
            "excluded_positive_pairs_by_output_field": excluded_train_pairs,
            "included_positive_pairs_by_output_field": included_train_pairs,
            "candidate_audit": candidate_audit,
            "hard_negative_audit": hard_negative_audit,
            "pair_level_protocol_note": (
                "For each target field, pairs whose gold code is held out are excluded. "
                "The same narrative may remain for another target field with a seen gold code; "
                "held-out class documents are never encoded as training positives or negatives."
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "label_holdout_audit.json", label_holdout_metadata)

    train_examples = make_examples(
        train_records,
        training_candidates,
        seed=args.seed,
        negatives_per_positive=args.negatives_per_positive,
        negative_strategy=args.negative_strategy,
        hard_top_n=args.hard_top_n,
        text_format=args.text_format,
        hard_negative_lookup=hard_negative_lookup,
        mnrl_hard_negatives_per_positive=(
            args.mnrl_hard_negatives_per_positive if args.loss in {"mnrl", "cached-mnrl"} else 0
        ),
        include_negatives=args.loss == "cosine",
    )
    validation_examples = make_examples(
        validation_records,
        training_candidates,
        seed=args.seed + 1,
        negatives_per_positive=1,
        negative_strategy=args.negative_strategy,
        hard_top_n=args.hard_top_n,
        text_format=args.text_format,
        hard_negative_lookup={},
        mnrl_hard_negatives_per_positive=0,
        include_negatives=True,
    )
    if args.max_train_examples and args.max_train_examples > 0:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_validation_examples and args.max_validation_examples > 0:
        validation_examples = validation_examples[: args.max_validation_examples]

    model = SentenceTransformer(
        base_model,
        device=args.device,
        model_kwargs=model_kwargs(args),
        revision=args.model_revision,
    )
    if args.max_seq_length and args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length
    gradient_checkpointing_enabled = enable_gradient_checkpointing(model) if args.gradient_checkpointing else False
    trainable_parameters = apply_trainable_layer_limit(model, args.trainable_last_n_layers)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = build_train_loss(args, model)
    optimizer_class, optimizer_params = optimizer_kwargs(args)

    evaluator = None
    if validation_examples:
        evaluator = BinaryClassificationEvaluator.from_input_examples(
            validation_examples[: min(len(validation_examples), 10000)],
            name="esaw-validation",
        )

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)
    print(f"Record train: {len(train_records)} | validation: {len(validation_records)}")
    if args.split_file:
        print(f"Record test: {len(test_records)} | split file: {args.split_file}")
    print(f"Base model: {base_model}")
    print(f"Model revision: {args.model_revision or 'default'}")
    print(f"Max seq length: {model.max_seq_length}")
    print(f"Torch dtype: {args.torch_dtype}")
    print(f"Loss: {args.loss}")
    if args.loss in {"mnrl", "cached-mnrl"}:
        print(
            "MNRL options: "
            f"directions={args.mnrl_directions} "
            f"hardness_mode={args.mnrl_hardness_mode} "
            f"hardness_strength={args.mnrl_hardness_strength} "
            f"explicit_hard_negatives={args.mnrl_hard_negatives_per_positive}"
        )
    if args.hard_negative_file:
        print(f"Hard negative file: {args.hard_negative_file}")
    if label_holdout_metadata:
        print(
            "Label-wise zero-shot: "
            f"fold={args.label_holdout_fold} manifest_sha256={label_holdout_metadata['manifest_sha256']}"
        )
        print(
            "Held-out positive pairs excluded from training: "
            f"{sum(label_holdout_metadata['excluded_positive_pairs_by_output_field'].values())}"
        )
        print("Anti-leakage candidates: PASS | hard negatives: PASS")
    print(f"Text format: {args.text_format}")
    print(f"Optimizer: {args.optimizer}")
    print(f"AMP: {args.use_amp}")
    print(f"Gradient checkpointing: {gradient_checkpointing_enabled}")
    if trainable_parameters is not None:
        print(f"Trainable last layers: {args.trainable_last_n_layers} | trainable parameters: {trainable_parameters}")
    print(f"Coppie train: {len(train_examples)} | validation: {len(validation_examples)}")
    print(f"Warmup steps: {warmup_steps}")
    if args.checkpoint_path:
        print(
            "Checkpoint: "
            f"{args.checkpoint_path} | every {args.checkpoint_save_steps} steps "
            f"| limit {args.checkpoint_save_total_limit} | resume {args.resume_from_checkpoint}"
        )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=args.evaluation_steps if evaluator else 0,
        warmup_steps=warmup_steps,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        use_amp=args.use_amp,
        output_path=str(output_dir),
        show_progress_bar=True,
        checkpoint_path=args.checkpoint_path,
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    write_metadata(
        output_dir,
        args,
        base_model,
        candidates,
        len(train_records),
        len(validation_records),
        len(test_records),
        label_holdout_metadata,
    )
    print(f"Modello salvato in {output_dir}")


if __name__ == "__main__":
    main()
