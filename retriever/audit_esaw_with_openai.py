from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


TARGETS = [
    "COD_LOC",
    "COD_LAV",
    "COD_ATV",
    "COD_MAT_ATV",
    "COD_DEV",
    "COD_MAT_DEV",
    "COD_CON",
    "COD_MAT_CON",
]

FIELD_TAXONOMY = {
    "COD_LOC": "tipo_luogo",
    "COD_LAV": "tipo_lavoro",
    "COD_ATV": "attivita_fisica_specifica",
    "COD_MAT_ATV": "agente_materiale",
    "COD_DEV": "deviazione",
    "COD_MAT_DEV": "agente_materiale",
    "COD_CON": "contatto_modalita_lesione",
    "COD_MAT_CON": "agente_materiale",
}

FIELD_ROLES = {
    "COD_LOC": "Luogo effettivo in cui avviene l'evento lesivo, non destinazione o sede generica.",
    "COD_LAV": "Processo lavorativo generale in corso, piu' ampio dell'azione fisica specifica.",
    "COD_ATV": "Azione concreta della vittima immediatamente prima dell'evento anomalo.",
    "COD_MAT_ATV": "Oggetto/mezzo/superficie associato all'azione concreta della vittima.",
    "COD_DEV": "Evento anomalo, deviazione dall'andamento normale, che porta alla lesione.",
    "COD_MAT_DEV": "Oggetto/strumento coinvolto nell'ultimo evento anomalo, quello piu' vicino al contatto lesivo.",
    "COD_CON": "Modo in cui la lesione si produce: urto, contatto, schiacciamento, sforzo, ecc.",
    "COD_MAT_CON": "Oggetto/strumento/superficie con cui la vittima entra in contatto o che causa la lesione piu' seria.",
}


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload["results"][0]["items"]


def record_key(record: Mapping[str, Any], index: int) -> str:
    for field in ("id", "ID", "numero_caso", "NUMERO_CASO"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"index:{index}"


def record_text(record: Mapping[str, Any]) -> str:
    labels = {
        "luogo_infortunio": "Luogo infortunio",
        "lavorazione": "Lavorazione",
        "dettaglio_lavorazione": "Dettaglio lavorazione",
        "imprevisto": "Imprevisto",
        "conseguenza": "Conseguenza",
        "des_infortunio": "Descrizione infortunio",
    }
    parts = []
    for field, label in labels.items():
        value = record.get(field) or record.get(field.upper())
        if value:
            parts.append(f"{label}: {' '.join(str(value).split())}")
    return "\n".join(parts)


def normalize_code(field: str, code: str | None) -> str:
    if not code:
        return ""
    code = str(code)
    if FIELD_TAXONOMY[field] == "agente_materiale":
        parts = code.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    return code


def label_for(taxonomy: Mapping[str, Mapping[str, str]], field: str, code: str | None) -> str:
    return taxonomy.get(FIELD_TAXONOMY[field], {}).get(normalize_code(field, code), "")


def parent_code(taxonomy: Mapping[str, Mapping[str, str]], field: str, code: str | None) -> str | None:
    code = normalize_code(field, code)
    entries = taxonomy[FIELD_TAXONOMY[field]]
    if not code:
        return None
    if FIELD_TAXONOMY[field] == "agente_materiale":
        parent = f"{code.split('.')[0]}.00"
        return parent if parent in entries and parent != code else None
    if len(code) == 3:
        parent = f"{code[:2]}0"
    elif len(code) == 2:
        parent = f"{code[0]}0"
    else:
        return None
    return parent if parent in entries and parent != code else None


def ancestors(taxonomy: Mapping[str, Mapping[str, str]], field: str, code: str | None) -> List[str]:
    out: List[str] = []
    current = parent_code(taxonomy, field, code)
    while current:
        out.append(current)
        current = parent_code(taxonomy, field, current)
    return out


def is_noinfo(taxonomy: Mapping[str, Mapping[str, str]], field: str, code: str | None) -> bool:
    label = label_for(taxonomy, field, code).lower()
    norm = normalize_code(field, code)
    return "nessuna informazione" in label or "nessun agente materiale" in label or norm in {"000", "00", "00.00", "00.01", "00.02"}


def is_generalist(taxonomy: Mapping[str, Mapping[str, str]], field: str, code: str | None) -> bool:
    label = label_for(taxonomy, field, code).lower()
    norm = normalize_code(field, code)
    has_child = any(parent_code(taxonomy, field, candidate) == norm for candidate in taxonomy[FIELD_TAXONOMY[field]])
    return has_child or "non precisato" in label or "non precisata" in label


def candidate_line(taxonomy: Mapping[str, Mapping[str, str]], field: str, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    code = candidate["code"]
    ancs = ancestors(taxonomy, field, code)
    return {
        "code": code,
        "label": candidate.get("label") or label_for(taxonomy, field, code),
        "rank_dual_encoder": candidate.get("rank_dual_encoder"),
        "parent_code": candidate.get("parent_code"),
        "parent_label": candidate.get("parent_label"),
        "ancestors": ancs,
        "score": candidate.get("score"),
    }


def suspicious_fields(taxonomy: Mapping[str, Mapping[str, str]], prediction: Mapping[str, Any]) -> List[str]:
    fields: List[str] = []
    for field in TARGETS:
        gold = prediction["gold"][field]
        dual = prediction["dual_top1"][field]["code"]
        llm = prediction["llm_top1"][field]["code"]
        gold_general = is_generalist(taxonomy, field, gold)
        gold_missing = is_noinfo(taxonomy, field, gold)
        dual_specific = not is_noinfo(taxonomy, field, dual)
        llm_specific = not is_noinfo(taxonomy, field, llm)
        dual_correct = normalize_code(field, dual) == normalize_code(field, gold)
        llm_correct = normalize_code(field, llm) == normalize_code(field, gold)
        if gold_missing and (dual_specific or llm_specific):
            fields.append(field)
        elif gold_general and (normalize_code(field, dual) != normalize_code(field, gold) or normalize_code(field, llm) != normalize_code(field, gold)):
            fields.append(field)
        elif dual == llm and not dual_correct:
            fields.append(field)
        elif dual != llm and (not dual_correct or not llm_correct):
            fields.append(field)
    return fields


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_pilot_cases(
    taxonomy: Mapping[str, Mapping[str, str]],
    predictions: Sequence[Mapping[str, Any]],
    limit: int,
) -> List[tuple[Mapping[str, Any], List[str]]]:
    scored: List[tuple[int, Mapping[str, Any], List[str]]] = []
    for pred in predictions:
        fields = suspicious_fields(taxonomy, pred)
        if not fields:
            continue
        score = 0
        for field in fields:
            gold = pred["gold"][field]
            dual = pred["dual_top1"][field]["code"]
            llm = pred["llm_top1"][field]["code"]
            if is_noinfo(taxonomy, field, gold):
                score += 5
            if is_generalist(taxonomy, field, gold):
                score += 4
            if dual == llm and normalize_code(field, dual) != normalize_code(field, gold):
                score += 3
            if field in {"COD_DEV", "COD_MAT_ATV", "COD_MAT_DEV", "COD_CON", "COD_MAT_CON"}:
                score += 1
        scored.append((score, pred, fields[:4]))
    scored.sort(key=lambda item: (-item[0], str(item[1]["record_key"])))
    return [(pred, fields) for _, pred, fields in scored[:limit]]


def select_cases(
    taxonomy: Mapping[str, Mapping[str, str]],
    predictions: Sequence[Mapping[str, Any]],
    limit: int,
    case_mode: str,
    field_mode: str,
) -> List[tuple[Mapping[str, Any], List[str]]]:
    if case_mode == "pilot":
        cases = select_pilot_cases(taxonomy, predictions, limit)
    elif case_mode == "suspicious":
        cases = [(pred, suspicious_fields(taxonomy, pred)) for pred in predictions]
        cases = [(pred, fields) for pred, fields in cases if fields]
    elif case_mode == "all-records":
        cases = [(pred, list(TARGETS)) for pred in predictions]
    else:
        raise ValueError(f"case_mode non supportato: {case_mode}")

    normalized: List[tuple[Mapping[str, Any], List[str]]] = []
    for pred, fields in cases:
        if field_mode == "all":
            normalized.append((pred, list(TARGETS)))
        elif field_mode == "suspicious":
            normalized.append((pred, fields))
        else:
            raise ValueError(f"field_mode non supportato: {field_mode}")
    return normalized[:limit]


SYSTEM_PROMPT = """Sei un revisore esperto di codifica ESAW per infortuni sul lavoro.
Devi auditare gold label esistenti, non inventare una tassonomia nuova.

Regole ESAW da applicare:
- COD_LOC: luogo effettivo dell'evento lesivo; non confondere con destinazione, datore di lavoro o contesto generale.
- COD_LAV: processo lavorativo generale; e' piu' ampio dell'azione fisica specifica.
- COD_ATV: azione concreta della vittima immediatamente prima dell'evento anomalo.
- COD_MAT_ATV: agente materiale associato all'azione concreta della vittima.
- COD_DEV: evento anomalo, deviazione dall'andamento normale, che conduce alla lesione.
- COD_MAT_DEV: oggetto/strumento coinvolto nell'ultimo evento anomalo, piu' vicino nel tempo al contatto lesivo.
- COD_CON: modalita' con cui la lesione si produce.
- COD_MAT_CON: oggetto/strumento/superficie con cui la vittima entra in contatto o che causa la lesione piu' seria.

Criteri di audit:
- Se la gold e' un nodo padre/non precisato ma il testo supporta chiaramente un figlio candidato, puoi correggere al figlio.
- Se la gold e' Nessuna informazione ma il testo contiene informazione sufficiente, puoi correggere a un candidato specifico.
- Se il testo non basta, mantieni la gold o usa needs_human_review.
- Scegli solo tra gold e candidati forniti. Non creare codici fuori lista.
- Usa source=keep_gold solo se audited_code coincide con la gold, source=dual solo se coincide col top-1 dual, source=gemma solo se coincide col top-1 Gemma, source=other_top10 per altri candidati ammessi.
- Dai motivazioni brevi e concrete, legate alle regole ESAW.
"""

DEEPSEEK_JSON_INSTRUCTIONS = """
Formato obbligatorio della risposta:
- Rispondi con UN SOLO oggetto JSON, non con una lista.
- Non usare markdown, non usare ```json.
- L'oggetto deve avere esattamente queste chiavi principali:
  {"record_key": "...", "field_audits": [...]}
- field_audits deve contenere un elemento per ogni campo richiesto.
- Ogni elemento di field_audits deve avere esattamente queste chiavi:
  field, audited_code, source, gold_quality, confidence, needs_human_review, rationale.
- source deve essere uno tra: keep_gold, dual, gemma, other_top10, needs_human_review.
- gold_quality deve essere uno tra: valid_specific, too_general_parent, no_information_but_inferable, wrong_variable, ambiguous.
- confidence deve essere uno tra: high, medium, low.
- needs_human_review deve essere true o false.
- rationale deve essere una motivazione breve in italiano.
- Non usare chiavi come motivation, reason, explanation.
"""


def build_user_payload(
    taxonomy: Mapping[str, Mapping[str, str]],
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    fields: Sequence[str],
) -> Dict[str, Any]:
    audit_fields = []
    for field in fields:
        gold_code = prediction["gold"][field]
        dual = prediction["dual_top1"][field]
        gemma = prediction["llm_top1"][field]
        candidates = prediction["dual_candidates"][field]
        allowed: Dict[str, Dict[str, Any]] = {
            gold_code: {
                "code": gold_code,
                "label": label_for(taxonomy, field, gold_code),
                "source": "gold",
            }
        }
        for candidate in candidates[:10]:
            allowed[candidate["code"]] = candidate_line(taxonomy, field, candidate)
        audit_fields.append(
            {
                "field": field,
                "role": FIELD_ROLES[field],
                "gold": {
                    "code": gold_code,
                    "label": label_for(taxonomy, field, gold_code),
                    "is_generalist_or_parent": is_generalist(taxonomy, field, gold_code),
                    "is_no_information": is_noinfo(taxonomy, field, gold_code),
                },
                "dual_top1": candidate_line(taxonomy, field, dual),
                "gemma_top1": candidate_line(taxonomy, field, gemma),
                "allowed_choices": list(allowed.values()),
            }
        )
    return {
        "record_key": prediction["record_key"],
        "accident_text": record_text(record),
        "fields_to_audit": audit_fields,
    }


AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["record_key", "field_audits"],
    "properties": {
        "record_key": {"type": "string"},
        "field_audits": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "field",
                    "audited_code",
                    "source",
                    "gold_quality",
                    "confidence",
                    "needs_human_review",
                    "rationale",
                ],
                "properties": {
                    "field": {"type": "string", "enum": TARGETS},
                    "audited_code": {"type": "string"},
                    "source": {
                        "type": "string",
                        "enum": ["keep_gold", "dual", "gemma", "other_top10", "needs_human_review"],
                    },
                    "gold_quality": {
                        "type": "string",
                        "enum": [
                            "valid_specific",
                            "too_general_parent",
                            "no_information_but_inferable",
                            "wrong_variable",
                            "ambiguous",
                        ],
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "needs_human_review": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


def parse_response_text(response: Mapping[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return str(response["output_text"])
    chunks: List[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if "text" in content:
                chunks.append(content["text"])
    return "\n".join(chunks)


def parse_chat_response_text(response: Mapping[str, Any]) -> str:
    return response["choices"][0]["message"].get("content") or ""


def loads_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        candidate = "".join(ch for ch in candidate if ch in "\t\n\r" or ord(ch) >= 32)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("No JSON object found", text, 0)


def call_openai(
    model: str,
    payload: Mapping[str, Any],
    dry_run: bool = False,
    retries: int = 2,
    timeout: int = 180,
    max_output_tokens: int = 1800,
) -> Dict[str, Any]:
    if dry_run:
        return {"record_key": payload["record_key"], "field_audits": []}
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY non impostata.")
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Audita le gold ESAW nel payload seguente. "
                    "Rispondi solo con JSON conforme allo schema.\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "esaw_gold_audit",
                "strict": True,
                "schema": AUDIT_SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"OpenAI API transient error after {retries + 1} attempts: {exc}") from exc
            time.sleep(2 ** attempt)
    text = parse_response_text(raw)
    return loads_json_object(text)


def call_deepseek(
    model: str,
    payload: Mapping[str, Any],
    dry_run: bool = False,
    retries: int = 2,
    timeout: int = 180,
    max_output_tokens: int = 3200,
    thinking: bool = True,
    reasoning_effort: str = "high",
) -> Dict[str, Any]:
    if dry_run:
        return {"record_key": payload["record_key"], "field_audits": []}
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY non impostata.")
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Audita le gold ESAW nel payload seguente. "
                    "Rispondi solo con JSON valido conforme allo schema richiesto.\n\n"
                    + DEEPSEEK_JSON_INSTRUCTIONS
                    + "\nPayload:\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_output_tokens,
    }
    if thinking:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = reasoning_effort
    else:
        body["thinking"] = {"type": "disabled"}
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API error {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"DeepSeek API transient error after {retries + 1} attempts: {exc}") from exc
            time.sleep(2 ** attempt)
    text = parse_chat_response_text(raw)
    try:
        return loads_json_object(text)
    except json.JSONDecodeError as exc:
        preview = text[:1000].replace("\n", "\\n")
        raise RuntimeError(f"DeepSeek ha restituito JSON non valido: {preview}") from exc


def call_model(
    provider: str,
    model: str,
    payload: Mapping[str, Any],
    dry_run: bool = False,
    retries: int = 2,
    timeout: int = 180,
    max_output_tokens: int = 1800,
    deepseek_thinking: bool = True,
    deepseek_reasoning_effort: str = "high",
) -> Dict[str, Any]:
    if provider == "openai":
        return call_openai(
            model,
            payload,
            dry_run=dry_run,
            retries=retries,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
        )
    if provider == "deepseek":
        return call_deepseek(
            model,
            payload,
            dry_run=dry_run,
            retries=retries,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            thinking=deepseek_thinking,
            reasoning_effort=deepseek_reasoning_effort,
        )
    raise ValueError(f"provider non supportato: {provider}")


from canonical_audit import validate_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot/runner per gold audit ESAW con OpenAI o DeepSeek.")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--case-mode", choices=["pilot", "suspicious", "all-records"], default="pilot")
    parser.add_argument("--field-mode", choices=["suspicious", "all"], default="suspicious")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--deepseek-no-thinking", action="store_true")
    parser.add_argument("--deepseek-reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def csv_rows_from_result(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fields_by_name = {
        field_payload["field"]: field_payload
        for field_payload in row["request"].get("fields_to_audit", [])
    }
    result_rows: List[Dict[str, Any]] = []
    for item in row["audit"].get("field_audits", []):
        field = item["field"]
        field_payload = fields_by_name[field]
        gold_info = field_payload["gold"]
        result_rows.append(
            {
                "record_key": row["record_key"],
                "field": field,
                "gold_code": gold_info["code"],
                "gold_label": gold_info["label"],
                "dual_code": field_payload["dual_top1"]["code"],
                "gemma_code": field_payload["gemma_top1"]["code"],
                "audited_code": item["audited_code"],
                "source": item["source"],
                "gold_quality": item["gold_quality"],
                "confidence": item["confidence"],
                "needs_human_review": item["needs_human_review"],
                "rationale": item["rationale"],
                "validation_errors": "; ".join(row.get("validation_errors", [])),
            }
        )
    return result_rows


def main() -> None:
    args = parse_args()
    records = load_dataset(args.dataset)
    by_key = {record_key(record, index): record for index, record in enumerate(records)}
    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    predictions = load_predictions(args.predictions)
    cases = select_cases(taxonomy, predictions, args.limit, args.case_mode, args.field_mode)
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: List[Dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and output.exists():
        # Revalidate archived responses, including record identity. Invalid rows
        # must not mark work complete or survive beside replacement duplicates.
        from collections import Counter
        import shutil
        existing_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts = Counter(str(row.get("record_key", "")) for row in existing_rows)
        valid_rows = []
        for existing_row in existing_rows:
            key = str(existing_row.get("record_key", ""))
            request = existing_row.get("request", {})
            if (not key or counts[key] != 1 or str(request.get("record_key", "")) != key
                    or existing_row.get("validation_errors")
                    or validate_audit(request, existing_row.get("audit", {}))):
                continue
            completed.add(key)
            valid_rows.append(existing_row)
            csv_rows.extend(csv_rows_from_result(existing_row))
        if len(valid_rows) != len(existing_rows):
            backup = output.with_name(output.name + ".pre-validation-" + str(time.time_ns()) + ".bak")
            shutil.copy2(output, backup)
            temporary = output.with_name(output.name + ".validated.tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid_rows), encoding="utf-8")
            temporary.replace(output)

    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as out:
        for index, (prediction, fields) in enumerate(cases, 1):
            key = str(prediction["record_key"])
            if key in completed:
                print(f"[{index}/{len(cases)}] record={key} skipped=resume", flush=True)
                continue
            payload = build_user_payload(taxonomy, by_key[key], prediction, fields)
            audit = call_model(
                args.provider,
                args.model,
                payload,
                dry_run=args.dry_run,
                retries=args.retries,
                timeout=args.timeout,
                max_output_tokens=args.max_output_tokens,
                deepseek_thinking=not args.deepseek_no_thinking,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
            )
            errors = validate_audit(payload, audit)
            row = {
                "case_index": index,
                "record_key": key,
                "fields_requested": fields,
                "request": payload,
                "audit": audit,
                "validation_errors": errors,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if not errors:
                csv_rows.extend(csv_rows_from_result(row))
            print(f"[{index}/{len(cases)}] record={key} fields={','.join(fields)} errors={len(errors)}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(csv_rows[0].keys()) if csv_rows else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
