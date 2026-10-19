from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from esaw_biencoder.data import (
    Candidate,
    TEXT_FORMATS,
    accident_to_text,
    build_candidates,
    build_explicit_taxonomy,
    label_to_output_code,
    load_dataset,
    load_taxonomy,
    query_text,
    record_value,
)
from esaw_biencoder.evaluate import (
    TEXT_FORMAT_AUTO,
    cosine_matrix,
    filter_records_by_split,
    load_candidates,
    resolve_text_format,
)
from esaw_biencoder.retrieval import top_candidate_details
from esaw_biencoder.schema import TARGET_FIELDS, TargetField


class TransformerEncoderFallback:
    def __init__(self, model_path: str, device: str = "auto", max_length: int = 256) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        outputs: List[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="encode")
        with torch.no_grad():
            for start in iterator:
                batch = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                positions = torch.arange(hidden.size(1), device=self.device).unsqueeze(0)
                last_token_positions = (encoded["attention_mask"] * positions).argmax(dim=1)
                pooled = hidden[torch.arange(hidden.size(0), device=self.device), last_token_positions]
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.detach().float().cpu())
        embeddings = torch.cat(outputs, dim=0)
        return embeddings.numpy() if convert_to_numpy else embeddings


def load_dual_encoder(model_path: str, device: str = "auto"):
    try:
        return SentenceTransformer(model_path)
    except Exception as exc:
        print(
            f"SentenceTransformer load failed ({type(exc).__name__}: {exc}). "
            "Falling back to AutoModel last-token pooling.",
            file=sys.stderr,
            flush=True,
        )
        return TransformerEncoderFallback(model_path, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark di reranking LLM sopra il dual encoder ESAW. "
            "Il modello LLM deve essere gia' locale o servito da un endpoint; lo script non scarica pesi."
        )
    )
    parser.add_argument("--model", required=True, help="Directory del dual encoder v6.")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--taxonomy", default="tassonomia.json")
    parser.add_argument("--split-file", default="artifacts/splits/esaw-fixed-seed13-test10-val10.json")
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-top-k", type=int, choices=[3, 5, 10, 12], default=10)
    parser.add_argument(
        "--candidate-input-jsonl",
        default="",
        help=(
            "JSONL opzionale con candidati gia' preparati/rerankati per campo. "
            "Ogni riga puo' essere un record completo con dual_candidates oppure una riga campo "
            "con record_key, field e rerank_top5_candidates/rerank_top_candidates."
        ),
    )
    parser.add_argument(
        "--text-format",
        choices=(TEXT_FORMAT_AUTO, *TEXT_FORMATS),
        default=TEXT_FORMAT_AUTO,
    )
    parser.add_argument(
        "--rebuild-candidates",
        action="store_true",
        help="Ricostruisce i candidati dalla tassonomia invece di usare candidates.json del modello.",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "openai-compatible", "ollama", "noop", "oracle"],
        default="noop",
        help=(
            "noop conserva il ranking del dual encoder; oracle misura il tetto massimo usando il gold; "
            "transformers, openai-compatible e ollama sono i backend reali."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Path locale o nome servito del LLM, es. una directory Gemma gia' scaricata.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permette a transformers di scaricare risorse mancanti. Default: disattivato.",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument(
        "--ollama-think",
        action="store_true",
        help="Lascia attivo il thinking nei modelli Ollama che lo supportano. Default: disattivato.",
    )
    parser.add_argument(
        "--ollama-think-level",
        choices=["low", "medium", "high"],
        default="",
        help="Livello thinking Ollama; se impostato prevale su --ollama-think.",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=4096,
        help="Contesto Ollama da usare per ridurre KV cache e memoria. Default: 4096.",
    )
    parser.add_argument(
        "--ollama-unload-every",
        type=int,
        default=0,
        help="Se >0, scarica il modello Ollama ogni N record completati per liberare VRAM/cache.",
    )
    parser.add_argument(
        "--conservative-rerank",
        action="store_true",
        help="Chiede al LLM di mantenere il rank 1 salvo evidenza chiara a favore di un altro candidato.",
    )
    parser.add_argument(
        "--hierarchical-rerank",
        action="store_true",
        help=(
            "Usa un prompt gerarchico: prima sceglie la macroclasse/parent dei candidati, "
            "poi la foglia specifica. Utile quando il flat rerank confonde generico e specifico."
        ),
    )
    parser.add_argument(
        "--single-record-prompt",
        action="store_true",
        help="Usa una sola chiamata LLM per record e restituisce le 8 variabili in un unico JSON.",
    )
    parser.add_argument(
        "--compact-single-record-prompt",
        action="store_true",
        help="Riduce guide e motivazioni del prompt unico, mantenendo caso, Top-K e 8 decisioni ESAW.",
    )
    parser.add_argument(
        "--compact-field-prompt",
        action="store_true",
        help="Usa una richiesta JSON minima per ogni singola variabile ESAW.",
    )
    parser.add_argument(
        "--acceptance-policy",
        choices=["llm", "esaw-rules"],
        default="llm",
        help="Policy per accettare una modifica proposta dal LLM. 'esaw-rules' filtra cambi rischiosi.",
    )
    parser.add_argument("--no-consistency-pass", action="store_true")
    parser.add_argument("--predictions-output", default="")
    parser.add_argument(
        "--resume-predictions",
        action="store_true",
        help="Se --predictions-output esiste, salta i record gia' completati e continua in append.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0.0,
        help="Ferma ordinatamente il benchmark dopo questo numero di minuti, salvando il progresso.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def complete_target_records(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        record
        for record in records
        if all(record_value(record, target.field) for target in TARGET_FIELDS)
    ]


def record_key(record: Mapping[str, Any], index: int) -> str:
    for field in ("id", "numero_caso"):
        value = record.get(field) or record.get(field.upper())
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"index:{index}"


def candidate_payload(candidate: Candidate, score: float, rank: int) -> Dict[str, Any]:
    return {
        "rank_dual_encoder": rank,
        "code": candidate.code,
        "taxonomy_code": candidate.taxonomy_code,
        "label": candidate.label,
        "parent_code": candidate.parent_code,
        "parent_label": candidate.parent_label,
        "ancestors": candidate.ancestors,
        "score": round(float(score), 6),
        "prototype_id": candidate.prototype_id,
    }


def pick_best_unique(details: Sequence[tuple[Candidate, float]], top_k: int) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for candidate, score in details:
        if candidate.code in seen:
            continue
        seen.add(candidate.code)
        rows.append(candidate_payload(candidate, score, len(rows) + 1))
        if len(rows) >= top_k:
            break
    return rows


def render_candidate_list(candidates: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for item in candidates:
        parent = f"; macroclasse={item['parent_label']}" if item.get("parent_label") else ""
        lines.append(
            f"{item['rank_dual_encoder']}. codice={item['code']}; label={item['label']}"
            f"{parent}; score_dual={item['score']}"
        )
    return "\n".join(lines)


def normalized_parent_code(item: Mapping[str, Any]) -> str:
    parent = str(item.get("parent_code") or "").strip()
    if parent:
        return parent

    ancestors = item.get("ancestors")
    if isinstance(ancestors, list) and ancestors:
        first = str(ancestors[0] or "").strip()
        if first:
            return first

    taxonomy_code = str(item.get("taxonomy_code") or item.get("code") or "").strip()
    if len(taxonomy_code) == 2 and taxonomy_code.isdigit() and taxonomy_code[1] != "0":
        return f"{taxonomy_code[0]}0"
    if len(taxonomy_code) == 3 and taxonomy_code.isdigit() and taxonomy_code[2] != "0":
        return f"{taxonomy_code[:2]}0"
    if "." in taxonomy_code:
        parts = taxonomy_code.split(".")
        if len(parts) >= 2 and parts[1] != "00":
            return f"{parts[0]}.00"
    return taxonomy_code or "ROOT"


def candidate_parent_key(item: Mapping[str, Any]) -> str:
    parent_code = normalized_parent_code(item)
    parent_label = str(item.get("parent_label") or "").strip()
    if parent_label:
        return f"{parent_code} | {parent_label}"
    if parent_code == str(item.get("taxonomy_code") or item.get("code") or "").strip():
        return f"{parent_code} | {item.get('label') or 'ROOT'}"
    return f"{parent_code} | macroclasse parent"


def render_hierarchical_candidate_list(candidates: Sequence[Mapping[str, Any]]) -> str:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for item in candidates:
        groups.setdefault(candidate_parent_key(item), []).append(item)

    lines = []
    for parent_index, (parent, items) in enumerate(groups.items(), start=1):
        best_rank = min(int(item["rank_dual_encoder"]) for item in items)
        best_score = max(float(item["score"]) for item in items)
        lines.append(f"Macro {parent_index}: parent={parent}; best_rank_dual={best_rank}; best_score_dual={best_score:.6f}")
        for item in items:
            lines.append(
                f"  - rank={item['rank_dual_encoder']}; codice={item['code']}; "
                f"taxonomy_code={item.get('taxonomy_code')}; parent_code={normalized_parent_code(item)}; "
                f"label={item['label']}; score_dual={item['score']}"
            )
    return "\n".join(lines)


def render_compact_hierarchical_candidate_list(candidates: Sequence[Mapping[str, Any]]) -> str:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for item in candidates:
        groups.setdefault(candidate_parent_key(item), []).append(item)

    lines = []
    for parent, items in groups.items():
        lines.append(f"parent={parent}")
        for item in items:
            lines.append(
                f"- r{item['rank_dual_encoder']} code={item['code']} score={item['score']} label={item['label']}"
            )
    return "\n".join(lines)


def render_current_labels(labels: Mapping[str, Mapping[str, Any]]) -> str:
    lines = []
    for target in TARGET_FIELDS:
        item = labels.get(target.output_field)
        if not item:
            continue
        lines.append(f"- {target.output_field}: {item.get('code')} | {item.get('label')}")
    return "\n".join(lines) if lines else "- nessuna etichetta ancora scelta"


def record_evidence(record: Mapping[str, Any]) -> str:
    fields = [
        ("luogo_infortunio", "luogo"),
        ("lavorazione", "lavorazione"),
        ("dettaglio_lavorazione", "azione prima dell'evento"),
        ("imprevisto", "evento anomalo/deviazione"),
        ("conseguenza", "lesione/contatto"),
        ("des_infortunio", "descrizione libera"),
    ]
    lines = []
    for field, label in fields:
        value = str(record_value(record, field) or "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def target_decision_guide(target: TargetField) -> str:
    guides = {
        "COD_LOC": (
            "Decidi il luogo fisico dell'evento. Non scegliere il settore lavorativo. "
            "Se il lavoratore e' su/in un veicolo o su strada durante spostamento, valuta prima le classi di trasporto/strada. "
            "Se il testo dichiara esplicitamente 'in cantiere', 'in magazzino', 'in reparto', 'in scuola' o simili, "
            "quel luogo dichiarato pesa piu' dell'oggetto coinvolto nella dinamica."
        ),
        "COD_LAV": (
            "Decidi il tipo di lavoro generale svolto in quel momento. "
            "Non dedurre il settore dell'azienda se il testo descrive solo camminare, uscire, salire, guidare o spostarsi: in quei casi valuta circolazione/trasporto. "
            "Se pero' il testo descrive assistenza a una persona/bambino/ospite, preferisci servizio/cura/assistenza alla persona quando candidato. "
            "Se descrive controllo, stoccaggio, carico o scarico merci in magazzino, valuta magazzinaggio prima di circolazione."
        ),
        "COD_ATV": (
            "Decidi l'azione fisica concreta subito prima dell'imprevisto. "
            "Usa soprattutto 'azione prima dell'evento' e descrizione libera. "
            "Per riposizionare/sollevare/spostare una persona o un carico, valuta trasporto manuale/sollevamento prima di movimenti generici o manipolazione. "
            "Se il rank 1 e' un codice generico 'non precisato' ma il testo esplicita l'azione, preferisci la foglia specifica: "
            "'sistemare pezzi' supporta prendere/afferrare/deporre; 'scaricare dal furgone' supporta trasporto verticale/portare in basso."
        ),
        "COD_MAT_ATV": (
            "Decidi l'agente materiale collegato all'azione fisica specifica. "
            "Se l'azione e' assistere, sollevare, riposizionare o movimentare un paziente/ospite/persona, l'agente e' spesso Umani, non il mobile su cui si trova. "
            "Scegli mobili/veicoli/superfici solo se sono l'oggetto direttamente usato nell'azione. "
            "Per 'pezzi di lamiera preparati' valuta pezzi lavorati/elementi; per 'scaricare materiale/assi/carichi' valuta carichi movimentati a mano; "
            "per attivita' su forno di ricottura valuta macchina a caldo/forno. "
            "Se l'attivita' fisica e' camminare/seguire/spostarsi, non scegliere Umani solo perche' nel contesto c'e' una persona assistita."
        ),
        "COD_DEV": (
            "Decidi l'evento anomalo che rompe il normale svolgimento. "
            "Usa soprattutto il campo 'evento anomalo/deviazione'. "
            "Se c'e' cedimento/dolore interno durante sollevare, portare, alzarsi o caricare, valuta movimento sotto sforzo prima di passo falso/scivolamento. "
            "Se il testo dice 'dal furgone', 'dal forno', 'da un'altezza', 'cadendo da' o 'scendeva da', valuta caduta di persona dall'alto prima di caduta allo stesso livello. "
            "Se dice 'ha messo male il piede', 'cede il ginocchio', 'torsione' senza caduta chiara, valuta passo falso/torsione senza caduta. "
            "Se dice 'scontro fortuito' o collisione tra persone senza caduta, valuta presenza di terzo/pericolo se candidato."
        ),
        "COD_MAT_DEV": (
            "Decidi l'agente materiale coinvolto nell'evento anomalo, non quello della lesione finale. "
            "Per urto contro spigolo/lamiera scegli il materiale urtato; per sforzo durante movimentazione scegli il carico/persona movimentata; per scivolamento scegli la superficie solo se il testo supporta superficie/pavimento. "
            "Trigger utili: bancale/pallet/palette -> accessori per stoccaggio/palette; pezzi di lamiera -> pezzi lavorati/elementi; "
            "staffa di ferro -> elemento costitutivo/strutturale, non necessariamente l'intero macchinario; retro del van/furgone -> veicolo leggero."
        ),
        "COD_CON": (
            "Decidi la modalita' della lesione, cioe' come si produce il danno. "
            "Usa conseguenza e descrizione libera. Se la diagnosi manca, non inventare un contatto specifico: preferisci nessuna informazione se candidata. "
            "Per incidenti stradali/tamponamenti o scontri tra persone valuta urto/collisione con veicolo/persona. "
            "Se il testo dice caduto/cadendo/caduta/sbatteva a terra o sulla spalla, valuta movimento verticale schiacciamento su/contro da caduta prima di contatto duro/abrasivo. "
            "Se invece descrive urto contro spigolo/lamiera senza dinamica di caduta, valuta contatto tagliente/duro/abrasivo."
        ),
        "COD_MAT_CON": (
            "Decidi l'agente materiale che produce direttamente il contatto lesivo. "
            "Non scegliere un oggetto di contesto solo perche' nominato. "
            "Se la lesione e' da sforzo interno senza contatto esterno chiaro, valuta nessun agente materiale se candidato. "
            "Trigger utili: sbatte contro bancale -> palette/bancale; spigolo/pezzo di lamiera -> pezzi lavorati/elementi; "
            "caduta a terra/sulla spalla -> superficie a livello; contatto con ospite/giocatore/persona -> Umani."
        ),
    }
    return guides[target.output_field]


def field_prompt(
    record: Mapping[str, Any],
    target: TargetField,
    candidates: Sequence[Mapping[str, Any]],
    current_labels: Mapping[str, Mapping[str, Any]],
    conservative: bool = False,
) -> str:
    rank_policy = (
        "Il dual encoder ha gia' selezionato candidati forti: il tuo compito e' correggere solo quando la descrizione lo dimostra.\n"
        "3. Controlla il candidato rank 1: se e' compatibile con il testo e nessun altro candidato e' chiaramente migliore, devi mantenere il rank 1.\n"
        "4. Cambia il rank 1 solo quando trovi parole esplicite nel testo che supportano meglio un altro candidato e contraddicono o superano il rank 1.\n"
        "- Il candidato rank 1 del dual encoder e' una baseline forte: cambialo solo se un altro candidato e' chiaramente piu' supportato dal testo.\n"
        if conservative
        else
        "Il dual encoder propone una top-k: usala solo come lista di candidati ammessi, non come verita'.\n"
        "3. Confronta tutti i candidati ammessi contro il testo e scegli quello piu' coerente con la variabile ESAW richiesta.\n"
        "4. Puoi cambiare il rank 1 quando un altro candidato spiega meglio il campo specifico, anche se il rank 1 e' genericamente plausibile.\n"
        "- Non premiare automaticamente il rank 1: il tuo obiettivo e' scegliere il miglior candidato nella top-k.\n"
    )
    return f"""Sei un classificatore ESAW per infortuni sul lavoro in italiano.
Devi rerankare SOLO i candidati proposti dal dual encoder per UNA variabile ESAW.
{rank_policy}

Algoritmo obbligatorio:
1. Identifica quale fase ESAW richiede la variabile: luogo, lavoro, attivita', agente dell'attivita', deviazione, agente della deviazione, contatto, agente del contatto.
2. Leggi la guida specifica della variabile.
5. Prima di rispondere, verifica che selected_code sia il codice che la tua motivazione indica come scelta finale.
6. La lista ranking deve essere ordinata dal migliore al peggiore e ranking[0] deve essere identico a selected_code.

Regole:
- Segui la fase ESAW richiesta: attivita', deviazione e contatto sono fasi diverse.
- Non scegliere la classe piu' "intuitiva" in generale: scegli quella coerente con la variabile richiesta.
- Usa i campi strutturati sotto come prove, dando priorita' al campo indicato nella guida della variabile.
- Scegli solo un codice presente nella lista candidati.
- Se due candidati sono davvero equivalenti per evidenza testuale, preferisci quello con rank_dual_encoder migliore.
- Se un candidato e' piu' specifico ma non e' esplicitamente supportato, preferisci la classe piu' generale o il rank migliore.
- Se il testo non contiene informazione sufficiente per questa variabile, preferisci classi generiche o "nessuna informazione" quando sono candidate.
- Non confondere COD_CON con COD_MAT_CON: COD_CON e' la modalita' del danno, COD_MAT_CON e' l'oggetto/agente che produce il danno.
- Non confondere COD_DEV con COD_MAT_DEV: COD_DEV e' l'evento anomalo, COD_MAT_DEV e' l'oggetto/agente coinvolto nell'evento anomalo.
- Non inventare codici.
- Rispondi solo con JSON valido, senza markdown.

Campi dell'infortunio:
{record_evidence(record)}

Guida specifica per {target.output_field}:
{target_decision_guide(target)}

Etichette gia' scelte o provvisorie per le altre variabili:
{render_current_labels(current_labels)}

Variabile da decidere: {target.output_field} - {target.title}
Significato: {target.role}

Candidati del dual encoder:
{render_candidate_list(candidates)}

Formato risposta:
{{"selected_code":"...", "rank1_compatible":true, "changed_rank1":false, "ranking":["stesso_codice_di_selected_code","altro_codice"], "reason":"breve motivazione con la scelta finale coerente con selected_code"}}"""


def compact_field_prompt(
    record: Mapping[str, Any],
    target: TargetField,
    candidates: Sequence[Mapping[str, Any]],
    current_labels: Mapping[str, Mapping[str, Any]],
    conservative: bool = False,
) -> str:
    del current_labels, conservative
    return f"""Scegli il codice ESAW per una sola variabile.
Ragiona internamente in modo conciso e rispondi SOLO con JSON valido.
Scegli esclusivamente uno dei codici candidati. Non inventare codici.

CASO:
{record_evidence(record)}

VARIABILE: {target.output_field} - {target.title}
SIGNIFICATO: {target.role}

CANDIDATI:
{render_compact_hierarchical_candidate_list(candidates)}

OUTPUT: {{"selected_code":"codice"}}"""


def hierarchical_field_prompt(
    record: Mapping[str, Any],
    target: TargetField,
    candidates: Sequence[Mapping[str, Any]],
    current_labels: Mapping[str, Mapping[str, Any]],
    conservative: bool = False,
) -> str:
    rank_policy = (
        "Il rank 1 del dual encoder e' una baseline forte, ma non deve bloccare una foglia esplicita. "
        "Mantieni il rank 1 se e' specifico e compatibile; cambialo quando il rank 1 e' generico/non precisato "
        "e un altro candidato della stessa macroclasse o di una macroclasse chiaramente provata e' supportato da parole testuali esplicite.\n"
        if conservative
        else
        "Il rank dual encoder e' un indizio, non una verita'. Devi pero' motivare esplicitamente prima la macroclasse e poi la foglia.\n"
    )
    return f"""Sei un classificatore ESAW gerarchico per infortuni sul lavoro in italiano.
Devi scegliere SOLO tra i candidati proposti dal dual encoder per UNA variabile ESAW.

Procedura obbligatoria in due livelli:
1. Decidi la macroclasse/parent ESAW piu' coerente con la variabile richiesta. Non scegliere subito la foglia.
2. Dentro quella macroclasse, scegli la classe foglia piu' coerente.
3. Se il testo supporta solo la macroclasse ma non una foglia specifica, preferisci la classe generica/non precisata o il rank migliore dentro quella macroclasse.
4. Se il testo non contiene informazione sufficiente, non inferire attivita' o agenti materiali impliciti: mantieni "nessuna informazione"/"nessun agente" se e' candidato.
5. Se una foglia specifica sembra semanticamente naturale ma il testo non la prova in modo esplicito, preferisci il codice piu' generale.

{rank_policy}

Regole ESAW:
- Il parent va scelto per codice, non solo per descrizione testuale.
- Per ogni candidato, considera prima il suo parent_code/parent ESAW mostrato nella lista. Esempio generale: un figlio come 41 appartiene al parent 40, un figlio come 63 appartiene al parent 60, e lo stesso criterio vale per tutti gli altri figli candidati.
- Non confrontare subito foglie appartenenti a parent diversi: prima scegli il parent piu' supportato dal testo, poi scegli la foglia migliore solo fra i candidati di quel parent.
- Prima rispetta la fase: attivita', agente attivita', deviazione, agente deviazione, contatto, agente contatto sono variabili diverse.
- COD_DEV descrive l'evento anomalo; COD_CON descrive la modalita' del danno.
- COD_MAT_DEV e COD_MAT_CON sono agenti materiali diversi: non copiare automaticamente l'oggetto nominato nel testo.
- Le classi "non precisato" sono valide quando la macroclasse e' chiara ma la sottoclasse non e' provata.
- Le classi "nessuna informazione"/"nessun agente materiale" sono valide quando il dataset non esplicita un agente o un'azione codificabile.
- Regola anti-generico: se il rank 1 e' una macroclasse generica/non precisata ma il testo nomina chiaramente un'azione, un evento o un agente specifico candidato, preferisci la foglia specifica.
- Caduta da quota vs stesso livello: "dal furgone", "dal forno", "da scala", "da un'altezza", "scendeva da" indicano COD_DEV di caduta dall'alto/da mezzo se candidato; "a terra", "sul pavimento", "camminando e inciampa/scivola" indicano caduta o scivolamento allo stesso livello.
- Agenti materiali ad alta precisione: bancale/pallet/palette -> accessori di stoccaggio/palette; pezzi di lamiera/pezzi preparati/spigolo metallico -> pezzi lavorati o elementi; materiale/assi/carichi scaricati o movimentati -> carichi movimentati a mano; furgone/van -> veicolo leggero; camion/autocarro -> veicolo industriale.
- COD_CON non e' COD_MAT_CON: COD_CON e' la modalita' del danno; COD_MAT_CON e' l'oggetto/superficie/persona del contatto. Per cadute esplicite preferisci la modalita' da caduta/schiacciamento su-contro quando candidata, non solo "materiale duro".
- COD_MAT_ATV, COD_MAT_DEV e COD_MAT_CON non vanno copiati automaticamente: scegli per ciascuno l'agente della fase specifica.
- Rispondi solo con JSON valido, senza markdown.

Campi dell'infortunio:
{record_evidence(record)}

Guida specifica per {target.output_field}:
{target_decision_guide(target)}

Etichette gia' scelte o provvisorie per le altre variabili:
{render_current_labels(current_labels)}

Variabile da decidere: {target.output_field} - {target.title}
Significato: {target.role}

Candidati raggruppati per macroclasse/parent:
{render_hierarchical_candidate_list(candidates)}

Formato risposta:
{{"selected_parent_code":"codice parent scelto, es. 40 o 60", "selected_parent":"macroclasse scelta o label parent", "parent_reason":"perche' questa macroclasse e' corretta", "selected_code":"...", "leaf_reason":"perche' questa foglia e non il generico/rank1", "rank1_compatible":true, "changed_rank1":false, "ranking":["stesso_codice_di_selected_code","altro_codice"], "reason":"sintesi finale coerente con selected_code"}}"""


def consistency_prompt(
    record: Mapping[str, Any],
    choices: Mapping[str, Mapping[str, Any]],
    candidates_by_field: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    candidate_blocks = []
    for target in TARGET_FIELDS:
        candidate_blocks.append(
            f"{target.output_field} ({target.title}; {target.role})\n"
            f"{render_candidate_list(candidates_by_field[target.output_field])}"
        )
    return f"""Sei un revisore ESAW. Hai 8 etichette scelte per lo stesso infortunio.
Devi controllare se nel loro insieme hanno senso rispetto alla dinamica descritta.

Regole:
- Puoi confermare le scelte oppure correggere una o piu' variabili.
- Per ogni variabile puoi scegliere solo codici presenti nella relativa lista candidati.
- Non inventare codici.
- Rispondi solo con JSON valido con una chiave "labels".

Infortunio:
{accident_to_text(record)}

Scelte correnti:
{render_current_labels(choices)}

Candidati disponibili per campo:

{chr(10).join(candidate_blocks)}

Formato risposta:
{{"labels": {{"COD_LOC":"...", "COD_LAV":"...", "COD_ATV":"...", "COD_MAT_ATV":"...", "COD_DEV":"...", "COD_MAT_DEV":"...", "COD_CON":"...", "COD_MAT_CON":"..."}}, "reason":"breve controllo di coerenza"}}"""


def single_record_prompt(
    record: Mapping[str, Any],
    candidates_by_field: Mapping[str, Sequence[Mapping[str, Any]]],
    conservative: bool = False,
    hierarchical: bool = True,
) -> str:
    candidate_blocks = []
    guide_blocks = []
    for target in TARGET_FIELDS:
        field = target.output_field
        renderer = render_compact_hierarchical_candidate_list if hierarchical else render_candidate_list
        candidate_blocks.append(
            f"{field} ({target.title}; {target.role})\n"
            f"{renderer(candidates_by_field[field])}"
        )
        guide_blocks.append(f"{field}: {target_decision_guide(target)}")

    rank_policy = (
        "Il rank 1 del dual encoder e' una baseline forte: cambialo solo se un altro candidato e' chiaramente piu' supportato dal testo."
        if conservative
        else
        "Il rank del dual encoder e' un indizio forte ma non una verita': scegli il candidato piu' coerente per ogni variabile."
    )
    return f"""Sei un classificatore ESAW gerarchico per infortuni sul lavoro in italiano.
Devi scegliere le 8 variabili ESAW dello stesso caso con UNA sola decisione coerente.

Obiettivo:
- Per ogni variabile scegli SOLO un codice presente nei candidati di quella variabile.
- Restituisci un JSON valido, senza markdown e senza testo extra.
- Mantieni coerenza tra evento, agente dell'evento, contatto e agente del contatto.

Procedura obbligatoria:
1. Leggi prima tutti i campi dell'infortunio.
2. Distingui le fasi ESAW: luogo, tipo lavoro, attivita' fisica, agente attivita', deviazione, agente deviazione, contatto, agente contatto.
3. Per ogni variabile scegli prima la macroclasse/parent piu' coerente; poi scegli la foglia migliore tra i candidati di quella macroclasse.
4. Se il testo supporta solo la macroclasse ma non una foglia specifica, preferisci il codice generico/non precisato candidato.
5. Se l'informazione manca, preferisci "nessuna informazione" o "nessun agente materiale" quando sono candidati.
6. Non copiare automaticamente lo stesso agente materiale tra MAT_ATV, MAT_DEV e MAT_CON: possono essere diversi.
7. Non confondere COD_DEV con COD_CON: COD_DEV e' l'evento anomalo; COD_CON e' come avviene la lesione.
8. {rank_policy}

Regole critiche:
- Le gold label umane possono essere troppo generiche: se il testo prova una foglia specifica candidata, sceglila.
- Pero' non inventare dettagli non presenti nel testo.
- Per caduta a terra/sulla spalla: COD_CON spesso e' movimento verticale/schiacciamento su/contro; COD_MAT_CON spesso e' superficie a livello.
- Per sforzo interno senza contatto esterno chiaro: COD_MAT_CON spesso e' nessun agente materiale, se candidato.
- Per collisione/tamponamento/scontro con persona o veicolo: COD_CON e COD_MAT_CON devono riflettere il soggetto/oggetto del contatto.
- Per materiali: bancale/pallet -> accessori di stoccaggio; lamiera/spigolo/pezzo lavorato -> pezzi lavorati/elementi; furgone/van -> veicolo leggero; autocarro/camion -> veicolo industriale.

Campi dell'infortunio:
{record_evidence(record)}

Guide specifiche:
{chr(10).join(guide_blocks)}

Candidati disponibili:

{chr(10).join(candidate_blocks)}

Formato risposta obbligatorio:
{{
  "labels": {{
    "COD_LOC": "codice scelto",
    "COD_LAV": "codice scelto",
    "COD_ATV": "codice scelto",
    "COD_MAT_ATV": "codice scelto",
    "COD_DEV": "codice scelto",
    "COD_MAT_DEV": "codice scelto",
    "COD_CON": "codice scelto",
    "COD_MAT_CON": "codice scelto"
  }},
  "field_reasons": {{
    "COD_LOC": "breve motivo",
    "COD_LAV": "breve motivo",
    "COD_ATV": "breve motivo",
    "COD_MAT_ATV": "breve motivo",
    "COD_DEV": "breve motivo",
    "COD_MAT_DEV": "breve motivo",
    "COD_CON": "breve motivo",
    "COD_MAT_CON": "breve motivo"
  }},
  "consistency_reason": "breve verifica di coerenza globale"
}}"""


def compact_single_record_prompt(
    record: Mapping[str, Any],
    candidates_by_field: Mapping[str, Sequence[Mapping[str, Any]]],
    hierarchical: bool = True,
) -> str:
    candidate_blocks = []
    for target in TARGET_FIELDS:
        field = target.output_field
        renderer = render_compact_hierarchical_candidate_list if hierarchical else render_candidate_list
        candidate_blocks.append(f"{field} ({target.title})\n{renderer(candidates_by_field[field])}")
    output_fields = ", ".join(
        f'"{target.output_field}": "codice"' for target in TARGET_FIELDS
    )
    return f"""Classifica questo infortunio nelle 8 variabili ESAW.
Ragiona internamente in modo conciso, poi rispondi SOLO con JSON valido.
Per ogni campo scegli esattamente un codice dalla relativa lista Top-K; non inventare codici.
COD_DEV e' l'evento anomalo, COD_CON e' il contatto lesivo. I tre agenti materiali possono essere diversi.
Preferisci una classe specifica solo se il testo la dimostra; altrimenti usa la classe generica candidata.

CASO:
{record_evidence(record)}

CANDIDATI:
{chr(10).join(candidate_blocks)}

OUTPUT OBBLIGATORIO:
{{"labels": {{{output_fields}}}}}"""


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    fallback: Dict[str, Any] = {}
    selected_match = re.search(r'"selected_code"\s*:\s*"([^"]+)"', text)
    if selected_match:
        fallback["selected_code"] = selected_match.group(1)
    ranking_match = re.search(r'"ranking"\s*:\s*\[([^\]]*)', text, flags=re.DOTALL)
    if ranking_match:
        ranking = re.findall(r'"([^"]+)"', ranking_match.group(1))
        if ranking:
            fallback["ranking"] = ranking
    rank1_match = re.search(r'"rank1_compatible"\s*:\s*(true|false)', text)
    if rank1_match:
        fallback["rank1_compatible"] = rank1_match.group(1) == "true"
    changed_match = re.search(r'"changed_rank1"\s*:\s*(true|false)', text)
    if changed_match:
        fallback["changed_rank1"] = changed_match.group(1) == "true"
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text, flags=re.DOTALL)
    if reason_match:
        fallback["reason"] = reason_match.group(1)
    if fallback:
        fallback["_partial_json_recovered"] = True

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return fallback
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload
    return fallback


class LLMBackend:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class NoopBackend(LLMBackend):
    def generate(self, prompt: str) -> str:
        return "{}"


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Errore endpoint LLM {exc.code}: {detail}") from exc
        return payload["choices"][0]["message"]["content"]


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        api_base: str,
        max_new_tokens: int,
        temperature: float,
        think: bool,
        num_ctx: int,
        seed: int,
    ) -> None:
        if not model:
            raise ValueError("--llm-model e' obbligatorio con --backend ollama")
        self.model = model
        self.api_base = api_base.rstrip("/")
        if self.api_base.endswith("/v1"):
            self.api_base = self.api_base[:-3]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.think = think
        self.num_ctx = num_ctx
        self.seed = seed
        self.last_metadata: Dict[str, Any] = {}

    def generate(self, prompt: str) -> str:
        self.last_metadata = {}
        thinking_budget_hint = os.environ.get("OLLAMA_THINKING_BUDGET_HINT", "").strip()
        if self.think and thinking_budget_hint:
            prompt = (
                "ISTRUZIONE SUL RAGIONAMENTO: mantieni il thinking interno conciso, "
                f"entro circa {thinking_budget_hint} token. Concludi sempre con la risposta finale richiesta.\n\n"
                + prompt
            )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.think,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "num_ctx": self.num_ctx,
                "seed": self.seed,
            },
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            timeout_seconds = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT_SECONDS", "180"))
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Errore endpoint Ollama {exc.code}: {detail}") from exc
        message = payload.get("message", {})
        thinking_text = str(message.get("thinking", ""))
        content_text = str(message.get("content", ""))
        self.last_metadata = {
            "thinking_enabled": bool(self.think),
            "thinking_present": bool(thinking_text.strip()),
            "thinking_chars": len(thinking_text),
            "thinking_words": len(thinking_text.split()),
            "content_chars": len(content_text),
            "prompt_eval_count": payload.get("prompt_eval_count"),
            "eval_count": payload.get("eval_count"),
            "total_duration_ns": payload.get("total_duration"),
            "load_duration_ns": payload.get("load_duration"),
            "prompt_eval_duration_ns": payload.get("prompt_eval_duration"),
            "eval_duration_ns": payload.get("eval_duration"),
            "done_reason": payload.get("done_reason"),
        }
        return content_text

    def unload(self) -> None:
        body = {
            "model": self.model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=30).read()
        except Exception:
            pass


class TransformersBackend(LLMBackend):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        torch_dtype: str,
        max_new_tokens: int,
        temperature: float,
        trust_remote_code: bool,
        local_files_only: bool,
    ) -> None:
        if not model_name_or_path:
            raise ValueError("--llm-model e' obbligatorio con --backend transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype_map[torch_dtype],
            device_map=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            model_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            model_input = prompt
        outputs = self.pipe(
            model_input,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-6),
            return_full_text=False,
        )
        return str(outputs[0]["generated_text"])


def make_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend in {"noop", "oracle"}:
        return NoopBackend()
    if args.backend == "openai-compatible":
        if not args.llm_model:
            raise ValueError("--llm-model e' obbligatorio con --backend openai-compatible")
        api_key = os.environ.get(args.api_key_env, "EMPTY")
        return OpenAICompatibleBackend(
            model=args.llm_model,
            api_base=args.api_base,
            api_key=api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    if args.backend == "ollama":
        return OllamaBackend(
            model=args.llm_model,
            api_base=args.api_base,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            think=args.ollama_think_level or args.ollama_think,
            num_ctx=args.ollama_num_ctx,
            seed=args.seed,
        )
    if args.backend == "transformers":
        return TransformersBackend(
            model_name_or_path=args.llm_model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            trust_remote_code=args.trust_remote_code,
            local_files_only=not args.allow_download,
        )
    raise ValueError(f"Backend non supportato: {args.backend}")


def valid_code_from_response(
    response: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    fallback: str,
) -> str:
    valid_codes = {str(item["code"]) for item in candidates}
    selected = str(response.get("selected_code", "")).strip()
    if selected in valid_codes:
        return selected
    ranking = response.get("ranking")
    if isinstance(ranking, list):
        for item in ranking:
            code = str(item).strip()
            if code in valid_codes:
                return code
    return fallback


def valid_code_from_labels(
    labels: Any,
    field: str,
    candidates: Sequence[Mapping[str, Any]],
    fallback: str,
) -> str:
    if isinstance(labels, Mapping):
        selected = str(labels.get(field, "")).strip()
        if selected in {str(item["code"]) for item in candidates}:
            return selected
    return fallback


def item_by_code(candidates: Sequence[Mapping[str, Any]], code: str) -> Mapping[str, Any]:
    for item in candidates:
        if item["code"] == code:
            return item
    return candidates[0]


def lower_text(value: Any) -> str:
    return str(value or "").lower()


def is_missing_or_absent_code(code: str, label: str = "") -> bool:
    normalized_label = lower_text(label)
    return code in {"00", "000", "00.00.00.00", "00.01.00.00", "00.02.00.00"} or (
        "nessuna informazione" in normalized_label
    )


def accept_esaw_rule_change(
    record: Mapping[str, Any],
    target: TargetField,
    dual_item: Mapping[str, Any],
    llm_item: Mapping[str, Any],
) -> tuple[bool, str]:
    dual_code = str(dual_item["code"])
    llm_code = str(llm_item["code"])
    if llm_code == dual_code:
        return True, "unchanged"

    text = lower_text(record.get("text") or accident_to_text(record))
    dual_label = str(dual_item.get("label", ""))

    if is_missing_or_absent_code(dual_code, dual_label):
        return False, "blocked_change_from_missing_or_absent_rank1"

    field = target.output_field
    if field == "COD_LOC" and llm_code == "081" and any(
        marker in text for marker in ("palestra", "piscina", "sala sport", "scienze motorie")
    ):
        return True, "accepted_sports_indoor_location"

    if field == "COD_DEV" and llm_code == "52" and any(
        marker in text for marker in ("cade a terra", "caduta", "scivola", "inciampa")
    ):
        return True, "accepted_explicit_slip_trip_fall"

    if field in {"COD_MAT_ATV", "COD_MAT_DEV", "COD_MAT_CON"}:
        if llm_code == "14.03.00.00" and any(
            marker in text for marker in ("lamiera", "spigolo", "pezzo lavorato", "pezzi di lamiera")
        ):
            return True, "accepted_sheet_metal_workpiece"
        if llm_code == "14.08.00.00" and any(
            marker in text for marker in ("scatolone", "cartone", "imball", "stocc")
        ):
            return True, "accepted_stored_package"

    return False, "blocked_no_high_precision_rule"


def safe_generate(backend: LLMBackend, prompt: str, retries: int) -> tuple[str, Optional[str]]:
    attempts = max(1, retries + 1)
    last_error: Optional[str] = None
    for attempt in range(1, attempts + 1):
        try:
            return backend.generate(prompt), None
        except Exception as exc:
            last_error = str(exc)
            print(f"LLM error attempt {attempt}/{attempts}: {last_error}", file=sys.stderr)
            unload = getattr(backend, "unload", None)
            if callable(unload):
                unload()
            if attempt < attempts:
                time.sleep(min(10, 2 * attempt))
    return "", last_error


def gold_codes_for_record(record: Mapping[str, Any]) -> Dict[str, str]:
    return {
        target.output_field: label_to_output_code(target, record_value(record, target.field))
        for target in TARGET_FIELDS
    }


def score_predictions(
    rows: Sequence[Mapping[str, Any]],
    prediction_key: str,
) -> Dict[str, Any]:
    field_hits = {target.output_field: 0 for target in TARGET_FIELDS}
    record_hits = 0
    for row in rows:
        all_correct = True
        gold = row["gold"]
        predictions = row[prediction_key]
        for target in TARGET_FIELDS:
            field = target.output_field
            hit = predictions[field]["code"] == gold[field]
            field_hits[field] += int(hit)
            all_correct = all_correct and hit
        record_hits += int(all_correct)

    total = max(1, len(rows))
    field_metrics = {
        field: {"top1_accuracy": round(hits / total, 4), "hits": hits, "records": len(rows)}
        for field, hits in field_hits.items()
    }
    mean_top1 = round(sum(item["top1_accuracy"] for item in field_metrics.values()) / len(field_metrics), 4)
    return {
        "records": len(rows),
        "mean_top1_accuracy": mean_top1,
        "record_exact@1": round(record_hits / total, 4),
        "record_hits": record_hits,
        "field_metrics": field_metrics,
    }


def load_completed_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("record_key") or "")
            if key:
                completed[key] = row
    return completed


def load_candidate_input(path: Path, candidate_top_k: int) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    by_record: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_key_value = str(row.get("record_key") or "")
            if not record_key_value:
                continue

            if isinstance(row.get("dual_candidates"), Mapping):
                by_record.setdefault(record_key_value, {})
                for field, candidates in row["dual_candidates"].items():
                    if isinstance(candidates, list):
                        by_record[record_key_value][str(field)] = [dict(item) for item in candidates[:candidate_top_k]]
                continue

            field = str(row.get("field") or "")
            if not field:
                continue
            candidates = None
            for key in ("rerank_top5_candidates", "rerank_top_candidates", "candidates"):
                value = row.get(key)
                if isinstance(value, list):
                    candidates = value
                    break
            if candidates is None:
                continue
            normalized: List[Dict[str, Any]] = []
            for index, item in enumerate(candidates[:candidate_top_k], start=1):
                candidate = dict(item.get("candidate", item)) if isinstance(item, Mapping) else {}
                if not candidate.get("code"):
                    continue
                candidate.setdefault("rank_dual_encoder", int(candidate.get("rank", index)))
                candidate.setdefault("score", float(candidate.get("rerank_score", candidate.get("retriever_score", 0.0))))
                normalized.append(candidate)
            if normalized:
                by_record.setdefault(record_key_value, {})[field] = normalized
    return by_record


def code_present_metrics(rows: Sequence[Mapping[str, Any]], candidate_key: str) -> Dict[str, Any]:
    field_hits = {target.output_field: 0 for target in TARGET_FIELDS}
    record_hits = 0
    for row in rows:
        all_present = True
        for target in TARGET_FIELDS:
            field = target.output_field
            codes = {item["code"] for item in row[candidate_key][field]}
            hit = row["gold"][field] in codes
            field_hits[field] += int(hit)
            all_present = all_present and hit
        record_hits += int(all_present)
    total = max(1, len(rows))
    return {
        "records": len(rows),
        "mean_recall": round(sum(hits / total for hits in field_hits.values()) / len(field_hits), 4),
        "record_recall": round(record_hits / total, 4),
        "field_recall": {
            field: round(hits / total, 4)
            for field, hits in field_hits.items()
        },
    }


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()
    rng = random.Random(args.seed)
    model_dir = Path(args.model)
    text_format = resolve_text_format(model_dir, args.text_format)

    records = filter_records_by_split(load_dataset(args.dataset), args.split_file, args.split)
    records = complete_target_records(records)
    if args.num_shards < 1:
        raise ValueError("--num-shards deve essere >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index deve essere compreso tra 0 e num-shards - 1")
    if args.sample_size and len(records) > args.sample_size:
        records = rng.sample(records, args.sample_size)
    if args.num_shards > 1:
        records = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_index]

    candidate_input: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if args.candidate_input_jsonl:
        candidate_input = load_candidate_input(Path(args.candidate_input_jsonl), args.candidate_top_k)
        complete_candidate_keys = {
            key
            for key, fields_for_record in candidate_input.items()
            if all(
                target.output_field in fields_for_record and fields_for_record[target.output_field]
                for target in TARGET_FIELDS
            )
        }
        records = [
            record
            for index, record in enumerate(records)
            if record_key(record, index) in complete_candidate_keys
        ]
        if not records:
            raise ValueError(f"--candidate-input-jsonl non contiene record completi per lo split {args.split}")

    rows: List[Dict[str, Any]] = [
        {
            "record_index": index,
            "record_key": record_key(record, index),
            "text": accident_to_text(record),
            "gold": gold_codes_for_record(record),
            "dual_candidates": {},
            "dual_top1": {},
            "llm_top1": {},
            "llm_raw": {},
        }
        for index, record in enumerate(records)
    ]

    candidate_source = "dual_encoder"
    if args.candidate_input_jsonl:
        candidate_source = args.candidate_input_jsonl
        missing: List[str] = []
        for row in rows:
            fields_for_record = candidate_input.get(row["record_key"], {})
            for target in TARGET_FIELDS:
                field = target.output_field
                top_items = fields_for_record.get(field, [])
                if not top_items:
                    missing.append(f"{row['record_key']}:{field}")
                    continue
                row["dual_candidates"][field] = top_items[: args.candidate_top_k]
                row["dual_top1"][field] = dict(item_by_code(row["dual_candidates"][field], row["dual_candidates"][field][0]["code"]))
        if missing:
            preview = ", ".join(missing[:20])
            raise ValueError(
                f"--candidate-input-jsonl incompleto: mancano {len(missing)} record/campi. Primi mancanti: {preview}"
            )
    else:
        model = load_dual_encoder(args.model, device=args.device)
        if args.rebuild_candidates:
            candidates = build_candidates(build_explicit_taxonomy(load_taxonomy(args.taxonomy)), text_format=text_format)
        else:
            candidates = load_candidates(model_dir, args.taxonomy, text_format)

        for target in TARGET_FIELDS:
            field_candidates = candidates[target.field]
            candidate_embeddings = model.encode(
                [candidate.text for candidate in field_candidates],
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            query_embeddings = model.encode(
                [query_text(record, target, text_format) for record in records],
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=True,
            )
            scores = cosine_matrix(query_embeddings, candidate_embeddings)
            detail_top_k = max(args.candidate_top_k * 3, args.candidate_top_k)
            for row_index, row in enumerate(rows):
                details = top_candidate_details(scores[row_index], field_candidates, detail_top_k)
                top_items = pick_best_unique(details, args.candidate_top_k)
                row["dual_candidates"][target.output_field] = top_items
                row["dual_top1"][target.output_field] = dict(item_by_code(top_items, top_items[0]["code"]))

        del model
        gc.collect()

    backend = make_backend(args)
    completed_by_key: Dict[str, Dict[str, Any]] = {}
    prediction_handle = None
    if args.predictions_output:
        pred_path = Path(args.predictions_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume_predictions:
            completed_by_key = load_completed_predictions(pred_path)
        prediction_handle = pred_path.open("a" if args.resume_predictions else "w", encoding="utf-8")

    completed_this_run = 0
    stopped_for_runtime = False
    try:
        for row_index, (record, row) in enumerate(zip(records, rows), start=1):
            if row["record_key"] in completed_by_key:
                rows[row_index - 1] = completed_by_key[row["record_key"]]
                continue

            if args.max_runtime_minutes > 0:
                elapsed_minutes = (time.monotonic() - started_at) / 60.0
                if elapsed_minutes >= args.max_runtime_minutes:
                    stopped_for_runtime = True
                    print(
                        f"Stopping before record {row_index}/{len(rows)} after {elapsed_minutes:.2f} minutes",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

            current_choices: Dict[str, Mapping[str, Any]] = {
                target.output_field: row["dual_top1"][target.output_field]
                for target in TARGET_FIELDS
            }
            if args.single_record_prompt and args.backend not in {"noop", "oracle"}:
                prompt_builder = (
                    compact_single_record_prompt
                    if args.compact_single_record_prompt
                    else single_record_prompt
                )
                prompt_kwargs = {"hierarchical": args.hierarchical_rerank}
                if not args.compact_single_record_prompt:
                    prompt_kwargs["conservative"] = args.conservative_rerank
                raw, error = safe_generate(
                    backend,
                    prompt_builder(
                        record,
                        row["dual_candidates"],
                        **prompt_kwargs,
                    ),
                    args.llm_retries,
                )
                response = extract_json_object(raw)
                response["_raw"] = raw
                generation_metadata = getattr(backend, "last_metadata", None)
                if isinstance(generation_metadata, Mapping):
                    response["_generation_metadata"] = dict(generation_metadata)
                if not error and not raw.strip():
                    done_reason = (
                        str(generation_metadata.get("done_reason", ""))
                        if isinstance(generation_metadata, Mapping)
                        else ""
                    )
                    error = (
                        "empty final response after native thinking"
                        + (f" (done_reason={done_reason})" if done_reason else "")
                    )
                if not error and not isinstance(response.get("labels"), Mapping):
                    error = "missing labels object in final LLM response"
                if error:
                    response["_error"] = error
                    response["reason"] = "fallback to dual encoder rank 1 after LLM error"
                labels = response.get("labels")
                field_reasons = response.get("field_reasons")
                for target in TARGET_FIELDS:
                    field = target.output_field
                    candidates_for_field = row["dual_candidates"][field]
                    fallback = str(candidates_for_field[0]["code"])
                    selected_code = valid_code_from_labels(labels, field, candidates_for_field, fallback)
                    selected_item = dict(item_by_code(candidates_for_field, selected_code))
                    field_response = {
                        "selected_code": selected_code,
                        "reason": (
                            str(field_reasons.get(field, ""))
                            if isinstance(field_reasons, Mapping)
                            else str(response.get("consistency_reason") or response.get("reason") or "")
                        ),
                        "_single_record_prompt": True,
                    }
                    if error:
                        field_response["_error"] = error
                    if args.acceptance_policy == "esaw-rules":
                        accepted, acceptance_reason = accept_esaw_rule_change(
                            record,
                            target,
                            row["dual_top1"][field],
                            selected_item,
                        )
                        field_response["_acceptance_policy"] = args.acceptance_policy
                        field_response["_acceptance_reason"] = acceptance_reason
                        field_response["_accepted_change"] = accepted
                        if not accepted:
                            selected_item = dict(row["dual_top1"][field])
                            field_response["_selected_before_policy"] = selected_code
                            selected_code = str(selected_item["code"])
                            field_response["selected_code"] = selected_code
                    row["llm_top1"][field] = selected_item
                    row["llm_raw"][field] = field_response
                    current_choices[field] = selected_item
                row["llm_raw"]["single_record"] = response
            else:
                for target in TARGET_FIELDS:
                    field = target.output_field
                    candidates_for_field = row["dual_candidates"][field]
                    fallback = str(candidates_for_field[0]["code"])

                    if args.backend == "oracle":
                        selected_code = row["gold"][field] if row["gold"][field] in {c["code"] for c in candidates_for_field} else fallback
                        response = {"selected_code": selected_code, "reason": "oracle ceiling"}
                    elif args.backend == "noop":
                        selected_code = fallback
                        response = {"selected_code": selected_code, "reason": "dual encoder unchanged"}
                    else:
                        raw, error = safe_generate(
                            backend,
                            (
                                compact_field_prompt
                                if args.compact_field_prompt
                                else (
                                    hierarchical_field_prompt
                                    if args.hierarchical_rerank
                                    else field_prompt
                                )
                            )(
                                record,
                                target,
                                candidates_for_field,
                                current_choices,
                                conservative=args.conservative_rerank,
                            ),
                            args.llm_retries,
                        )
                        response = extract_json_object(raw)
                        response["_raw"] = raw
                        generation_metadata = getattr(backend, "last_metadata", None)
                        if isinstance(generation_metadata, Mapping):
                            response["_generation_metadata"] = dict(generation_metadata)
                        if not error and not raw.strip():
                            done_reason = (
                                str(generation_metadata.get("done_reason", ""))
                                if isinstance(generation_metadata, Mapping)
                                else ""
                            )
                            error = (
                                "empty final response after native thinking"
                                + (f" (done_reason={done_reason})" if done_reason else "")
                            )
                        if not error and not response.get("selected_code"):
                            error = "missing selected_code in final LLM response"
                        if error:
                            response["_error"] = error
                            response["selected_code"] = fallback
                            response["reason"] = "fallback to dual encoder rank 1 after LLM error"
                        selected_code = valid_code_from_response(response, candidates_for_field, fallback)

                    selected_item = dict(item_by_code(candidates_for_field, selected_code))
                    if args.acceptance_policy == "esaw-rules":
                        accepted, acceptance_reason = accept_esaw_rule_change(
                            record,
                            target,
                            row["dual_top1"][field],
                            selected_item,
                        )
                        response["_acceptance_policy"] = args.acceptance_policy
                        response["_acceptance_reason"] = acceptance_reason
                        response["_accepted_change"] = accepted
                        if not accepted:
                            selected_item = dict(row["dual_top1"][field])
                            response["_selected_before_policy"] = selected_code
                            selected_code = str(selected_item["code"])
                    row["llm_top1"][field] = selected_item
                    row["llm_raw"][field] = response
                    current_choices[field] = selected_item

            if not args.single_record_prompt and not args.no_consistency_pass and args.backend not in {"noop", "oracle"}:
                raw, error = safe_generate(
                    backend,
                    consistency_prompt(record, row["llm_top1"], row["dual_candidates"]),
                    args.llm_retries,
                )
                response = extract_json_object(raw)
                response["_raw"] = raw
                if error:
                    response["_error"] = error
                labels = response.get("labels")
                if isinstance(labels, dict):
                    for target in TARGET_FIELDS:
                        field = target.output_field
                        code = str(labels.get(field, "")).strip()
                        valid = {item["code"] for item in row["dual_candidates"][field]}
                        if code in valid:
                            row["llm_top1"][field] = dict(item_by_code(row["dual_candidates"][field], code))
                row["llm_raw"]["consistency"] = response

            if prediction_handle is not None:
                prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                prediction_handle.flush()
                completed_by_key[row["record_key"]] = row
                completed_this_run += 1

            if args.ollama_unload_every > 0 and completed_this_run > 0:
                if completed_this_run % args.ollama_unload_every == 0:
                    unload = getattr(backend, "unload", None)
                    if callable(unload):
                        unload()

            if row_index % 25 == 0:
                print(f"Processed {row_index}/{len(rows)} records", file=sys.stderr)
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

    completed_rows = [
        row
        for row in rows
        if row.get("llm_top1") and all(target.output_field in row["llm_top1"] for target in TARGET_FIELDS)
    ]
    payload = {
        "model": args.model,
        "llm_model": args.llm_model,
        "backend": args.backend,
        "split": args.split,
        "split_file": args.split_file,
        "sample_size": args.sample_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "records": len(completed_rows),
        "target_records": len(rows),
        "completed_records_total": len(completed_rows),
        "completed_records_this_run": completed_this_run,
        "complete": len(completed_rows) == len(rows),
        "stopped_for_runtime": stopped_for_runtime,
        "text_format": text_format,
        "candidate_top_k": args.candidate_top_k,
        "candidate_source": candidate_source,
        "consistency_pass": not args.no_consistency_pass,
        "conservative_rerank": args.conservative_rerank,
        "hierarchical_rerank": args.hierarchical_rerank,
        "single_record_prompt": args.single_record_prompt,
        "compact_single_record_prompt": args.compact_single_record_prompt,
        "compact_field_prompt": args.compact_field_prompt,
        "acceptance_policy": args.acceptance_policy,
        "ollama_num_ctx": args.ollama_num_ctx if args.backend == "ollama" else None,
        "ollama_think": (args.ollama_think_level or args.ollama_think) if args.backend == "ollama" else None,
        "ollama_thinking_budget_hint": (
            os.environ.get("OLLAMA_THINKING_BUDGET_HINT", "") if args.backend == "ollama" else None
        ),
        "temperature": args.temperature,
        "seed": args.seed,
        "dual_encoder": score_predictions(completed_rows, "dual_top1"),
        "llm_rerank": score_predictions(completed_rows, "llm_top1"),
        f"dual_recall@{args.candidate_top_k}": code_present_metrics(completed_rows, "dual_candidates"),
    }
    payload["delta"] = {
        "mean_top1_accuracy": round(
            payload["llm_rerank"]["mean_top1_accuracy"] - payload["dual_encoder"]["mean_top1_accuracy"],
            4,
        ),
        "record_exact@1": round(
            payload["llm_rerank"]["record_exact@1"] - payload["dual_encoder"]["record_exact@1"],
            4,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
