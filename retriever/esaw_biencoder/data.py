from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .schema import INPUT_FIELDS, TARGET_FIELDS, TargetField

TEXT_FORMAT_LEGACY = "legacy"
TEXT_FORMAT_V5 = "v5"
TEXT_FORMAT_V6 = "v6"
TEXT_FORMAT_V7 = "v7"
TEXT_FORMATS = (TEXT_FORMAT_LEGACY, TEXT_FORMAT_V5, TEXT_FORMAT_V6, TEXT_FORMAT_V7)

QWEN_ESAW_RETRIEVAL_INSTRUCTION = (
    "Given an Italian occupational accident report, retrieve the correct ESAW taxonomy class "
    "for the requested accident variable."
)


@dataclass(frozen=True)
class TaxonomyEntry:
    taxonomy: str
    code: str
    output_code: str
    label: str
    parent_code: Optional[str]
    parent_label: Optional[str]
    ancestors: List[str]
    text: str


@dataclass(frozen=True)
class Candidate:
    field: str
    output_field: str
    code: str
    taxonomy_code: str
    label: str
    text: str
    parent_code: Optional[str]
    parent_label: Optional[str]
    ancestors: List[str]
    prototype_id: str = "base"


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list) and results:
            items = results[0].get("items")
            if isinstance(items, list):
                return items
    raise ValueError(f"Formato dataset non riconosciuto: {path}")


def load_taxonomy(path: str | Path) -> Dict[str, Dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Formato tassonomia non riconosciuto: {path}")
    return payload


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip()


def record_value(record: Mapping[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    upper = field.upper()
    if upper in record:
        return record[upper]
    lower = field.lower()
    if lower in record:
        return record[lower]
    return None


def material_taxonomy_code(code: str) -> str:
    parts = normalize_value(code).split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return normalize_value(code)


def material_output_code(taxonomy_code: str) -> str:
    parts = taxonomy_code.split(".")
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}.00.00"
    return taxonomy_code


def parent_code_for(taxonomy_name: str, code: str, taxonomy: Mapping[str, str]) -> Optional[str]:
    if taxonomy_name == "agente_materiale":
        group = code.split(".")[0]
        parent = f"{group}.00"
        return parent if parent in taxonomy and parent != code else None

    if len(code) == 3:
        parent = f"{code[:2]}0"
    elif len(code) == 2:
        parent = f"{code[0]}0"
    else:
        return None
    return parent if parent in taxonomy and parent != code else None


def build_ancestors(taxonomy_name: str, code: str, taxonomy: Mapping[str, str]) -> List[str]:
    ancestors: List[str] = []
    current = parent_code_for(taxonomy_name, code, taxonomy)
    while current:
        ancestors.append(current)
        current = parent_code_for(taxonomy_name, current, taxonomy)
    return ancestors


def make_taxonomy_entry(
    taxonomy_name: str,
    code: str,
    label: str,
    taxonomy: Mapping[str, str],
) -> TaxonomyEntry:
    ancestors = build_ancestors(taxonomy_name, code, taxonomy)
    parent_code = ancestors[0] if ancestors else None
    parent_label = taxonomy.get(parent_code) if parent_code else None
    hierarchy_labels = [taxonomy[a] for a in reversed(ancestors) if a in taxonomy]
    output_code = material_output_code(code) if taxonomy_name == "agente_materiale" else code
    text_parts = [
        f"Tassonomia: {taxonomy_name.replace('_', ' ')}.",
        f"Codice: {output_code}.",
        f"Descrizione classe: {label}.",
    ]
    if hierarchy_labels:
        text_parts.append("Gerarchia: " + " > ".join(hierarchy_labels + [label]) + ".")
    return TaxonomyEntry(
        taxonomy=taxonomy_name,
        code=code,
        output_code=output_code,
        label=label,
        parent_code=parent_code,
        parent_label=parent_label,
        ancestors=ancestors,
        text=" ".join(text_parts),
    )


def build_explicit_taxonomy(raw_taxonomy: Mapping[str, Mapping[str, str]]) -> Dict[str, List[TaxonomyEntry]]:
    explicit: Dict[str, List[TaxonomyEntry]] = {}
    for taxonomy_name, taxonomy in raw_taxonomy.items():
        explicit[taxonomy_name] = [
            make_taxonomy_entry(taxonomy_name, code, label, taxonomy)
            for code, label in sorted(taxonomy.items())
        ]
    return explicit


def accident_to_text(record: Mapping[str, Any]) -> str:
    labels = {
        "luogo_infortunio": "Luogo infortunio",
        "lavorazione": "Lavorazione",
        "dettaglio_lavorazione": "Dettaglio lavorazione",
        "imprevisto": "Imprevisto",
        "conseguenza": "Conseguenza",
        "des_infortunio": "Descrizione infortunio",
    }
    chunks = []
    for field in INPUT_FIELDS:
        value = normalize_value(record_value(record, field))
        if value:
            chunks.append(f"{labels[field]}: {value}")
    return ". ".join(chunks)


def normalize_text_format(text_format: str | None) -> str:
    if not text_format:
        return TEXT_FORMAT_LEGACY
    normalized = text_format.strip().lower()
    if normalized not in TEXT_FORMATS:
        raise ValueError(f"Formato testo non supportato: {text_format}")
    return normalized


def query_text(record: Mapping[str, Any], target: TargetField, text_format: str = TEXT_FORMAT_LEGACY) -> str:
    accident = accident_to_text(record)
    normalized_text_format = normalize_text_format(text_format)
    if normalized_text_format == TEXT_FORMAT_V6:
        return (
            f"Instruct: {QWEN_ESAW_RETRIEVAL_INSTRUCTION}\n"
            f"Query: Variabile ESAW: {target.title}. "
            f"Definizione della variabile: {target.role}. "
            f"Infortunio: {accident}."
        )
    if normalized_text_format == TEXT_FORMAT_V5:
        return (
            f"Variabile ESAW da classificare: {target.title}. "
            f"Criterio: {target.role}. "
            f"Infortunio: {accident}."
        )
    return (
        f"Infortunio da classificare. {accident}. "
        f"Variabile ESAW richiesta: {target.title}. "
        f"Significato della variabile: {target.role}."
    )


def candidate_text(entry: TaxonomyEntry, target: TargetField, text_format: str = TEXT_FORMAT_LEGACY) -> str:
    normalized_text_format = normalize_text_format(text_format)
    if normalized_text_format == TEXT_FORMAT_V7:
        if target.taxonomy == "agente_materiale":
            parts = [
                f"Agente materiale ESAW per {target.title}: {entry.label}.",
                f"Ruolo nella dinamica dell'infortunio: {target.role}.",
            ]
            if entry.parent_label:
                parts.append(f"Famiglia materiale: {entry.parent_label}.")
            parts.append(
                "Usare questa classe quando l'oggetto, la superficie, la sostanza, "
                "la macchina, il veicolo, la persona o l'ambiente indicato nel testo "
                "e' l'agente materiale rilevante per questa variabile."
            )
            parts.append(f"Codice ESAW: {entry.output_code}.")
            return " ".join(parts)
        return candidate_text(entry, target, TEXT_FORMAT_V5)
    if normalized_text_format == TEXT_FORMAT_V6:
        parts = [
            f"ESAW class document for {target.title}.",
            f"Correct when the accident variable means: {target.role}.",
            f"Class label: {entry.label}.",
        ]
        if entry.parent_label:
            parts.append(f"Parent or macro class: {entry.parent_label}.")
        if "non precisat" in entry.label.lower():
            parts.append("Specificity: generic class for cases where the subclass is not explicit.")
        elif "nessuna informazione" in entry.label.lower():
            parts.append("Specificity: use when the report does not provide enough information.")
        elif "nessun " in entry.label.lower():
            parts.append("Specificity: use when the report explicitly indicates absence of this element.")
        else:
            parts.append("Specificity: use only when this specific class is supported by the report.")
        parts.append(f"Official ESAW code: {entry.output_code}.")
        return " ".join(parts)
    if normalized_text_format == TEXT_FORMAT_V5:
        parts = [
            f"Classe ESAW per {target.title}: {entry.label}.",
            f"Ruolo: {target.role}.",
        ]
        if entry.parent_label:
            parts.append(f"Macroclasse: {entry.parent_label}.")
        parts.append(f"Codice ESAW: {entry.output_code}.")
        return " ".join(parts)
    return (
        f"Variabile ESAW: {target.title}. "
        f"Ruolo nella dinamica: {target.role}. "
        f"{entry.text}"
    )


MATERIAL_MACRO_HINTS: Dict[str, str] = {
    "00": "assenza di agente materiale, nessun oggetto coinvolto, informazione mancante, agente non identificabile",
    "01": "pavimenti, suolo, terreno, superfici di passaggio, muri, porte, ostacoli fissi a livello",
    "02": "scale, tetti, coperture, terrazze, passerelle, impalcature, piattaforme e superfici in altezza",
    "03": "scavi, buche, fosse, pozzi, trincee, aperture e superfici in profondita",
    "04": "tubi, canalizzazioni, condotte, sistemi di alimentazione e distribuzione di sostanze",
    "05": "motori, organi di trasmissione, ingranaggi, energia accumulata, parti meccaniche in movimento",
    "06": "utensili manuali non motorizzati, martello, sega, coltello, forbici, cacciavite, pinza",
    "07": "utensili manuali motorizzati, trapano, smerigliatrice, motosega, avvitatore, martello demolitore",
    "08": "utensile manuale senza informazione sulla motorizzazione",
    "09": "macchine portatili o mobili, macchine agricole mobili, macchine di cantiere, pulitrici mobili",
    "10": "macchine fisse, presse, torni, seghe, macchine industriali, impianti di lavorazione",
    "11": "nastri trasportatori, gru, carrelli, muletti, montacarichi, scaffali, pallet, imballaggi",
    "12": "veicoli terrestri, autocarri, camion, automobili, carrelli elevatori, motocicli, biciclette",
    "13": "mezzi di trasporto non terrestri, imbarcazioni, aeromobili, veicoli ferroviari",
    "14": "materiali, oggetti, prodotti, carichi, frammenti, polveri, pezzi, componenti, imballaggi",
    "15": "sostanze chimiche, corrosive, tossiche, infiammabili, esplosive, gas, vapori, liquidi",
    "16": "dispositivi di sicurezza, protezioni, DPI, imbracature, barriere e attrezzature protettive",
    "17": "attrezzature d'ufficio, oggetti personali, sportivi, domestici, armi, apparecchiature comuni",
    "18": "persone, colleghi, pazienti, clienti, animali, piante, organismi viventi, agenti biologici",
    "19": "residui alla rinfusa, rifiuti, ammassi di materiale, scarti non confezionati",
    "20": "fenomeni naturali e fisici, pioggia, ghiaccio, neve, vento, acqua, fango, rumore, luce",
    "99": "altro agente materiale non classificato, oggetto o situazione materiale diversa dalle classi note",
}


MATERIAL_CODE_HINTS: Dict[str, str] = {
    "00.01": "nessun agente materiale; la dinamica non coinvolge oggetti, superfici, macchine, persone o sostanze come agente",
    "00.02": "nessuna informazione sull'agente materiale; il testo non consente di identificarlo",
    "01.01": "porte, muri, pareti, finestre, vetrate, ostacoli fissi, elementi di edificio a livello",
    "01.02": "suolo, pavimento, terreno, superficie di calpestio, corridoio, piazzale, campo, pavimento bagnato o ingombro",
    "01.03": "superficie galleggiante a livello, pontone, piattaforma galleggiante, superficie su acqua",
    "02.01": "scale, rampe, tetti, terrazze, aperture in quota, coperture, parti fisse di edificio in altezza",
    "02.02": "passerelle, scale fisse, tralicci, strutture fisse in quota",
    "02.03": "impalcature, scale mobili, piattaforme sospese, piattaforme elevatrici mobili",
    "02.04": "impalcature temporanee, imbracature, seggiolini, strutture provvisorie in altezza",
    "06.02": "coltello, forbici, cesoie, potatore, utensile manuale per tagliare o separare",
    "06.05": "trapano manuale, cacciavite, chiave, utensile per forare, stringere, avvitare",
    "06.09": "zappa, vanga, pala, piccone, rastrello, utensile agricolo manuale o per suolo",
    "07.02": "cesoia elettrica, forbici motorizzate, utensile meccanizzato per taglio",
    "07.05": "trapano, avvitatore, utensile elettrico per forare o avvitare",
    "07.09": "martello demolitore, utensile agricolo motorizzato, attrezzo meccanizzato per lavorare il suolo",
    "09.02": "macchina agricola mobile, trattore attrezzato, falciatrice, macchina mobile per lavorazione del suolo",
    "09.03": "macchina mobile da cantiere, escavatore, betoniera mobile, attrezzatura mobile di costruzione",
    "09.04": "lavasciuga, macchina mobile per pulizia pavimenti",
    "10.07": "pressa, schiacciatrice, macchina fissa che comprime o forma tramite pressione",
    "10.10": "tornio, fresatrice, mola, levigatrice, trapano a colonna, macchina fissa di lavorazione",
    "10.11": "sega fissa, sega circolare, sega a nastro, macchina fissa per segare",
    "10.12": "cesoia, taglierina, pressa da taglio, macchina fissa per tagliare o fendere",
    "11.03": "gru, carroponte, impianto di sollevamento a carico sospeso",
    "11.04": "carrello, muletto, transpallet, carriola, dispositivo mobile di movimentazione",
    "11.05": "gancio, braca, fune, catena, apparecchio di sollevamento o stivaggio",
    "11.08": "scaffale, scaffalatura, pallet, ripiano di stoccaggio",
    "11.09": "cassa, bottiglia, contenitore, recipiente, imballaggio mobile",
    "12.01": "autocarro, camion, autobus, veicolo industriale",
    "12.02": "automobile, furgone, veicolo leggero per merci o passeggeri",
    "12.03": "bicicletta, motocicletta, scooter, veicolo a due o tre ruote",
    "14.01": "mattone, tegola, putrella, prefabbricato, cassaforma, materiale da costruzione",
    "14.04": "vite, chiodo, bullone, dado, elemento di assemblaggio",
    "14.05": "scheggia, frammento, polvere, particella, pezzo rotto, detrito",
    "14.10": "carico trasportato da muletto, carrello o dispositivo meccanizzato",
    "14.11": "carico sospeso, materiale appeso a gru, braca o apparecchio di sollevamento",
    "14.12": "carico movimentato a mano, scatola, pacco, oggetto sollevato o spostato manualmente",
    "15.01": "acido, soda, sostanza corrosiva o caustica",
    "15.03": "benzina, solvente, liquido infiammabile, sostanza combustibile",
    "15.05": "gas, vapore, atmosfera asfissiante o inerte",
    "15.08": "acqua, liquido non pericoloso, sostanza inerte senza pericolo specifico",
    "18.01": "albero, pianta, coltura, ramo, vegetazione",
    "18.02": "animale domestico o da allevamento",
    "18.03": "animale selvatico, insetto, serpente",
    "18.06": "persona, collega, lavoratore, paziente, cliente, utente, essere umano",
    "18.99": "organismo vivente diverso, altro essere vivente non specificato",
    "20.02": "pioggia, neve, ghiaccio, fango, acqua, vento, grandine, elemento atmosferico o naturale",
}


def material_prototype_texts(entry: TaxonomyEntry, target: TargetField, text_format: str) -> List[tuple[str, str]]:
    normalized_text_format = normalize_text_format(text_format)
    base = candidate_text(entry, target, normalized_text_format)
    if normalized_text_format != TEXT_FORMAT_V7 or target.taxonomy != "agente_materiale":
        return [("base", base)]

    macro = macro_code(entry.code)
    hints = MATERIAL_MACRO_HINTS.get(macro, "")
    code_hint = MATERIAL_CODE_HINTS.get(entry.code, "")
    prototypes = [("base", base)]
    if hints:
        prototypes.append(
            (
                "macro",
                " ".join(
                    [
                        f"Famiglia agente materiale {entry.output_code}: {entry.label}.",
                        f"Indizi lessicali e oggetti tipici: {hints}.",
                        f"Variabile ESAW: {target.title}; ruolo: {target.role}.",
                    ]
                ),
            )
        )
    if code_hint:
        prototypes.append(
            (
                "specific",
                " ".join(
                    [
                        f"Classe specifica agente materiale {entry.output_code}: {entry.label}.",
                        f"Esempi e sinonimi: {code_hint}.",
                        "Preferire questa classe quando questi elementi compaiono nella descrizione dell'infortunio.",
                    ]
                ),
            )
        )
    if entry.parent_label:
        prototypes.append(
            (
                "contrastive",
                " ".join(
                    [
                        f"Distinzione tassonomica: {entry.label}.",
                        f"Appartiene alla famiglia {entry.parent_label}.",
                        f"Non confondere con altre famiglie di agenti materiali se il testo indica: {hints or entry.parent_label}.",
                    ]
                ),
            )
        )

    seen: set[str] = set()
    unique: List[tuple[str, str]] = []
    for prototype_id, text in prototypes:
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append((prototype_id, normalized))
    return unique


def build_candidates(
    explicit_taxonomy: Mapping[str, Sequence[TaxonomyEntry]],
    text_format: str = TEXT_FORMAT_LEGACY,
) -> Dict[str, List[Candidate]]:
    candidates: Dict[str, List[Candidate]] = {}
    normalized_text_format = normalize_text_format(text_format)
    for target in TARGET_FIELDS:
        entries = explicit_taxonomy[target.taxonomy]
        field_candidates: List[Candidate] = []
        for entry in entries:
            for prototype_id, text in material_prototype_texts(entry, target, normalized_text_format):
                field_candidates.append(
                    Candidate(
                        field=target.field,
                        output_field=target.output_field,
                        code=entry.output_code,
                        taxonomy_code=entry.code,
                        label=entry.label,
                        text=text,
                        parent_code=entry.parent_code,
                        parent_label=entry.parent_label,
                        ancestors=entry.ancestors,
                        prototype_id=prototype_id,
                    )
                )
        candidates[target.field] = field_candidates
    return candidates


def label_to_taxonomy_code(target: TargetField, label: Any) -> str:
    raw = normalize_value(label)
    if target.taxonomy == "agente_materiale":
        return material_taxonomy_code(raw)
    return raw


def label_to_output_code(target: TargetField, label: Any) -> str:
    taxonomy_code = label_to_taxonomy_code(target, label)
    if target.taxonomy == "agente_materiale":
        return material_output_code(taxonomy_code)
    return taxonomy_code


def candidate_lookup(candidates: Mapping[str, Sequence[Candidate]]) -> Dict[str, Dict[str, Candidate]]:
    lookup: Dict[str, Dict[str, Candidate]] = {}
    for field, field_candidates in candidates.items():
        lookup[field] = {}
        for candidate in field_candidates:
            lookup[field].setdefault(candidate.taxonomy_code, candidate)
    return lookup


def code_parts(code: str) -> List[int]:
    parts: List[int] = []
    for chunk in normalize_value(code).replace(".", " ").split():
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return parts


def code_distance(left: str, right: str) -> int:
    left_parts = code_parts(left)
    right_parts = code_parts(right)
    size = max(len(left_parts), len(right_parts))
    left_parts.extend([0] * (size - len(left_parts)))
    right_parts.extend([0] * (size - len(right_parts)))
    return sum(abs(a - b) for a, b in zip(left_parts, right_parts))


def macro_code(code: str) -> str:
    normalized = normalize_value(code)
    if "." in normalized:
        return normalized.split(".", 1)[0]
    return normalized[:1]


def nearest_candidates(
    target_code: str,
    candidates: Sequence[Candidate],
    top_n: int,
) -> List[Candidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            code_distance(target_code, candidate.taxonomy_code),
            candidate.taxonomy_code,
        ),
    )
    return ordered[: max(1, min(top_n, len(ordered)))]


def sample_negative(
    rng: random.Random,
    target_code: str,
    field_candidates: Sequence[Candidate],
    prefer_same_parent: bool = True,
    strategy: str = "parent",
    hard_top_n: int = 8,
) -> Candidate:
    positives = [c for c in field_candidates if c.taxonomy_code == target_code]
    positive_parent = positives[0].parent_code if positives else None
    positive_macro = macro_code(target_code)
    pool = [c for c in field_candidates if c.taxonomy_code != target_code]

    if strategy == "random":
        return rng.choice(pool)

    if strategy == "hard":
        hard_pool: List[Candidate] = []
        if positive_parent:
            hard_pool = [c for c in pool if c.parent_code == positive_parent]
        if not hard_pool:
            hard_pool = [c for c in pool if macro_code(c.taxonomy_code) == positive_macro]
        if hard_pool:
            return rng.choice(nearest_candidates(target_code, hard_pool, hard_top_n))
        return rng.choice(nearest_candidates(target_code, pool, hard_top_n))

    if prefer_same_parent and positive_parent:
        hard_pool = [c for c in pool if c.parent_code == positive_parent]
        if hard_pool:
            pool = hard_pool
    return rng.choice(pool)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_labeled_records(records: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for record in records:
        if any(normalize_value(record_value(record, target.field)) for target in TARGET_FIELDS):
            yield record
