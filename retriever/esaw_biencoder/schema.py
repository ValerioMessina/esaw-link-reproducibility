from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


INPUT_FIELDS: List[str] = [
    "luogo_infortunio",
    "lavorazione",
    "dettaglio_lavorazione",
    "imprevisto",
    "conseguenza",
    "des_infortunio",
]


@dataclass(frozen=True)
class TargetField:
    field: str
    taxonomy: str
    title: str
    role: str
    output_field: str


TARGET_FIELDS: List[TargetField] = [
    TargetField(
        field="cod_loc",
        taxonomy="tipo_luogo",
        title="Tipo di luogo",
        role="luogo in cui e' avvenuto l'infortunio",
        output_field="COD_LOC",
    ),
    TargetField(
        field="cod_lav",
        taxonomy="tipo_lavoro",
        title="Tipo di lavoro",
        role="lavorazione o contesto lavorativo generale",
        output_field="COD_LAV",
    ),
    TargetField(
        field="cod_atv",
        taxonomy="attivita_fisica_specifica",
        title="Attivita' fisica specifica",
        role="azione concreta svolta dall'infortunato subito prima dell'evento",
        output_field="COD_ATV",
    ),
    TargetField(
        field="cod_mat_atv",
        taxonomy="agente_materiale",
        title="Agente materiale dell'attivita'",
        role="oggetto, superficie, mezzo o ambiente collegato all'attivita' fisica specifica",
        output_field="COD_MAT_ATV",
    ),
    TargetField(
        field="cod_dev",
        taxonomy="deviazione",
        title="Deviazione",
        role="evento anomalo o imprevisto che ha causato la lesione",
        output_field="COD_DEV",
    ),
    TargetField(
        field="cod_mat_dev",
        taxonomy="agente_materiale",
        title="Agente materiale della deviazione",
        role="oggetto, superficie, mezzo o ambiente coinvolto nell'evento anomalo",
        output_field="COD_MAT_DEV",
    ),
    TargetField(
        field="cod_con",
        taxonomy="contatto_modalita_lesione",
        title="Contatto - modalita' della lesione",
        role="modalita' con cui l'infortunato e' entrato in contatto con la causa della lesione",
        output_field="COD_CON",
    ),
    TargetField(
        field="cod_mat_con",
        taxonomy="agente_materiale",
        title="Agente materiale del contatto",
        role="oggetto, superficie, mezzo o ambiente che ha prodotto il contatto lesivo",
        output_field="COD_MAT_CON",
    ),
]

TARGET_BY_FIELD: Dict[str, TargetField] = {target.field: target for target in TARGET_FIELDS}
