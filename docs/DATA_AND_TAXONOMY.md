# External inputs and taxonomy policy

No private datasets, per-record split files, predictions, silver labels,
narratives, record IDs or trained weights are part of this repository.
Do not place them here even temporarily; use an external secure work directory.

The input reader supports a JSON list or the original export wrapper
`{"results": [{"items": [...]}]}`. Narrative fields are `luogo_infortunio`,
`lavorazione`, `dettaglio_lavorazione`, `imprevisto`, `conseguenza`,
`des_infortunio`. Targets are `cod_loc`, `cod_lav`, `cod_atv`, `cod_mat_atv`,
`cod_dev`, `cod_mat_dev`, `cod_con`, `cod_mat_con`. Uppercase aliases are
supported by the historical readers. No example here is an institutional case.

The frozen archive has 50,000 records, with 40,000 train / 5,000 validation /
5,000 test, seed 13. Keep the original private split for historical comparison.
`scripts/make_split.py` implements the original shuffle algorithm; identical
memberships require the identical input record order and identity convention.
Generating a split on another archive does not reproduce the historical split.
The split was not grouped by event or deduplicated by text. An audit found 12
nonempty test records with an exact text match in train; exclusion sensitivity
does not prove absence of near duplicates or shared events.

Provide the original `tassonomia.json` externally. Expected SHA-256:
`f14ed4c351603d6d23e53b310f0b4d57af58737d5cd3b75016454cd0065f0366`.
Six inventories contain 63 / 36 / 40 / 52 / 48 / 193 entries, respectively:
`tipo_luogo`, `tipo_lavoro`, `attivita_fisica_specifica`, `deviazione`,
`contatto_modalita_lesione`, `agente_materiale`.
There are 818 field–code documents because material agents appear in three
roles. HTC adds eight technical field roots, for 826 nodes; roots are not ESAW
codes. The same material code has different candidate text in each role.

The supplied activity-19 description incorrectly mentions group 20 instead
of group 10, duplicating the activity-29 label. Historical results use that
unchanged file. Any corrected taxonomy is a different, explicitly versioned
experiment and requires regenerated candidate embeddings. This release does
not activate a correction. Source: Eurostat ESAW Summary Methodology 2013,
Annex I, p.26:
https://ec.europa.eu/eurostat/documents/3859598/5926181/KS-RA-12-102-EN.PDF/56cd35ba-1e8a-4af3-9f9a-b3c47611ff1c

In the checked archive, material codes `XX.YY.00.00` map reversibly to `XX.YY`.
Reject deeper nonzero components rather than truncating an unfamiliar archive.
Input checks verify this condition; the historical retriever's normalization
alone is not sufficient validation for new datasets.

