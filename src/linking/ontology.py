from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import OntologyEntry, OntologyName


def _read_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Mapping ontology phải là JSON object: {path}")
    return payload


def load_icd_entries(path: str | Path) -> list[OntologyEntry]:
    payload = _read_mapping(path)
    grouped: dict[str, dict[str, Any]] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    for section in ("verified", "review"):
        for record in payload.get(section, []):
            concept = record.get("selected_concept")
            if not isinstance(concept, dict) or not concept.get("code"):
                continue
            code = str(concept["code"])
            grouped.setdefault(code, concept)
            for value in (record.get("vn_mention"), concept.get("name_vi")):
                if isinstance(value, str) and value.strip():
                    aliases[code].add(value.strip())
    return [
        OntologyEntry(
            code=code,
            name=str(concept.get("name_vi") or code),
            aliases=tuple(sorted(aliases[code])),
            description=str(concept.get("group_vi") or ""),
            ontology="ICD10",
            metadata={"group_vi": concept.get("group_vi")},
        )
        for code, concept in sorted(grouped.items())
    ]


def load_rxnorm_entries(path: str | Path) -> list[OntologyEntry]:
    payload = _read_mapping(path)
    grouped: dict[str, dict[str, Any]] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    for section in ("verified", "review"):
        for record in payload.get(section, []):
            concept = record.get("selected_concept")
            if not isinstance(concept, dict) or not concept.get("rxcui"):
                continue
            code = str(concept["rxcui"])
            grouped.setdefault(code, concept)
            parsed = record.get("parsed") or {}
            for value in (
                record.get("original_text"),
                parsed.get("drug_name") if isinstance(parsed, dict) else None,
                concept.get("name"),
            ):
                if isinstance(value, str) and value.strip():
                    aliases[code].add(value.strip())
    return [
        OntologyEntry(
            code=code,
            name=str(concept.get("name") or code),
            aliases=tuple(sorted(aliases[code])),
            description=str(concept.get("tty") or ""),
            ontology="RXNORM",
            metadata={"tty": concept.get("tty")},
        )
        for code, concept in sorted(grouped.items())
    ]


def ensure_single_ontology(entries: list[OntologyEntry]) -> OntologyName:
    ontologies = {entry.ontology for entry in entries}
    if len(ontologies) != 1:
        raise ValueError("Một linker chỉ được chứa đúng một ontology.")
    if not entries:
        raise ValueError("Ontology không có entry.")
    return entries[0].ontology

