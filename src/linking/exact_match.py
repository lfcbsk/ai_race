from __future__ import annotations

from collections import defaultdict

from .schemas import OntologyEntry
from .text import normalize_mention


class ExactAliasIndex:
    def __init__(self, entries: list[OntologyEntry]) -> None:
        self._index: dict[str, list[OntologyEntry]] = defaultdict(list)
        for entry in entries:
            for alias in {entry.name, *entry.aliases}:
                key = normalize_mention(alias)
                if key and all(existing.code != entry.code for existing in self._index[key]):
                    self._index[key].append(entry)

    def search(self, mention: str) -> list[OntologyEntry]:
        return list(self._index.get(normalize_mention(mention), []))

