from __future__ import annotations

import re
import unicodedata


def normalize_mention(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.replace("đ", "d")
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def tokenize(text: str) -> list[str]:
    normalized = normalize_mention(text)
    return normalized.split() if normalized else []

