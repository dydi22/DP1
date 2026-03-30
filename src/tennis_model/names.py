from __future__ import annotations

import re
import unicodedata


def normalize_player_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", ascii_name)
    collapsed = re.sub(r"\s+", " ", cleaned).strip().lower()
    return collapsed
