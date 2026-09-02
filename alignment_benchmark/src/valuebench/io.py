"""Small deterministic I/O helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def stable_int(text: str) -> int:
    return int(sha256_text(text)[:16], 16)


def seeded_sample(rows: list[dict], limit: int, seed_key: str) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    rng = random.Random(stable_int(seed_key))
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in indices]
