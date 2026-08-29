"""Reproducibility helpers shared by training and evaluation entry points."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "lm_eval",
)


def tagged_run_dir(base_dir: Path, run_tag: str | None) -> Path:
    """Return the legacy directory or an isolated ``runs/<tag>`` directory."""
    if not run_tag:
        return base_dir
    if not RUN_TAG_PATTERN.fullmatch(run_tag):
        raise ValueError(
            "Run tags must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens."
        )
    return base_dir / "runs" / run_tag


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        result.update({"bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)})
        if resolved.suffix.lower() in {".jsonl", ".csv", ".txt"}:
            with resolved.open("rb") as handle:
                result["line_count"] = sum(1 for _ in handle)
    return result


def git_metadata(repo_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {
        "root": str(repo_root.resolve()),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def installed_package_versions() -> dict[str, str | None]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return _jsonable(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_run_manifest(
    output_dir: Path,
    args: argparse.Namespace | Mapping[str, Any],
    input_files: Mapping[str, Path],
    *,
    metadata: Mapping[str, Any] | None = None,
    status: str = "started",
) -> Path:
    """Write or update a machine-readable experiment manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    now = datetime.now(timezone.utc).isoformat()
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}

    is_new_attempt = status == "started"
    attempt = int(previous.get("attempt", 0)) + 1 if is_new_attempt else int(previous.get("attempt", 1))
    manifest = {
        "schema_version": 1,
        "attempt": attempt,
        "status": status,
        "started_at_utc": now if is_new_attempt else previous.get("started_at_utc", now),
        "updated_at_utc": now,
        "command": [sys.executable, *sys.argv],
        "arguments": _jsonable(args),
        "metadata": _jsonable(metadata or {}),
        "input_files": {
            name: file_fingerprint(Path(path)) for name, path in input_files.items()
        },
        "git": git_metadata(),
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": installed_package_versions(),
        },
        "scheduler": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    if status == "completed":
        manifest["completed_at_utc"] = now

    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path
