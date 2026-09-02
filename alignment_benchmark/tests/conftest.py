from __future__ import annotations

from pathlib import Path

import pytest

from valuebench.config import load_config


@pytest.fixture
def config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "paper.yaml"


@pytest.fixture
def config(config_path, tmp_path, monkeypatch):
    monkeypatch.setenv("VALUEBENCH_OUTPUT_ROOT", str(tmp_path / "outputs"))
    return load_config(config_path)
