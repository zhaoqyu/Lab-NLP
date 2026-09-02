from __future__ import annotations

from collections import Counter

from valuebench.data import prepare_aita
from valuebench.io import read_json
from valuebench.io import read_jsonl
from valuebench.taxonomy import BASIC_VALUES, FINE_TO_BASIC, to_basic
from valuebench.teacher import raw_kvs_rows


def test_fixed_taxonomy_is_complete():
    assert len(BASIC_VALUES) == 10
    assert len(FINE_TO_BASIC) == 20
    assert set(FINE_TO_BASIC.values()) == set(BASIC_VALUES)
    assert to_basic("Self direction thought") == "Self_direction"
    assert to_basic("Power-resources") == "Power"


def test_real_kvs_splits_and_categories(config):
    rows = raw_kvs_rows(config)
    counts = Counter(row["split"] for row in rows)
    assert counts == {"train": 378, "eval": 108, "test": 108}
    assert len({row["source_id"] for row in rows}) == 594
    assert set(row["fine_value"] for row in rows) == set(FINE_TO_BASIC)


def test_real_aita_is_mapped_without_synthesis(config):
    raw = read_json(config.paths.aita)
    expected = sum(len(examples) for examples in raw.values())
    summary = prepare_aita(config)
    assert expected == 4335
    assert summary["raw_rows"] == expected
    assert summary["rows"] == 4192
    assert summary["exact_duplicates_removed"] == 143
    assert set(summary["basic_value_counts"]) == set(BASIC_VALUES)
    rows = list(read_jsonl(config.paths.output_root / "data" / "aita.jsonl"))
    assert len({row["source_id"] for row in rows}) == summary["rows"]
