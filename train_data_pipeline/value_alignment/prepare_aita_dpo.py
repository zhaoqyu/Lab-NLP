#!/usr/bin/env python3
"""Deprecated guard for the former, incorrect AITA-training entry point."""

def main() -> None:
    raise SystemExit(
        "AITA is evaluation-only in the corrected pipeline. "
        "Use value_alignment/prepare_kvs_dpo.py for training data and "
        "value_alignment/prepare_aita_eval.py for the AITA test set."
    )


if __name__ == "__main__":
    main()
