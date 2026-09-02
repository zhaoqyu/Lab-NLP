"""Rebuild the clean Colab notebooks committed with the benchmark."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
REPO = "https://github.com/zhaoqyu/Lab-NLP.git"
BRANCH = "mike"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def setup_cells(notebook: str, title: str, objective: str, *, secret: bool = False) -> list[dict]:
    badge = (
        "https://colab.research.google.com/github/zhaoqyu/Lab-NLP/blob/"
        f"{BRANCH}/alignment_benchmark/notebooks/{notebook}"
    )
    cells = [
        markdown(
            f"# {title}\n\n"
            f"[Open in Google Colab]({badge})\n\n"
            f"**Objective.** {objective}\n\n"
            "Use a GPU runtime. Persistent artifacts are written to Google Drive, so interrupted Colab sessions "
            "can resume. Run cells from top to bottom."
        ),
        code(
            "import os\n"
            "import subprocess\n"
            "from pathlib import Path\n\n"
            "from google.colab import drive\n\n"
            "drive.mount('/content/drive')\n"
            "assert subprocess.run(['nvidia-smi'], check=False).returncode == 0, 'Select a GPU runtime first.'\n"
        ),
        code(
            f"REPO_URL = {REPO!r}\n"
            f"BRANCH = {BRANCH!r}\n"
            "REPO_DIR = Path('/content/Lab-NLP')\n\n"
            "if not (REPO_DIR / '.git').exists():\n"
            "    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)\n"
            "else:\n"
            "    subprocess.run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_DIR, check=True)\n"
            "    subprocess.run(['git', 'checkout', BRANCH], cwd=REPO_DIR, check=True)\n"
            "    subprocess.run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO_DIR, check=True)\n\n"
            "os.chdir(REPO_DIR)\n"
            "print('Repository:', REPO_DIR)\n"
        ),
        code(
            "subprocess.run(\n"
            "    ['python', '-m', 'pip', 'install', '-q', '-r', 'alignment_benchmark/requirements-colab.txt'],\n"
            "    check=True,\n"
            ")\n"
            "subprocess.run(\n"
            "    ['python', '-m', 'pip', 'install', '-q', '-e', './alignment_benchmark', '--no-deps'],\n"
            "    check=True,\n"
            ")\n"
        ),
        code(
            "OUTPUT_ROOT = Path('/content/drive/MyDrive/Lab-NLP/valuebench-paper')\n"
            "OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "os.environ['VALUEBENCH_OUTPUT_ROOT'] = str(OUTPUT_ROOT)\n"
            "CONFIG = 'alignment_benchmark/configs/paper.yaml'\n\n"
            "def run(*arguments: str) -> None:\n"
            "    command = ['valuebench', *arguments, '--config', CONFIG]\n"
            "    print('Running:', ' '.join(command))\n"
            "    subprocess.run(command, check=True)\n\n"
            "print('Persistent output:', OUTPUT_ROOT)\n"
        ),
    ]
    if secret:
        cells.append(
            code(
                "from google.colab import userdata\n\n"
                "api_key = userdata.get('OPENROUTER_API_KEY')\n"
                "assert api_key, 'Add OPENROUTER_API_KEY in the Colab Secrets panel.'\n"
                "os.environ['OPENROUTER_API_KEY'] = api_key\n"
                "del api_key\n"
                "print('OpenRouter secret loaded without displaying it.')\n"
            )
        )
    cells.append(code("run('doctor', '--strict')\n"))
    return cells


def notebook(cells: list[dict], name: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_data() -> dict:
    cells = setup_cells(
        "01_build_data.ipynb",
        "01 - Build Audited KVS Method Data",
        "Create one auditable canonical pair per KVS source, then derive fair views for every method.",
        secret=True,
    )
    cells.extend(
        [
            markdown(
                "## 1. Preview the Teacher contract\n\n"
                "This writes one pending prompt locally without making a paid request. Inspect it before the full run."
            ),
            code(
                "run('teacher', '--dry-run', '--limit', '1')\n\n"
                "jobs = OUTPUT_ROOT / 'data' / 'teacher_jobs.jsonl'\n"
                "print(jobs.read_text(encoding='utf-8')[:4000])\n"
            ),
            markdown(
                "## 2. Generate all canonical records\n\n"
                "The run is atomic and resumable. It requests strict JSON and stores no private chain-of-thought. "
                "OpenRouter usage charges apply."
            ),
            code("run('teacher')\n"),
            markdown("## 3. Audit fidelity and prepare the external test set"),
            code("run('audit-data', '--fail-on-quality')\nrun('prepare-aita')\n"),
            markdown(
                "## 4. Score the frozen base model\n\n"
                "These ratings become SFT control labels. Preference margins provide the HyPO mismatch diagnostic."
            ),
            code("run('collect-baselines')\nrun('summarize-mismatch')\n"),
            markdown("## 5. Build and verify all equal-source method views"),
            code("run('build-views')\nrun('validate-data')\nrun('make-plan')\nrun('status')\n"),
            markdown(
                "## Checkpoint\n\n"
                "The canonical data, reports, baselines, method views, and experiment plan now live on Drive. "
                "Continue with notebook 02."
            ),
        ]
    )
    return notebook(cells, "01_build_data.ipynb")


def train_one() -> dict:
    cells = setup_cells(
        "02_train_one_run.ipynb",
        "02 - Train One Registered Alignment Run",
        "Fit or resume one QLoRA control/intervention adapter in a Colab-safe unit of work.",
    )
    cells.extend(
        [
            markdown(
                "## Select one registered run\n\n"
                "Start with controls. A target evaluation requires the control adapter for the same method and seed. "
                "Valid methods are `sft`, `dpo`, `hypo`, `ipo`, `simpo`, and `orpo`."
            ),
            code(
                "METHOD = 'dpo'\n"
                "TARGET = 'control'  # or one of the ten basic values, e.g. Security\n"
                "SEED = 13\n\n"
                "METHOD, TARGET, SEED\n"
            ),
            code(
                "run(\n"
                "    'train',\n"
                "    '--method', METHOD,\n"
                "    '--target', TARGET,\n"
                "    '--seed', str(SEED),\n"
                ")\n"
            ),
            markdown(
                "## Inspect progress\n\n"
                "The trainer resumes the last checkpoint if the runtime stopped. A completed run has a final adapter, "
                "manifest, trainer state, TensorBoard logs, and `DONE` marker."
            ),
            code("run('status')\n"),
            markdown(
                "## Queue-style alternative\n\n"
                "Set `USE_REGISTERED_QUEUE = True` to run exactly one ready evaluation or pending construction job. "
                "Rerun the notebook in later sessions until `status` is complete."
            ),
            code("USE_REGISTERED_QUEUE = False\nif USE_REGISTERED_QUEUE:\n    run('run-next')\n"),
        ]
    )
    return notebook(cells, "02_train_one_run.ipynb")


def steer_evaluate() -> dict:
    cells = setup_cells(
        "03_steer_and_evaluate.ipynb",
        "03 - Activation Steering and Locked Evaluation",
        "Select CAA interventions on KVS validation and score completed methods on locked KVS/AITA data.",
    )
    cells.extend(
        [
            markdown(
                "## 1. Build one activation-steering intervention\n\n"
                "Vectors use KVS train only. The layer and coefficient are selected on KVS validation only."
            ),
            code(
                "TARGET = 'Security'\n"
                "SITE = 'block'  # block or attn\n\n"
                "run('build-steering', '--target', TARGET, '--site', SITE)\n"
            ),
            markdown("## 2. Evaluate the selected steering vector"),
            code(
                "run(\n"
                "    'evaluate',\n"
                "    '--method', f'steering_{SITE}',\n"
                "    '--target', TARGET,\n"
                "    '--seed', '-1',\n"
                ")\n"
            ),
            markdown(
                "## 3. Evaluate one completed adapter intervention\n\n"
                "Both the target and same-method control adapter must already be complete."
            ),
            code(
                "RUN_ADAPTER_EVALUATION = False\n"
                "METHOD = 'dpo'\n"
                "ADAPTER_TARGET = 'Security'\n"
                "SEED = 13\n\n"
                "if RUN_ADAPTER_EVALUATION:\n"
                "    run(\n"
                "        'evaluate',\n"
                "        '--method', METHOD,\n"
                "        '--target', ADAPTER_TARGET,\n"
                "        '--seed', str(SEED),\n"
                "    )\n"
            ),
            markdown(
                "## Queue-style evaluation\n\n"
                "Once adapters or vectors exist, `run-next --phase evaluate` chooses one ready, unfinished evaluation."
            ),
            code(
                "RUN_NEXT_READY_EVALUATION = False\n"
                "if RUN_NEXT_READY_EVALUATION:\n"
                "    run('run-next', '--phase', 'evaluate')\n\n"
                "run('status')\n"
            ),
        ]
    )
    return notebook(cells, "03_steer_and_evaluate.ipynb")


def analyze() -> dict:
    cells = setup_cells(
        "04_analyze_results.ipynb",
        "04 - Aggregate Results and Build Paper Artifacts",
        "Enforce experiment completeness, compute registered statistics, and export final tables and figures.",
    )
    cells.extend(
        [
            markdown(
                "## 1. Completeness gate\n\n"
                "Final aggregation deliberately fails if any registered construction or evaluation is missing."
            ),
            code("run('status')\n"),
            markdown("## 2. Bootstrap registered metrics"),
            code("run('aggregate-results')\n"),
            markdown("## 3. Export paper tables, figures, and checksums"),
            code("run('paper-artifacts')\n"),
            code(
                "import pandas as pd\n"
                "from IPython.display import display\n\n"
                "paper = OUTPUT_ROOT / 'paper'\n"
                "display(pd.read_csv(paper / 'table_main_method_comparison.csv'))\n"
                "display(pd.read_csv(paper / 'table_per_value_results.csv').head(20))\n"
            ),
            code(
                "from IPython.display import Image, display\n\n"
                "for image_path in sorted(paper.glob('figure_*.png')):\n"
                "    print(image_path.name)\n"
                "    display(Image(filename=str(image_path), width=900))\n"
            ),
            markdown("## 4. Create a portable paper-artifact archive"),
            code(
                "import shutil\n\n"
                "archive = shutil.make_archive(str(OUTPUT_ROOT / 'paper_artifacts'), 'zip', root_dir=paper)\n"
                "print('Archive:', archive)\n"
            ),
            markdown(
                "## Interpretation note\n\n"
                "Do not write conclusions from direction alone. Use confidence intervals, FDR-adjusted per-value "
                "tests, sparse-cell flags, drift, seed stability, and efficiency together."
            ),
        ]
    )
    return notebook(cells, "04_analyze_results.ipynb")


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    payloads = {
        "01_build_data.ipynb": build_data(),
        "02_train_one_run.ipynb": train_one(),
        "03_steer_and_evaluate.ipynb": steer_evaluate(),
        "04_analyze_results.ipynb": analyze(),
    }
    for name, payload in payloads.items():
        path = NOTEBOOKS / name
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
