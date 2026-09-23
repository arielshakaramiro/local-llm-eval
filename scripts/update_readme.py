"""Fill the generated tables in README.md and README.id.md from the files the notebook writes to results/.

Usage, from the repository root:

    python scripts/update_readme.py

Only the blocks between <!-- NAME:START --> and <!-- NAME:END --> are replaced, in both READMEs.
Hand-written parts are marked TODO in the README; they are left alone and reported.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CSV_BLOCKS = {
    "BENCHMARK_SUMMARY": "benchmark_summary.csv",
    "BENCHMARK_BY_LEVEL": "benchmark_by_level.csv",
    "EXTRACTION_SUMMARY": "extraction_summary.csv",
}
ENV_BLOCK = "ENVIRONMENT"
ENV_FILE = "environment.json"
ALL_BLOCKS = [*CSV_BLOCKS, ENV_BLOCK]


def inject(text: str, name: str, content: str) -> str:
    """Replace the content between the START/END markers of `name`."""
    pattern = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"marker block {name} not found in README")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content.strip()}\n{m.group(3)}", text, count=1)


def csv_table(path: Path) -> str:
    return pd.read_csv(path).to_markdown(index=False)


def environment_block(path: Path) -> str:
    info = json.loads(path.read_text(encoding="utf-8"))
    offload = {True: "yes", False: "no (CPU-only build)"}.get(info.get("gpu_offload_supported"), "unknown")
    lines = [
        f"- **Accelerator:** {info.get('gpu') or 'none (CPU only)'}",
        f"- **GPU offload active:** {offload}",
        f"- **llama-cpp-python:** {info.get('llama_cpp_python') or 'unknown'}",
        f"- **Python:** {info.get('python')}",
        f"- **Platform:** {info.get('platform')}",
    ]
    return "\n".join(lines)


def missing_files(results_dir: Path) -> list[str]:
    needed = [*CSV_BLOCKS.values(), ENV_FILE]
    return [name for name in needed if not (results_dir / name).exists()]


def fill(readme_text: str, results_dir: Path) -> str:
    missing = missing_files(results_dir)
    if missing:
        raise FileNotFoundError(f"missing in {results_dir}: {', '.join(missing)}. Run the notebook first.")
    for block, filename in CSV_BLOCKS.items():
        readme_text = inject(readme_text, block, csv_table(results_dir / filename))
    return inject(readme_text, ENV_BLOCK, environment_block(results_dir / ENV_FILE))


def main() -> None:
    results = ROOT / "results"
    readmes = [ROOT / "README.md", ROOT / "README.id.md"]

    try:
        missing = missing_files(results)
        if missing:
            raise FileNotFoundError(f"missing in {results}: {', '.join(missing)}. Run the notebook first.")
        for readme in readmes:
            if not readme.exists():
                raise FileNotFoundError(f"{readme.name} not found in {ROOT}")
            readme.write_text(fill(readme.read_text(encoding="utf-8"), results), encoding="utf-8")
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"error: {err}")

    print(f"Updated: {', '.join(r.name for r in readmes)}")
    for readme in readmes:
        todo_lines = [i for i, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1)
                      if "TODO" in line]
        if todo_lines:
            print(f"{readme.name}: still to write by hand: {len(todo_lines)} TODO line(s), at line(s) {todo_lines}")


if __name__ == "__main__":
    main()
