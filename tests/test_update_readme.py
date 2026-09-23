import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import update_readme as ur  # noqa: E402


def _readme(blocks=ur.ALL_BLOCKS):
    body = "\n".join(f"### {b}\n<!-- {b}:START -->\nold\n<!-- {b}:END -->\n" for b in blocks)
    return "# Title\n\nhand-written text\n\n" + body + "\nTODO write takeaways\n"


def _write_results(d: Path):
    pd.DataFrame({"model": ["A", "B"], "accuracy_pct": [50.0, 70.0]}).to_csv(d / "benchmark_summary.csv", index=False)
    pd.DataFrame({"level": ["easy", "hard"], "A": [90.0, 10.0]}).to_csv(d / "benchmark_by_level.csv", index=False)
    pd.DataFrame({"model": ["A"], "mode": ["prompt_only"], "parsed": [87.5]}).to_csv(d / "extraction_summary.csv", index=False)
    (d / "environment.json").write_text(json.dumps({
        "python": "3.12.0", "platform": "Linux", "gpu": "Tesla T4, 15360 MiB",
        "llama_cpp_python": "0.3.35", "gpu_offload_supported": True, "n_gpu_layers": -1,
    }), encoding="utf-8")


def test_inject_replaces_only_the_marked_block():
    text = "before\n<!-- X:START -->\nold\n<!-- X:END -->\nafter"
    out = ur.inject(text, "X", "new")
    assert out == "before\n<!-- X:START -->\nnew\n<!-- X:END -->\nafter"


def test_inject_missing_marker_raises():
    with pytest.raises(ValueError):
        ur.inject("no markers here", "X", "new")


def test_fill_end_to_end_and_idempotent(tmp_path):
    _write_results(tmp_path)
    once = ur.fill(_readme(), tmp_path)
    assert "| A " in once and "70" in once           # benchmark table
    assert "Tesla T4" in once and "**GPU offload active:** yes" in once
    assert "hand-written text" in once and "TODO write takeaways" in once   # untouched
    assert ur.fill(once, tmp_path) == once                                    # idempotent


def test_fill_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the notebook first"):
        ur.fill(_readme(), tmp_path)


def test_environment_block_flags_cpu_only_build(tmp_path):
    p = tmp_path / "environment.json"
    p.write_text(json.dumps({"gpu": None, "gpu_offload_supported": False, "python": "3.12"}), encoding="utf-8")
    block = ur.environment_block(p)
    assert "none (CPU only)" in block and "no (CPU-only build)" in block


def test_repository_readmes_have_every_marker_pair_and_the_figure():
    for name in ("README.md", "README.id.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for block in ur.ALL_BLOCKS:
            assert f"<!-- {block}:START -->" in text and f"<!-- {block}:END -->" in text, name
        assert "results/benchmark_accuracy.png" in text, name


def test_readmes_cross_link_each_other():
    en = (ROOT / "README.md").read_text(encoding="utf-8")
    idn = (ROOT / "README.id.md").read_text(encoding="utf-8")
    assert "README.id.md" in en
    assert "README.md" in idn
