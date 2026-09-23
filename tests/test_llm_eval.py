import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import llm_eval as le  # noqa: E402

QUESTIONS = le.load_json(ROOT / "data" / "questions_id.json")
SAMPLES = le.load_json(ROOT / "data" / "ocr_samples.json")


class FakeLLM:
    """Minimal stand-in for llama_cpp.Llama."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def create_chat_completion(self, messages, max_tokens, temperature, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "usage": {"completion_tokens": 7},
            "choices": [{"message": {"content": self.reply}, "finish_reason": "stop"}],
        }


# ── dataset integrity ──────────────────────────────────────────────
def test_question_set_shape():
    assert len(QUESTIONS) == 30
    assert len({q["id"] for q in QUESTIONS}) == 30
    levels = {lvl: sum(q["level"] == lvl for q in QUESTIONS) for lvl in le.LEVEL_ORDER}
    assert levels == {"easy": 10, "medium": 10, "hard": 10}


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_each_question_accepts_correct_and_rejects_wrong(q):
    forbid = q.get("forbid", [])
    assert le.is_correct(q["example_correct"], q["must"], forbid)
    assert not le.is_correct(q["example_wrong"], q["must"], forbid)


def test_ocr_samples_are_well_formed():
    assert len({s["id"] for s in SAMPLES}) == len(SAMPLES)
    for s in SAMPLES:
        assert set(s["truth"]) == {"part_no", "quantity", "has_anomaly"}
        assert s["truth"]["part_no"] in s["ocr_text"]


# ── grading ────────────────────────────────────────────────────────
def test_is_correct_accepts_full_sentence_answers():
    # exact-match grading would reject this; pattern grading must not
    assert le.is_correct("5 ditambah 3 adalah 8.", [[r"\b8\b"]])


def test_is_correct_requires_every_group():
    must = [[r"gaya"], [r"massa"]]
    assert le.is_correct("Gaya sama dengan massa kali percepatan", must)
    assert not le.is_correct("Gaya membuat benda bergerak", must)


def test_is_correct_forbid_blocks_answer():
    assert not le.is_correct("Ali paling pendek", [[r"ali"]], forbid=[r"paling pendek"])


# ── statistics ─────────────────────────────────────────────────────
def test_wilson_interval_known_values():
    lo, hi = le.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)

    lo, hi = le.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.2775, abs=1e-3)

    assert le.wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_known_values():
    assert le.mcnemar_exact(0, 0) == 1.0
    assert le.mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert le.mcnemar_exact(8, 2) == pytest.approx(0.109375)
    assert le.mcnemar_exact(3, 3) == 1.0


def _fake_results():
    rows = []
    # model A gets q1,q2 right; model B gets q2,q3 right
    correct = {"A": {"q1": True, "q2": True, "q3": False}, "B": {"q1": False, "q2": True, "q3": True}}
    for model, per_q in correct.items():
        for qid, ok in per_q.items():
            rows.append({"model": model, "id": qid, "level": "easy", "correct": ok,
                         "seconds": 2.0, "tok_per_s": 10.0, "truncated": False})
    return pd.DataFrame(rows)


def test_summarize_and_paired_comparison():
    res = _fake_results()
    summary = le.summarize(res)
    assert summary.loc["A", "correct"] == 2 and summary.loc["A", "n"] == 3
    assert summary.loc["A", "ci95_low_pct"] < summary.loc["A", "accuracy_pct"] < summary.loc["A", "ci95_high_pct"]

    pc = le.paired_comparison(res, "A", "B")
    assert (pc["only_a_correct"], pc["only_b_correct"], pc["both_correct"], pc["both_wrong"]) == (1, 1, 1, 0)
    assert pc["p_value"] == 1.0


# ── chat wrapper ───────────────────────────────────────────────────
def test_chat_returns_timing_and_finish_reason():
    llm = FakeLLM("  halo  ")
    r = le.chat(llm, "hai", system="sys", max_tokens=8)
    assert r["text"] == "halo" and r["tokens"] == 7 and r["finish"] == "stop"
    assert llm.calls[0]["messages"][0] == {"role": "system", "content": "sys"}


# ── structured extraction ──────────────────────────────────────────
GOOD = ('{"components": [{"part_no": "ABC-123", "description": "Gearbox", "quantity": 5}], '
        '"anomalies": [], "summary": "ok"}')
TRUTH_CLEAN = {"part_no": "ABC-123", "quantity": 5, "has_anomaly": False}


def test_parse_json_loose_handles_wrapped_and_broken_output():
    assert le.parse_json_loose("Here you go:\n" + GOOD + "\nDone.")["summary"] == "ok"
    assert le.parse_json_loose("no json here") is None
    assert le.parse_json_loose('{"components": [') is None


def test_check_schema_is_strict_about_types():
    assert le.check_schema(le.parse_json_loose(GOOD))
    string_qty = GOOD.replace('"quantity": 5', '"quantity": "5"')
    assert not le.check_schema(le.parse_json_loose(string_qty))
    assert not le.check_schema({"components": []})


def test_score_extraction_variants():
    ok = le.score_extraction(le.parse_json_loose(GOOD), TRUTH_CLEAN)
    assert all(ok.values())

    # quantity given as a string: schema violation, but the value is still right
    lenient = le.score_extraction(le.parse_json_loose(GOOD.replace('"quantity": 5', '"quantity": "5"')), TRUTH_CLEAN)
    assert lenient["quantity_ok"] and not lenient["schema_ok"]

    # claims anomalies on a clean record
    flagged = GOOD.replace('"anomalies": []', '"anomalies": ["odd"]')
    assert not le.score_extraction(le.parse_json_loose(flagged), TRUTH_CLEAN)["anomaly_ok"]

    # unparsed output scores False everywhere
    assert not any(le.score_extraction(None, TRUTH_CLEAN).values())


def test_extract_passes_schema_only_when_constrained():
    llm = FakeLLM(GOOD)
    pred, _ = le.extract(llm, "Part No: ABC-123", constrained=True)
    assert "response_format" in llm.calls[-1]["kwargs"]
    assert llm.calls[-1]["kwargs"]["response_format"]["schema"] is le.EXTRACTION_SCHEMA
    assert pred["components"][0]["part_no"] == "ABC-123"

    le.extract(llm, "Part No: ABC-123", constrained=False)
    assert "response_format" not in llm.calls[-1]["kwargs"]


def test_summarize_extraction_rates():
    df = pd.DataFrame([
        {"model": "M", "mode": "prompt_only", "seconds": 1.0, "parsed": True, "schema_ok": False,
         "part_no_ok": True, "quantity_ok": True, "anomaly_ok": False},
        {"model": "M", "mode": "prompt_only", "seconds": 3.0, "parsed": False, "schema_ok": False,
         "part_no_ok": False, "quantity_ok": False, "anomaly_ok": False},
    ])
    table = le.summarize_extraction(df)
    row = table.loc[("M", "prompt_only")]
    assert row["parsed"] == 50.0 and row["mean_latency_s"] == 2.0 and row["n"] == 2


# ── environment detection ──────────────────────────────────────────
@pytest.mark.parametrize("nvidia, apple, expected", [
    (False, False, 0), (True, False, -1), (False, True, -1), (True, True, -1),
])
def test_default_gpu_layers(monkeypatch, nvidia, apple, expected):
    monkeypatch.setattr(le, "detect_gpu", lambda: nvidia)
    monkeypatch.setattr(le, "detect_apple_silicon", lambda: apple)
    assert le.default_gpu_layers() == expected


def test_describe_environment_without_gpu(monkeypatch):
    monkeypatch.setattr(le, "detect_gpu", lambda: False)
    monkeypatch.setattr(le, "detect_apple_silicon", lambda: False)
    info = le.describe_environment()
    assert {"python", "platform", "gpu", "llama_cpp_python", "gpu_offload_supported", "n_gpu_layers"} <= set(info)
    assert info["gpu"] is None and info["n_gpu_layers"] == 0
