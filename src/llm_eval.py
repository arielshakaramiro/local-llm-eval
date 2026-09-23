"""Helpers for benchmarking local GGUF models with llama-cpp-python.

Covers model download/loading, chat calls with timing, regex-based answer
checking, confidence intervals and paired tests for small samples, and
schema-constrained JSON extraction.

Heavy dependencies (llama_cpp, huggingface_hub) are imported lazily so the
pure-Python parts can be unit-tested without them.
"""
from __future__ import annotations

import gc
import json
import math
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

DEFAULT_SYSTEM = "Kamu asisten yang menjawab singkat, jelas, dan dalam bahasa Indonesia."
LEVEL_ORDER = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Environment, download, loading
# ---------------------------------------------------------------------------
def detect_gpu() -> bool:
    """True if an NVIDIA GPU is visible to the system."""
    return shutil.which("nvidia-smi") is not None


def detect_apple_silicon() -> bool:
    """True on Apple Silicon Macs, where llama.cpp can offload to Metal."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def default_gpu_layers() -> int:
    """-1 (offload every layer) when an NVIDIA GPU or Apple Silicon is present, else 0."""
    return -1 if (detect_gpu() or detect_apple_silicon()) else 0


def describe_environment() -> dict:
    """Runtime details worth saving next to benchmark numbers."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": None,
        "llama_cpp_python": None,
        "gpu_offload_supported": None,  # False => this llama-cpp-python build is CPU-only
        "n_gpu_layers": default_gpu_layers(),
    }
    if detect_gpu():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            info["gpu"] = out or None
        except (OSError, subprocess.SubprocessError):
            pass
    elif detect_apple_silicon():
        info["gpu"] = "Apple Silicon (Metal)"

    try:
        import llama_cpp

        info["llama_cpp_python"] = getattr(llama_cpp, "__version__", None)
        supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        info["gpu_offload_supported"] = bool(supports()) if supports else None
    except Exception:  # not installed, or the native library failed to load
        pass
    return info


def download_gguf(repo_id: str, filename: str, local_dir: str | Path = "models") -> str:
    """Download a GGUF file (cached) and return its local path."""
    from huggingface_hub import hf_hub_download

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)


def load_llm(path: str | Path, n_ctx: int = 4096, n_gpu_layers: int | None = None):
    """Load a GGUF model.

    n_ctx defaults to 4096 because llama-cpp-python's own default (512) silently
    truncates longer answers. n_gpu_layers=-1 offloads every layer to the GPU, but
    only takes effect if llama-cpp-python was built with CUDA or Metal support.
    """
    from llama_cpp import Llama

    if n_gpu_layers is None:
        n_gpu_layers = default_gpu_layers()
    return Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chat + answer checking
# ---------------------------------------------------------------------------
def chat(llm, user: str, system: str | None = None, max_tokens: int = 256,
         temperature: float = 0.0, **kwargs) -> dict:
    """Send one message through the model's chat template and time it."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=temperature, **kwargs
    )
    dt = time.perf_counter() - t0

    n_tokens = out["usage"]["completion_tokens"]
    return {
        "text": out["choices"][0]["message"]["content"].strip(),
        "seconds": dt,
        "tokens": n_tokens,
        "tok_per_s": n_tokens / dt if dt > 0 else 0.0,
        "finish": out["choices"][0]["finish_reason"],  # "length" => cut off
    }


def is_correct(answer: str, must, forbid=()) -> bool:
    """Regex-based grading.

    `must` is a list of groups; every group needs at least one matching pattern.
    `forbid` is a list of patterns that must not appear. Matching is
    case-insensitive.
    """
    text = answer.strip().lower()
    for group in must:
        if not any(re.search(p, text) for p in group):
            return False
    return not any(re.search(p, text) for p in forbid)


# ---------------------------------------------------------------------------
# Statistics for small samples
# ---------------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar (sign) test on discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(models: dict, questions: list, system: str = DEFAULT_SYSTEM,
                  n_ctx: int = 4096, models_dir: str | Path = "models",
                  verbose: bool = True) -> pd.DataFrame:
    """Run every question against every model. Models load once each.

    `models` maps a display name to (hf_repo_id, gguf_filename).
    Greedy decoding (temperature 0) is used so runs are repeatable.
    """
    records = []
    for name, (repo, filename) in models.items():
        if verbose:
            print(f"\n=== {name} ===")
        llm = load_llm(download_gguf(repo, filename, models_dir), n_ctx=n_ctx)
        chat(llm, "Halo", max_tokens=8)  # warm-up, not recorded

        for q in questions:
            r = chat(llm, q["prompt"], system=system,
                     max_tokens=q["max_tokens"], temperature=0.0)
            ok = is_correct(r["text"], q["must"], q.get("forbid", ()))
            records.append({
                "model": name, "id": q["id"], "level": q["level"], "prompt": q["prompt"],
                "correct": ok, "seconds": r["seconds"], "tokens": r["tokens"],
                "tok_per_s": r["tok_per_s"], "truncated": r["finish"] == "length",
                "answer": r["text"],
            })
            if verbose:
                mark = "ok " if ok else "MISS"
                print(f"  [{q['id']}] {mark} {r['seconds']:5.1f}s  {q['prompt'][:52]}")

        del llm
        gc.collect()  # free RAM/VRAM before loading the next model

    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model accuracy with 95% Wilson interval, latency and throughput."""
    rows = []
    for model, g in results.groupby("model", sort=False):
        n, k = len(g), int(g["correct"].sum())
        lo, hi = wilson_interval(k, n)
        rows.append({
            "model": model, "n": n, "correct": k,
            "accuracy_pct": 100 * k / n,
            "ci95_low_pct": 100 * lo, "ci95_high_pct": 100 * hi,
            "mean_latency_s": g["seconds"].mean(),
            "mean_tokens_per_s": g["tok_per_s"].mean(),
            "truncated": int(g["truncated"].sum()),
        })
    return pd.DataFrame(rows).set_index("model").round(1)


def accuracy_by_level(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy (%) per difficulty level and model."""
    table = results.pivot_table(index="level", columns="model", values="correct", aggfunc="mean")
    order = [lvl for lvl in LEVEL_ORDER if lvl in table.index]
    return (table.reindex(order) * 100).round(0)


def paired_comparison(results: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Compare two models on the same questions (discordant pairs + exact test)."""
    wide = results.pivot_table(index="id", columns="model", values="correct", aggfunc="first")
    a, b = wide[model_a].astype(bool), wide[model_b].astype(bool)
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    return {
        "only_a_correct": only_a, "only_b_correct": only_b,
        "both_correct": int((a & b).sum()), "both_wrong": int((~a & ~b).sum()),
        "p_value": mcnemar_exact(only_a, only_b),
    }


# ---------------------------------------------------------------------------
# Structured extraction (OCR text -> JSON)
# ---------------------------------------------------------------------------
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part_no": {"type": "string"},
                    "description": {"type": "string"},
                    "material": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "supplier": {"type": "string"},
                    "dimensions": {"type": "string"},
                },
                "required": ["part_no", "description", "quantity"],
            },
        },
        "anomalies": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["components", "anomalies", "summary"],
}

EXTRACTION_SYSTEM = (
    "You are a manufacturing data assistant. Read OCR text taken from a technical drawing "
    "or a Bill of Materials (BOM) and return ONLY a JSON object shaped like this: "
    '{"components": [{"part_no": str, "description": str, "material": str, '
    '"quantity": int, "supplier": str, "dimensions": str}], '
    '"anomalies": [str], "summary": str}. '
    "Copy part numbers exactly as written. In \"anomalies\" list data-quality problems: "
    "empty fields, impossible values (negative or zero quantities or dimensions), or "
    "characters that look like OCR errors. Use [] when there are none."
)


def parse_json_loose(text: str):
    """Extract the outermost {...} from text and parse it; None on failure."""
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def check_schema(obj) -> bool:
    """Strict structural check that mirrors EXTRACTION_SCHEMA's key requirements."""
    if not isinstance(obj, dict):
        return False
    comps = obj.get("components")
    if not isinstance(comps, list):
        return False
    if not isinstance(obj.get("anomalies"), list) or not isinstance(obj.get("summary"), str):
        return False
    for c in comps:
        if not isinstance(c, dict):
            return False
        qty = c.get("quantity")
        if "part_no" not in c or isinstance(qty, bool) or not isinstance(qty, int):
            return False
    return True


def _to_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def score_extraction(pred, truth: dict) -> dict:
    """Score one parsed prediction against ground truth.

    truth = {"part_no": str, "quantity": int, "has_anomaly": bool}
    Unparsed outputs (pred=None) score False everywhere.
    """
    out = {"parsed": pred is not None, "schema_ok": False,
           "part_no_ok": False, "quantity_ok": False, "anomaly_ok": False}
    if not isinstance(pred, dict):
        out["parsed"] = False
        return out

    out["schema_ok"] = check_schema(pred)
    comps = pred.get("components") if isinstance(pred.get("components"), list) else []
    match = next(
        (c for c in comps
         if isinstance(c, dict) and str(c.get("part_no", "")).strip().upper() == truth["part_no"].upper()),
        None,
    )
    out["part_no_ok"] = match is not None
    if match is not None:
        out["quantity_ok"] = _to_int(match.get("quantity")) == truth["quantity"]

    anomalies = pred.get("anomalies")
    if isinstance(anomalies, list):
        out["anomaly_ok"] = (len(anomalies) > 0) == truth["has_anomaly"]
    return out


def extract(llm, ocr_text: str, constrained: bool):
    """Run extraction with or without schema-constrained decoding."""
    kwargs = {}
    if constrained:
        kwargs["response_format"] = {"type": "json_object", "schema": EXTRACTION_SCHEMA}
    r = chat(llm, f"OCR text:\n{ocr_text.strip()}\n\nReturn the JSON object.",
             system=EXTRACTION_SYSTEM, max_tokens=600, temperature=0.0, **kwargs)
    return parse_json_loose(r["text"]), r


def evaluate_extraction(models: dict, samples: list, n_ctx: int = 4096,
                        models_dir: str | Path = "models", verbose: bool = True) -> pd.DataFrame:
    """Compare prompt-only vs schema-constrained extraction on labelled samples."""
    records = []
    for name, (repo, filename) in models.items():
        if verbose:
            print(f"\n=== {name} ===")
        llm = load_llm(download_gguf(repo, filename, models_dir), n_ctx=n_ctx)
        chat(llm, "Hello", max_tokens=8)  # warm-up

        for constrained in (False, True):
            mode = "schema_constrained" if constrained else "prompt_only"
            for s in samples:
                pred, r = extract(llm, s["ocr_text"], constrained)
                score = score_extraction(pred, s["truth"])
                records.append({
                    "model": name, "mode": mode, "sample_id": s["id"],
                    "seconds": r["seconds"], "truncated": r["finish"] == "length",
                    **score, "raw_output": r["text"],
                })
            if verbose:
                print(f"  {mode}: {len(samples)} samples done")

        del llm
        gc.collect()

    return pd.DataFrame(records)


def summarize_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Rates (%) per model and decoding mode."""
    metrics = ["parsed", "schema_ok", "part_no_ok", "quantity_ok", "anomaly_ok"]
    grouped = df.groupby(["model", "mode"], sort=False)
    table = grouped[metrics].mean() * 100
    table["mean_latency_s"] = grouped["seconds"].mean()
    table["n"] = grouped.size()
    return table.round(1)
