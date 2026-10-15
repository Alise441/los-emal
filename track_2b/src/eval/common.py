"""Shared evaluation helpers: SwissDial loading, metrics, parallel calls, output files.

SwissDial (Dogan-Schönberger et al., 2021) is available from ETH Zurich on request for
research use. It is NOT redistributed in this repository; only aggregate scores are reported.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from losemal.config import PROJECT_DIR

RESULTS = PROJECT_DIR / "results"
WAV_RE = re.compile(r"ch_(?P<d>[a-z]{2})_(?P<id>\d+)\.wav$", re.IGNORECASE)
FILE_PREFIX = {"vs": "ws"}  # SwissDial uses "ws" for Valais in file names


def out_dir(name: str) -> Path:
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_swissdial_text(root: str | Path) -> list[dict]:
    """All sentence entries (id, de, ch_<dialect>...) from the SwissDial metadata JSON."""
    root = Path(root).expanduser()
    files = sorted(root.rglob("sentences_ch_de*.json"))
    if not files:
        raise FileNotFoundError(f"No sentences_ch_de*.json under {root} (extract SwissDial data.tar.gz there)")
    meta = next((f for f in files if "numerics" in f.name), files[0])
    data = json.loads(meta.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [dict(v, id=k) if isinstance(v, dict) and "id" not in v else v for k, v in data.items()]
    return [e for e in data if isinstance(e, dict) and "id" in e]


def load_swissdial_audio(root: str | Path, dialect: str = "zh") -> list[dict]:
    root = Path(root).expanduser()
    entries = {str(e["id"]).lstrip("0") or "0": e for e in load_swissdial_text(root)}
    prefix = FILE_PREFIX.get(dialect, dialect)
    clips, seen = [], set()
    for wav in root.rglob(f"ch_{prefix}_*.wav"):
        m = WAV_RE.search(wav.name)
        sid = (m.group("id").lstrip("0") or "0") if m else None
        e = entries.get(sid) if sid else None
        if not e or sid in seen:
            continue
        gsw, de = (e.get(f"ch_{dialect}") or "").strip(), (e.get("de") or "").strip()
        if gsw and de:
            seen.add(sid)
            clips.append({"id": sid, "wav": str(wav), "gsw": gsw, "de": de})
    return sorted(clips, key=lambda c: int(c["id"]))


def sample(items: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return rng.sample(items, min(n, len(items)))


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower().replace("ß", "ss")
    text = re.sub(r"[’'`´]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chrf(hyps: list[str], refs: list[str]) -> float:
    from sacrebleu.metrics import CHRF

    return CHRF().corpus_score([normalise(h) for h in hyps], [[normalise(r) for r in refs]]).score


def pmap(fn, items: list, workers: int | None = None) -> list:
    # EVAL_WORKERS / EVAL_DELAY (seconds after each call) keep the request rate under the endpoint's limit:
    # Public AI's free tier allows 100 requests per minute, and cached answers come back in well under a second.
    workers = workers or int(os.environ.get("EVAL_WORKERS", "4"))
    delay = float(os.environ.get("EVAL_DELAY", "0"))
    done = [0]

    def wrapped(x):
        try:
            return fn(x)
        finally:
            time.sleep(delay)
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(items):
                print(f"  {done[0]}/{len(items)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(wrapped, items))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f} % ({a}/{b})" if b else "n/a"


def first_line(text: str) -> str:
    line = next((ln for ln in (text or "").splitlines() if ln.strip()), "")
    return line.strip().strip('"„“«»').strip()
