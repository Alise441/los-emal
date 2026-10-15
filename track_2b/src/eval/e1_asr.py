"""E1 — Who understands spoken Zurich German: Whisper or Apertus 1.5 with audio input?

Takes N SwissDial Zurich clips and asks each system for the Standard German meaning,
then scores against the SwissDial Standard German sentence (WER + chrF, normalised).
Results are appended per clip, so the run can be stopped and resumed.

Needs its own environment, separate from the app (see `make venv-asr`): Apertus audio input
requires the Swiss AI transformers fork (commit 3797303), and Whisper runs with mlx-whisper on
Apple Silicon.

  make venv-asr
  make eval-asr SWISSDIAL=/path/to/swissdial            # whisper + apertus, 100 clips

Systems:
  whisper   mlx-whisper, default mlx-community/whisper-large-v3-turbo
  apertus   Apertus 1.5 8B with audio input, local weights
  hf:<id>   any Hugging Face speech-recognition model via the transformers pipeline

Besides WER/chrF, the summary counts clips where Apertus did not attempt the task (it claimed it
could not hear the audio, or repeated the instruction) and scores all systems without them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported MPS ops fall back to CPU

from .common import first_line, load_swissdial_audio, normalise, out_dir, sample

PROMPT = (
    "Hör dir diese Aufnahme auf Schweizerdeutsch an und gib genau wieder, was gesagt wird, "
    "aber auf Hochdeutsch. Antworte nur mit dem hochdeutschen Satz, ohne Erklärungen."
)

# Replies in which the model says it cannot hear / did not receive the audio.
REFUSAL = re.compile(
    r"es tut mir leid|nicht gehört|nicht hören|keine (audio|aufnahme)|nicht zur verfügung|nicht bereitgestellt"
    r"|als (text-?ki|ki|sprachmodell)|textbasiert",
    re.IGNORECASE,
)
PROMPT_WORDS = set(normalise(PROMPT).split())


def non_answer(hyp: str) -> str | None:
    """'refusal', 'echo' (the instruction repeated back) or None for an attempted transcription."""
    if REFUSAL.search(hyp or ""):
        return "refusal"
    words = set(normalise(hyp).split())
    if len(words) >= 5 and len(words & PROMPT_WORDS) / len(words) >= 0.6:
        return "echo"
    return None


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Whisper:
    def __init__(self, model_id: str):
        import mlx_whisper

        self.mod, self.model_id = mlx_whisper, model_id

    def __call__(self, wav: str) -> str:
        r = self.mod.transcribe(wav, path_or_hf_repo=self.model_id, language="de", task="transcribe", verbose=None)
        return r.get("text", "").strip()


class HFPipeline:
    def __init__(self, model_id: str):
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=model_id, device=pick_device())

    def __call__(self, wav: str) -> str:
        return self.pipe(wav, generate_kwargs={"language": "german", "task": "transcribe"})["text"].strip()


class ApertusAudio:
    """Apertus 1.5 with audio input. Audio is converted to 24 kHz mono first (model card)."""

    def __init__(self, model_id: str, tmp_dir: Path):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.tmp = tmp_dir
        self.device = pick_device()
        if self.device == "mps":
            # Apertus pads its 131072 text logits to the 266752-token vocabulary with F.pad, and torch's MPS
            # constant-pad kernel corrupts a pad that wide (logits come out as zeros), so every generated
            # token is noise. torch.cat builds the same padded tensor correctly.
            from transformers.models.apertus1p5 import modeling_apertus1p5

            def pad_logits(logits, vocab_size):
                tail = logits.new_full((*logits.shape[:-1], vocab_size - logits.shape[-1]), torch.finfo(logits.dtype).min)
                return torch.cat([logits, tail], dim=-1)

            modeling_apertus1p5._pad_logits_to_vocab_size = pad_logits
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        if self.device == "cuda":
            self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype, device_map="auto").eval()
        else:
            self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.audio_key = None  # discovered on the first clip: "path" or "url"
        self.cast_inputs = False
        print(f"Apertus loaded on {self.device} ({dtype})")

    def _to_24k(self, wav: str) -> str:
        import librosa
        import soundfile as sf

        out = self.tmp / (Path(wav).stem + "_24k.wav")
        if not out.exists():
            y, _ = librosa.load(wav, sr=24000, mono=True)
            sf.write(out, y, 24000)
        return str(out)

    def _inputs(self, audio_path: str, key: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "audio", key: audio_path}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        if self.cast_inputs:
            return inputs.to(self.model.device, dtype=self.model.dtype)
        return inputs.to(self.model.device)

    def __call__(self, wav: str) -> str:
        audio = self._to_24k(wav)
        keys = [self.audio_key] if self.audio_key else ["path", "url"]
        last = None
        for key in keys:
            for cast in ([self.cast_inputs] if self.audio_key else [False, True]):
                try:
                    self.cast_inputs = cast
                    inputs = self._inputs(audio, key)
                    with self.torch.inference_mode():
                        out = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                    self.audio_key = key
                    return self.processor.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                except Exception as e:  # noqa: BLE001
                    last = e
        raise RuntimeError(f"Apertus audio call failed: {last}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score(rows: list[dict]) -> dict:
    import jiwer
    from sacrebleu.metrics import CHRF

    refs = [normalise(r["ref_de"]) for r in rows]
    hyps = [normalise(r["hyp"]) for r in rows]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    return {
        "n": len(pairs),
        "wer": 100 * jiwer.wer([p[0] for p in pairs], [p[1] or "-" for p in pairs]),
        "chrf": CHRF().corpus_score([p[1] for p in pairs], [[p[0] for p in pairs]]).score,
        "sec": statistics.mean(r["seconds"] for r in rows) if rows else 0.0,
    }


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissdial", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--systems", default="whisper,apertus")
    ap.add_argument("--whisper-model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--apertus-model", default="swiss-ai/Apertus-v1.5-8B")
    args = ap.parse_args()

    out = out_dir("e1_asr")
    tmp = out_dir("e1_asr/audio_24k")
    clips = sample(load_swissdial_audio(args.swissdial, "zh"), args.n, seed=13)
    clip_ids = {c["id"] for c in clips}
    print(f"{len(clips)} ZH clips")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    all_rows: dict[str, list[dict]] = {}
    for system in systems:
        path = out / f"{slug(system)}.jsonl"
        done = {r["id"] for r in read_jsonl(path) if not r.get("error")}  # failed clips are retried
        todo = [c for c in clips if c["id"] not in done]
        print(f"== {system}: {len(done & clip_ids)} done, {len(todo)} to go")
        if todo:
            try:
                if system == "whisper":
                    run = Whisper(args.whisper_model)
                elif system == "apertus":
                    run = ApertusAudio(args.apertus_model, tmp)
                elif system.startswith("hf:"):
                    run = HFPipeline(system[3:])
                else:
                    raise ValueError(f"unknown system {system}")
            except Exception as e:  # noqa: BLE001
                print(f"!! {system} could not start: {e}")
                continue
            for i, c in enumerate(todo, 1):
                t0 = time.time()
                try:
                    hyp = first_line(run(c["wav"]))
                    err = None
                except Exception as e:  # noqa: BLE001
                    hyp, err = "", str(e)
                row = {"id": c["id"], "ref_de": c["de"], "ref_gsw": c["gsw"], "hyp": hyp,
                       "seconds": round(time.time() - t0, 2)}
                if err:
                    row["error"] = err
                    print(f"  {c['id']}: ERROR {err[:150]}")
                append_jsonl(path, row)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
        latest = {r["id"]: r for r in read_jsonl(path)}  # last attempt per clip wins
        all_rows[system] = [r for cid, r in latest.items() if cid in clip_ids]

    # score every system on the same clips
    ok_ids = [{r["id"] for r in rows if not r.get("error")} for rows in all_rows.values()]
    common_ids = set.intersection(*ok_ids) if ok_ids else set()
    lines = ["# E1 — understanding spoken Zurich German", "",
             f"{len(common_ids)} SwissDial ZH clips (seed 13), reference = SwissDial Standard German sentence. "
             "Lower WER and higher chrF are better. Both are rough, since a correct paraphrase is also penalised.", ""]
    for system, rows in all_rows.items():
        rows = [r for r in rows if r["id"] in common_ids]
        if not rows:
            lines.append(f"- **{system}**: no successful clips")
            continue
        s = score(rows)
        lines.append(f"- **{system}**: WER {s['wer']:.1f} %, chrF {s['chrf']:.1f}, {s['sec']:.1f} s per clip (n={s['n']})")

    # Clips where a system did not attempt the task. Only model outputs are listed, never SwissDial text.
    flagged = {}
    for system, rows in all_rows.items():
        for r in rows:
            kind = non_answer(r["hyp"]) if r["id"] in common_ids else None
            if kind:
                flagged.setdefault(system, []).append((r["id"], kind, r["hyp"]))
    lines += ["", "## Non-answers (task not attempted)", ""]
    for system in all_rows:
        items = flagged.get(system, [])
        kinds = {k: sum(1 for _, kind, _ in items if kind == k) for k in ("refusal", "echo")}
        lines.append(f"- **{system}**: {len(items)} of {len(common_ids)} "
                     f"({kinds['refusal']} said it could not hear the audio, {kinds['echo']} repeated the instruction)")
    skip = {cid for items in flagged.values() for cid, _, _ in items}
    if skip:
        keep = common_ids - skip
        lines += ["", f"Scores without these {len(skip)} clips (n={len(keep)}):"]
        for system, rows in all_rows.items():
            s = score([r for r in rows if r["id"] in keep])
            lines.append(f"- **{system}**: WER {s['wer']:.1f} %, chrF {s['chrf']:.1f}")
        lines += ["", "Flagged model outputs (check by hand):"]
        lines += [f"- {system} / clip {cid} / {kind}: {hyp[:140]}"
                  for system, items in flagged.items() for cid, kind, hyp in items]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
