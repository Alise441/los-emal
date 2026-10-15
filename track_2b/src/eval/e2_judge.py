"""E2 — Can Apertus judge whether a sentence is natural Zurich German?

This decides the product design: if the model cannot tell Zurich German from Standard
German or another dialect, it must not grade a learner's Zurich German.

Test items from SwissDial (same sentences in three versions):
  ch_zh → should be judged Zurich German (true)
  de    → Standard German (false)
  ch_be → Bernese German, a different dialect (false)
"""
from __future__ import annotations

import argparse

from losemal.llm import ApertusClient, extract_json

from .common import load_swissdial_text, out_dir, pct, pmap, sample, write_jsonl

PROMPT = ("Ist der folgende Satz natürliches Zürichdeutsch (Stadtzürcher Dialekt), so wie man ihn in Zürich "
          "sagen und in Dialektschreibung aufschreiben würde? Hochdeutsch oder ein anderer Schweizer Dialekt "
          "zählt NICHT als Zürichdeutsch.\n\nSatz: «{s}»\n\n"
          'Antworte nur mit JSON: {{"zurich": true}} oder {{"zurich": false}}')


def judge(cli: ApertusClient, sentence: str) -> bool | None:
    try:
        reply = cli.chat([{"role": "user", "content": PROMPT.format(s=sentence)}], temperature=0.0, max_tokens=40)
    except Exception:  # noqa: BLE001
        return None
    data = extract_json(reply.text) or {}
    v = data.get("zurich")
    return v if isinstance(v, bool) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissdial", required=True)
    ap.add_argument("--n", type=int, default=50, help="sentences per source")
    ap.add_argument("--models", default="swiss-ai/apertus-v1.5-70b,swiss-ai/apertus-v1.5-8b")
    args = ap.parse_args()

    entries = [e for e in load_swissdial_text(args.swissdial) if e.get("ch_zh") and e.get("ch_be") and e.get("de")]
    picked = sample(entries, args.n, seed=2)
    items = []
    for e in picked:
        items += [{"id": e["id"], "source": "ch_zh", "text": e["ch_zh"], "gold": True},
                  {"id": e["id"], "source": "de", "text": e["de"], "gold": False},
                  {"id": e["id"], "source": "ch_be", "text": e["ch_be"], "gold": False}]

    out = out_dir("e2_judge")
    lines = ["# E2 — Apertus as a judge of Zurich German", "",
             f"{args.n} SwissDial sentences, each as Zurich German (should be accepted), Standard German and "
             "Bernese German (both should be rejected).", ""]
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print("==", model)
        cli = ApertusClient(model=model)
        preds = pmap(lambda it: judge(cli, it["text"]), items)
        rows = [dict(it, pred=p) for it, p in zip(items, preds)]
        write_jsonl(out / f"{model.split('/')[-1]}.jsonl", rows)
        valid = [r for r in rows if r["pred"] is not None]
        acc = sum(r["pred"] == r["gold"] for r in valid)
        lines.append(f"## {model}")
        lines.append(f"- overall accuracy: {pct(acc, len(valid))} (chance = 33 % if always 'yes', 67 % if always 'no')")
        for src, name in (("ch_zh", "Zurich German accepted"), ("de", "Standard German rejected"),
                          ("ch_be", "Bernese German rejected")):
            sub = [r for r in valid if r["source"] == src]
            lines.append(f"- {name}: {pct(sum(r['pred'] == r['gold'] for r in sub), len(sub))}")
        lines.append(f"- unparseable answers: {len(rows) - len(valid)}")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
