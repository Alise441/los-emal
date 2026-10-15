"""E4 — How reliable is the app's feedback on the learner's Standard German?

Runs the real app prompt (losemal.tutor.Session) on data/learner_de_testset.jsonl:
the Zurich line is put into the conversation history, the learner's answer is sent,
and we read the structured feedback.

Reported:
- false alarms: correct sentences (incl. Swiss Standard German like «parkieren») flagged as wrong;
- error detection: wrong sentences flagged as wrong;
- correction quality: the correction contains the expected fix;
- comprehension check: misunderstandings detected / false misunderstanding alarms.
"""
from __future__ import annotations

import argparse
import json

from losemal.config import DATA_DIR
from losemal.llm import ApertusClient
from losemal.tutor import Session, Turn

from .common import normalise, out_dir, pct, pmap, write_jsonl


def run_item(cli: ApertusClient, item: dict) -> dict:
    s = Session(scenario_id=item["scenario"], level="B1", help_lang="en")
    s.history = [{"role": "user", "content": "(Die Szene beginnt.)"},
                 {"role": "assistant", "content": item["context_gsw"]}]
    s.turns = [Turn(reply_gsw=item["context_gsw"], reply_de=item["context_de"])]  # as the app has it
    try:
        t = s.respond(cli, item["learner_de"])
    except Exception as e:  # noqa: BLE001
        return dict(item, error=str(e))
    fb = t.feedback or {}
    return dict(item, pred_ok=fb.get("ok") if fb else None, corrected=fb.get("corrected_de", ""),
                tip=fb.get("tip", ""), pred_understood=t.understood, parse_ok=t.parse_ok,
                reply_gsw=t.reply_gsw, seconds=round(t.seconds, 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="swiss-ai/apertus-v1.5-70b,swiss-ai/apertus-v1.5-8b")
    ap.add_argument("--data", default=str(DATA_DIR / "learner_de_testset.jsonl"))
    args = ap.parse_args()
    items = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]

    out = out_dir("e4_feedback")
    lines = ["# E4 — Feedback on the learner's Standard German", "",
             f"{len(items)} synthetic items (data/learner_de_testset.jsonl): "
             + ", ".join(f"{k}: {sum(i['kind'] == k for i in items)}"
                         for k in ("correct", "swiss_variant", "error", "misunderstanding")), ""]
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print("==", model)
        cli = ApertusClient(model=model)
        rows = pmap(lambda it: run_item(cli, it), items)
        write_jsonl(out / f"{model.split('/')[-1]}.jsonl", rows)
        ok_rows = [r for r in rows if "error" not in r and r.get("pred_ok") is not None]

        correct = [r for r in ok_rows if r["gold_ok"]]
        swiss = [r for r in correct if r["kind"] == "swiss_variant"]
        wrong = [r for r in ok_rows if not r["gold_ok"]]
        flagged = [r for r in wrong if r["pred_ok"] is False]
        fixed = [r for r in flagged if any(x in normalise(r["corrected"]) for x in r["expected_in_correction"])]
        mis = [r for r in rows if "error" not in r and r["gold_understood"] is False]
        fit = [r for r in rows if "error" not in r and r["gold_understood"] is True]

        lines += [f"## {model}",
                  f"- false alarms on correct German: {pct(sum(r['pred_ok'] is False for r in correct), len(correct))}"
                  f" (Swiss Standard German items: {pct(sum(r['pred_ok'] is False for r in swiss), len(swiss))})",
                  f"- errors detected: {pct(len(flagged), len(wrong))}",
                  f"- … with the expected correction: {pct(len(fixed), len(wrong))}",
                  f"- misunderstandings detected: {pct(sum(r['pred_understood'] is False for r in mis), len(mis))}",
                  f"- false 'misunderstood' alarms: {pct(sum(r['pred_understood'] is False for r in fit), len(fit))}",
                  f"- failed calls / unparseable: {sum('error' in r for r in rows)} / "
                  f"{sum(1 for r in rows if 'error' not in r and not r.get('parse_ok', True))}", ""]
        missed = [r for r in wrong if r not in fixed][:6]
        if missed:
            lines.append("Missed or imperfect corrections:")
            lines += [f"- «{r['learner_de']}» → «{r.get('corrected', '')}»" for r in missed]
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
