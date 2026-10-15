"""E3 — Does the Zurich style guide make Apertus' dialect more Zurich-like?

Task: translate SwissDial Standard German sentences into Zurich German.
Prompts: (a) baseline instruction from our pilot, (b) the same plus the style guide used in the app.
Metrics:
- chrF against the SwissDial ch_zh transcript (character overlap, spelling-tolerant);
- Zurich marker share: among words that differ between Zurich and other dialects
  (hät/het, nöd/nid, gsi/gsy, de/dr, ...), the share of Zurich forms;
- Standard German leftovers per 100 words (ist, nicht, hat, ein, ...).
The two lexical metrics are validated on SwissDial references: Zurich references must score
high and Bernese references low, otherwise the metric is meaningless.
"""
from __future__ import annotations

import argparse
import statistics

from losemal.llm import ApertusClient
from losemal.prompts import TRANSLATE_TO_ZURICH as BASE
from losemal.prompts import ZURICH_STYLE_GUIDE

from .common import chrf, first_line, load_swissdial_text, normalise, out_dir, pmap, sample, write_jsonl

GUIDED = BASE + "\n\n" + ZURICH_STYLE_GUIDE  # the app's "how a local would say it" prompt

# Word forms that differ between Zurich and other Swiss German dialects (lower-case, after normalise()).
ZURICH = {"hät", "händ", "nöd", "nüd", "gsi", "und", "de"}
OTHER = {"het", "hed", "hei", "hän", "nid", "nit", "ned", "gsy", "gsii", "u", "dr", "ig", "mau", "viu", "wäut", "fäut"}
STANDARD = {"ist", "nicht", "hat", "haben", "war", "wurde", "ein", "eine", "einen", "einem", "einer", "auch", "sehr"}


def markers(texts: list[str]) -> dict:
    z = o = s = n = 0
    for t in texts:
        toks = normalise(t).split()
        n += len(toks)
        z += sum(tok in ZURICH for tok in toks)
        o += sum(tok in OTHER for tok in toks)
        s += sum(tok in STANDARD for tok in toks)
    return {"zurich_share": z / (z + o) if z + o else float("nan"), "zurich": z, "other": o,
            "standard_per_100": 100 * s / n if n else 0.0}


def fmt(m: dict) -> str:
    share = "n/a" if m["zurich_share"] != m["zurich_share"] else f"{100 * m['zurich_share']:.0f} %"
    return (f"Zurich marker share {share} ({m['zurich']} vs {m['other']}), "
            f"Standard German leftovers {m['standard_per_100']:.1f} per 100 words")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissdial", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", default="swiss-ai/apertus-v1.5-70b")
    args = ap.parse_args()

    entries = [e for e in load_swissdial_text(args.swissdial) if e.get("ch_zh") and e.get("de")]
    items = sample(entries, args.n, seed=3)
    refs = [e["ch_zh"] for e in items]
    be_refs = [e["ch_be"] for e in items if e.get("ch_be")]

    out = out_dir("e3_generation")
    lines = ["# E3 — Zurich German generation with and without the style guide", "",
             f"{len(items)} SwissDial sentences, Standard German → Zurich German.", "",
             "Metric sanity check on SwissDial references:",
             f"- Zurich references: {fmt(markers(refs))}",
             f"- Bernese references: {fmt(markers(be_refs))}",
             f"- Standard German input (copy baseline): chrF {chrf([e['de'] for e in items], refs):.1f}, "
             f"{fmt(markers([e['de'] for e in items]))}", ""]

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        cli = ApertusClient(model=model)
        lines.append(f"## {model}")
        for name, system in (("baseline prompt", BASE), ("with style guide", GUIDED)):
            print(f"== {model} / {name}")

            def run(e, system=system):
                try:
                    r = cli.chat([{"role": "system", "content": system}, {"role": "user", "content": e["de"]}],
                                 temperature=0.0, max_tokens=200)
                    return first_line(r.text), r.seconds
                except Exception:  # noqa: BLE001
                    return "", 0.0

            res = pmap(run, items)
            hyps = [h for h, _ in res]
            write_jsonl(out / f"{model.split('/')[-1]}_{name.replace(' ', '_')}.jsonl",
                        [{"id": e["id"], "de": e["de"], "ref": e["ch_zh"], "hyp": h} for e, h in zip(items, hyps)])
            ok = [(h, r) for h, r in zip(hyps, refs) if h]
            lat = [s for _, s in res if s]
            lines.append(f"- **{name}**: chrF {chrf([h for h, _ in ok], [r for _, r in ok]):.1f}; "
                         f"{fmt(markers([h for h, _ in ok]))}; median latency "
                         f"{statistics.median(lat) if lat else float('nan'):.1f} s; failed {len(hyps) - len(ok)}")
        lines.append("")

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
