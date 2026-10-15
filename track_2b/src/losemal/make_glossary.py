"""Generate the vocabulary of each situation once, with Apertus, into data/glossary.json.

For every scene, Apertus lists the Zurich German words and short phrases a newcomer is likely to hear
there and not know from a German course, with their Standard German and English meaning. The file is
generated once and then reviewed by hand, so the app shows the same checked list to every learner; each
entry's "source" says whether it is Apertus' output, Apertus' output corrected, or added in the review.
Running this again overwrites the reviewed file.

    python -m losemal.make_glossary                 # all scenes (uses LLM_NAME / LLM_BASE_URL / LLM_API_KEY)
    python -m losemal.make_glossary --scenes sbb    # only some scenes, the others are kept
"""
from __future__ import annotations

import argparse
import json

from .config import DATA_DIR
from .llm import ApertusClient, extract_json
from .prompts import ZURICH_STYLE_GUIDE
from .scenarios import BY_ID, SCENARIOS

OUT = DATA_DIR / "glossary.json"

PROMPT = """\
Du hilfst bei der App «Los emal!», mit der Zugezogene lernen, Zürichdeutsch zu verstehen.
Situation: {setting}
Wer spricht: {role}
Wer zuhört: {learner}

Erstelle ein kleines Wörterbuch für diese Situation: {n} Wörter oder kurze Wendungen, die man hier von
Einheimischen auf Züritüütsch hört und die Zugezogene aus dem Deutschkurs nicht kennen. Etwa zwei Drittel
einzelne Wörter (Nomen, Verben, Ausdrücke), ein Drittel typische kurze Sätze, die in dieser Situation fallen.
Nur echtes, gebräuchliches Stadtzürcher Züritüütsch; keine Wörter, die gleich sind wie auf Hochdeutsch.

{style}

Antworte nur mit JSON:
{{"words": [{{"gsw": "Wort oder Wendung auf Züritüütsch", "de": "Bedeutung auf Hochdeutsch", "en": "meaning in English"}}]}}"""


def clean(items, n: int) -> list[dict]:
    """Well-formed, distinct entries (gsw, de, en), at most n."""
    seen, out = set(), []
    for w in items if isinstance(items, list) else []:
        if not isinstance(w, dict):
            continue
        entry = {k: str(w.get(k) or "").strip() for k in ("gsw", "de", "en")}
        key = entry["gsw"].lower()
        if all(entry.values()) and key not in seen and entry["gsw"].lower() != entry["de"].lower():
            seen.add(key)
            out.append({**entry, "source": "apertus"})
    return out[:n]


def generate(client: ApertusClient, scenario_id: str, n: int) -> list[dict]:
    sc = BY_ID[scenario_id]
    prompt = PROMPT.format(setting=sc.setting, role=sc.role, learner=sc.learner, n=n, style=ZURICH_STYLE_GUIDE)
    reply = client.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1500)
    return clean((extract_json(reply.text) or {}).get("words"), n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=[s.id for s in SCENARIOS])
    ap.add_argument("--n", type=int, default=12, help="entries per scene")
    args = ap.parse_args()
    glossary = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    client = ApertusClient()
    for scenario_id in args.scenes:
        glossary[scenario_id] = generate(client, scenario_id, args.n)
        print(f"{scenario_id}: {len(glossary[scenario_id])} entries")
    ordered = {s.id: glossary[s.id] for s in SCENARIOS if s.id in glossary}
    OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT, "with model", client.model)


if __name__ == "__main__":
    main()
