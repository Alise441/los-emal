"""Generate each scene's first line once, with Apertus, into data/openings.json.

The opening line, its translations (Standard German, English), its hard words (explained in English and in
German) and two reply ideas are prepared in advance, so a scene starts without waiting for the model.
The opening lines are reviewed by hand after generating; --keep-lines then regenerates everything else
for the reviewed lines. Running without it overwrites the lines too.

    python -m losemal.make_openings                 # all scenes (uses LLM_NAME / LLM_BASE_URL / LLM_API_KEY)
    python -m losemal.make_openings --scenes sbb    # only some scenes, the others are kept
    python -m losemal.make_openings --keep-lines    # keep the (reviewed) lines, redo translations, words, ideas
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from .config import DATA_DIR
from .llm import ApertusClient
from .prompts import scene_context
from .scenarios import BY_ID, SCENARIOS
from .tutor import Session, hard_words, suggest_replies, translate

OUT = DATA_DIR / "openings.json"


def generate(client: ApertusClient, scenario_id: str, line: str = "") -> dict:
    sc = BY_ID[scenario_id]
    sess = Session(scenario_id=scenario_id, level="B1", help_lang="en")
    turn = sess.open_with({"gsw": line}) if line else sess.start(client)
    context = scene_context(sc)
    with ThreadPoolExecutor(max_workers=3) as pool:
        words = {lang: pool.submit(hard_words, client, turn.reply_gsw, lang, context) for lang in ("en", "de")}
        ideas = pool.submit(suggest_replies, client, sess)
        translate(client, turn, "en", sc)
        return {"gsw": turn.reply_gsw, "de": turn.reply_de, "help": {"en": turn.reply_help},
                "words": {lang: job.result() for lang, job in words.items()}, "ideas": ideas.result()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=[s.id for s in SCENARIOS])
    ap.add_argument("--keep-lines", action="store_true", help="keep the opening lines already in the file")
    args = ap.parse_args()
    openings = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    client = ApertusClient()
    for scenario_id in args.scenes:
        line = openings.get(scenario_id, {}).get("gsw", "") if args.keep_lines else ""
        openings[scenario_id] = generate(client, scenario_id, line)
        print(f"{scenario_id}: {openings[scenario_id]['gsw']}")
    ordered = {s.id: openings[s.id] for s in SCENARIOS if s.id in openings}
    OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT, "with model", client.model)


if __name__ == "__main__":
    main()
