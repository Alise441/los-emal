"""Conversation logic. Per learner turn: the role-play turn and, in parallel, short Apertus calls (did the
learner understand the local's line, the learner's sentence in Zurich German). Once the local's line is
there, translate(), hard_words() and suggest_replies() add its translations, hard words and two ideas for
the learner's answer, again in short calls; a scene's first line comes ready-made (Session.open_with).
Apertus does short, single tasks far more reliably than many fields in one reply (technical report, E4).
The app trains understanding: it does not correct the learner's German."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from .config import settings
from .llm import ApertusClient, LLMError, extract_json
from .prompts import (FINETUNED_TRANSLATE, TRANSLATE_TO_ZURICH, ZURICH_STYLE_GUIDE, comprehension_message,
                      german_check_message,
                      learner_message, opening_message, scene_context, suggest_message, system_prompt,
                      translate_message, words_message)
from .scenarios import BY_ID, Scenario

log = logging.getLogger(__name__)

MAX_HISTORY = 14   # messages kept in context (7 exchanges)
MAX_TURNS = 8      # learner turns before the scene is closed


@dataclass
class Turn:
    reply_gsw: str
    reply_de: str = ""
    reply_help: str = ""
    words: list[dict] = field(default_factory=list)
    understood: bool | None = None
    understood_note: str = ""
    local_gsw: str = ""
    suggestions_de: list[str] = field(default_factory=list)
    scene_done: bool = False
    learner_text: str = ""
    seconds: float = 0.0
    parse_ok: bool = True


def _s(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _b(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, str):
        return {"true": True, "false": False}.get(v.strip().lower())
    return None


def to_turn(data: dict | None, raw: str, learner_text: str, seconds: float) -> Turn:
    if data and not _s(data.get("reply_gsw")) and _s(data.get("reply_de")):  # DIALOGUE=de: translated later
        return Turn(reply_gsw="", reply_de=_s(data.get("reply_de")), scene_done=bool(_b(data.get("scene_done"))),
                    learner_text=learner_text, seconds=seconds)
    if not data or not _s(data.get("reply_gsw")):
        # Model ignored the format: still show something usable instead of failing.
        text = raw.strip().strip("{}").strip() or "Exgüsi, chönd Sie das nomal säge?"
        return Turn(reply_gsw=text[:400], learner_text=learner_text, seconds=seconds, parse_ok=False)

    return Turn(
        reply_gsw=_s(data.get("reply_gsw")),
        scene_done=bool(_b(data.get("scene_done"))),
        learner_text=learner_text,
        seconds=seconds,
    )


def parse_feedback(fb, learner_text: str) -> dict | None:
    """Verdict of the Standard German check (evaluation E4 only; the app does not correct German)."""
    if not isinstance(fb, dict) or not learner_text:
        return None
    ok = _b(fb.get("ok"))
    corrected = _s(fb.get("corrected_de"))
    # a "correction" identical to the input is not a correction
    if ok is False and corrected and _norm(corrected) == _norm(learner_text):
        ok = True
    if ok is None:  # no verdict -> show no feedback rather than a "correct" the model never gave
        return None
    if ok is False and not corrected:  # "wrong" without a correction gives the learner nothing to act on
        return None
    return {"ok": ok, "corrected_de": corrected if ok is False else "", "tip": _s(fb.get("tip"))}


# Unmistakable Zurich German words: a suggestion containing one is not the Standard German the learner speaks.
DIALECT = re.compile(r"\b(isch|händ|hät|nöd|nüt|gsi|chan|chönd|wänd|wott|zrugg|zruck|morn|hüt|öppis|chli|gärn"
                     r"|zäme|mer|bruuch\w*|chueche|charte)\b", re.IGNORECASE)


def clean_suggestions(suggestions: list[str], said: list[str]) -> list[str]:
    """Drop suggestions in dialect or repeating what the learner already said (or each other)."""
    out: list[str] = []
    for s in suggestions:
        if DIALECT.search(s):
            continue
        if all(SequenceMatcher(None, _norm(s), _norm(t)).ratio() < 0.8 for t in said + out):
            out.append(s)
    return out


def check_comprehension(client: ApertusClient, said: Turn, answer: str, help_lang: str) -> tuple[bool | None, str]:
    """Does the learner's answer fit the model's last line? (None, "") if it fails or gives no verdict."""
    try:
        reply = client.chat([{"role": "user", "content": comprehension_message(
            said.reply_gsw, said.reply_de, answer, help_lang)}], temperature=0.0, max_tokens=150)
    except LLMError as e:
        log.warning("Comprehension check failed: %s", e)
        return None, ""
    data = extract_json(reply.text) or {}
    fits = _b(data.get("fits"))
    return fits, _s(data.get("note")) if fits is False else ""


def check_german(client: ApertusClient, learner_text: str, help_lang: str) -> dict | None:
    """Short, focused check of the learner's Standard German; None if it fails. Used by the evaluation (E4)."""
    try:
        reply = client.chat([{"role": "user", "content": german_check_message(learner_text, help_lang)}],
                            temperature=0.0, max_tokens=200)
    except LLMError as e:
        log.warning("Standard German check failed: %s", e)
        return None
    return parse_feedback(extract_json(reply.text), learner_text)


def translate_line(client: ApertusClient, text: str, target: str, context: str = "", from_de: bool = False) -> str:
    """The local's line (or its Standard German version) in Standard German or the help language; "" if it fails."""
    try:
        reply = client.chat([{"role": "user", "content": translate_message(text, target, context, from_de)}],
                            temperature=0.0, max_tokens=200)
    except LLMError as e:
        log.warning("Translation (%s) failed: %s", target, e)
        return ""
    return reply.text.strip().split("\n\n")[0].strip().strip('"„“«»').strip()


def hard_words(client: ApertusClient, text: str, help_lang: str, context: str = "") -> list[dict]:
    """1-3 dialect words of the line with their Standard German meaning, explained in the help language."""
    try:
        reply = client.chat([{"role": "user", "content": words_message(text, help_lang, context)}],
                            temperature=0.0, max_tokens=300)
    except LLMError as e:
        log.warning("Word explanations failed: %s", e)
        return []
    return clean_words((extract_json(reply.text) or {}).get("words"), text)


def clean_words(items, line: str) -> list[dict]:
    """Keep short, well-formed words that actually occur in the line (the model sometimes invents one or
    returns a whole sentence)."""
    words = []
    for w in items if isinstance(items, list) else []:
        gsw = _s(w.get("gsw")) if isinstance(w, dict) else ""
        if gsw and len(gsw.split()) <= 3 and _norm(gsw) in _norm(line):
            words.append({"gsw": _s(w.get("gsw")), "de": _s(w.get("de")), "hint": _s(w.get("hint"))})
    return words[:3]


def translate(client: ApertusClient, turn: Turn, help_lang: str, sc: Scenario | None = None) -> None:
    """The local's line in Standard German (unless Apertus wrote it in Standard German itself), then in the help
    language from that version (which the model translates more reliably than the dialect)."""
    context = scene_context(sc) if sc else ""
    if not turn.reply_de:
        turn.reply_de = translate_line(client, turn.reply_gsw, "de", context)
    turn.reply_help = ""
    if help_lang != "de":
        turn.reply_help = translate_line(client, turn.reply_de or turn.reply_gsw, help_lang, context,
                                         from_de=bool(turn.reply_de))


def explain(client: ApertusClient, turn: Turn, help_lang: str, sc: Scenario | None = None) -> None:
    """Translations and hard words of the local's line: separate short calls, two at a time."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        words = pool.submit(hard_words, client, turn.reply_gsw, help_lang, scene_context(sc) if sc else "")
        translate(client, turn, help_lang, sc)
        turn.words = words.result()


def suggest_replies(client: ApertusClient, sess: Session) -> list[str]:
    """Two ideas for the learner's answer to the local's latest line; [] if the call fails."""
    said = [t.learner_text for t in sess.turns if t.learner_text]
    message = suggest_message(sess.scenario, sess.level, sess.turns[-1].reply_gsw, said)
    try:
        reply = client.chat([{"role": "user", "content": message}], temperature=0.4, max_tokens=200)
    except LLMError as e:
        log.warning("Reply ideas failed: %s", e)
        return []
    ideas = (extract_json(reply.text) or {}).get("ideas")
    return clean_suggestions([_s(x) for x in ideas if _s(x)] if isinstance(ideas, list) else [], said)[:2]


# Greetings, thanks and farewells, checked by hand. The fine-tuned translator learnt from SwissDial's read
# sentences, which hardly contain them, and invented words for them («Uf Wiederhersäge» for «Auf Wiedersehen»);
# a sentence that is exactly one of these is taken from here. Keys: lower case, without punctuation.
FORMULAS = {
    "auf wiedersehen": "Uf Widerluege", "wiedersehen": "Uf Widerluege", "tschüss": "Tschüss", "ciao": "Ciao",
    "danke tschüss": "Merci, tschüss", "danke auf wiedersehen": "Merci, uf Widerluege",
    "guten tag": "Grüezi", "grüezi": "Grüezi", "grüss gott": "Grüezi", "hallo": "Hoi", "hoi": "Hoi", "sali": "Sali",
    "guten morgen": "Guete Morge", "guten abend": "Guete Abig",
    "danke": "Merci", "danke schön": "Merci vilmal", "dankeschön": "Merci vilmal", "vielen dank": "Merci vilmal",
    "danke vielmals": "Merci vilmal", "herzlichen dank": "Merci vilmal", "besten dank": "Merci vilmal",
    "gern geschehen": "Gärn gscheh", "gerne geschehen": "Gärn gscheh", "bitte schön": "Bitte schön",
    "bitteschön": "Bitte schön", "bitte sehr": "Bitte schön",
    "einen schönen tag noch": "En schöne Tag no", "noch einen schönen tag": "En schöne Tag no",
    "schönen tag noch": "Schöne Tag no", "einen schönen abend noch": "En schöne Abig no",
    "schönen abend noch": "Schöne Abig no", "schönes wochenende": "Schöns Wuchenänd",
    "ein schönes wochenende": "Es schöns Wuchenänd", "bis später": "Bis spöter", "bis morgen": "Bis morn",
    "bis bald": "Bis gli", "entschuldigung": "Exgüsi", "gute besserung": "Gueti Besserig", "gute reise": "Gueti Reis",
    "kein problem": "Kei Problem", "sonst noch etwas": "Suscht no öppis", "was darf es sein": "Was dörfs sii",
    "kommt sofort": "Chunnt sofort", "guten appetit": "En Guete", "gern": "Gärn", "gerne": "Gärn", "nein": "Nei",
    "gut": "Guet", "sehr gut": "Sehr guet", "das klingt gut": "Das tönt guet", "danke das klingt gut": "Merci, das tönt guet",
    # stock phrases of the scenes the translator got wrong every time
    "haben sie ein halbtax": "Händ Sie es Halbtax", "haben sie eine cumulus-karte": "Händ Sie e Cumulus-Charte",
    "brauchen sie eine tüte": "Bruuched Sie es Säckli", "einfach oder retour": "Eifach oder retour",
}
_SENTENCE = re.compile(r"[^.!?…]+[.!?…]*")


def _formula(sentence: str) -> str | None:
    key = re.sub(r"[^\w -]", "", sentence.lower().replace("ß", "ss")).split()
    found = FORMULAS.get(" ".join(key))
    if found is None:
        return None
    end = re.search(r"[.!?…]+$", sentence.strip())
    return found + (end.group(0) if end else "")


def to_zurich(client: ApertusClient, text: str) -> str:
    """Standard German -> Zurich German. The translator gets the whole line (alone, a short fragment lost its
    context: «Kaffee» became «Chüffi»); then each sentence that is a formula of FORMULAS is replaced by its
    checked version. If the sentences do not line up, formulas are kept out and the rest is translated in pieces.
    "" if a call fails."""
    sentences = [p.strip() for p in _SENTENCE.findall(text) if p.strip()]
    fixed = [_formula(x) for x in sentences]
    if all(f is not None for f in fixed):
        return " ".join(fixed)
    if any(f is not None for f in fixed):
        whole = translate_to_zurich(client, text)
        if not whole:
            return ""
        parts = [p.strip() for p in _SENTENCE.findall(whole) if p.strip()]
        if len(parts) == len(sentences):
            return _fix_farewell(text, " ".join(f or q for f, q in zip(fixed, parts)))
        out, chunk = [], []
        for sentence, f in zip(sentences + [""], fixed + ["."]):  # a sentinel flushes the last piece
            if f is None:
                chunk.append(sentence)
                continue
            if chunk:
                done = translate_to_zurich(client, " ".join(chunk))
                if not done:
                    return ""
                out.append(done)
                chunk = []
            out.append(f if sentence else "")
        return _fix_farewell(text, " ".join(x for x in out if x))
    return _fix_farewell(text, translate_to_zurich(client, text))


def _fix_farewell(text: str, result: str) -> str:
    """«Auf Wiedersehen» inside a longer sentence: replace the invented forms seen so far."""
    if result and "wiedersehen" in text.lower():
        result = re.sub(r"(^|[.!?…] )?\b[Uu]fd? (?:Wi\w*|Us\w*)",
                        lambda m: (m.group(1) or "") + ("Uf" if m.group(1) is not None or m.start() == 0 else "uf")
                        + " Widerluege", result)
    return result


def translate_to_zurich(client: ApertusClient, text: str) -> str:
    """Standard German -> Zurich German, by a fine-tuned translator (client.fine_tuned, its training prompt) or by
    Apertus with the E3 prompt and style guide; "" if it fails."""
    if getattr(client, "fine_tuned", False):
        messages = [{"role": "user", "content": FINETUNED_TRANSLATE.format(de=text)}]
    else:
        messages = [{"role": "system", "content": TRANSLATE_TO_ZURICH + "\n\n" + ZURICH_STYLE_GUIDE},
                    {"role": "user", "content": text}]
    try:
        reply = client.chat(messages, temperature=0.0, max_tokens=200)
    except LLMError as e:
        log.warning("Zurich German translation failed: %s", e)
        return ""
    return next((ln for ln in reply.text.splitlines() if ln.strip()), "").strip().strip('"„“«»').strip()


def local_version(client: ApertusClient, learner_text: str) -> str:
    """The learner's sentence the way a local would say it; "" if it fails or is just a copy."""
    return clean_local(to_zurich(client, learner_text), learner_text)


def clean_local(text: str, learner_text: str) -> str:
    line = next((ln for ln in text.splitlines() if ln.strip()), "").strip().strip('"„“«»').strip()
    if changed_words(line, learner_text) <= 1:
        return ""  # the learner's sentence copied, at most a word changed: not a Zurich version
    return line


def changed_words(a: str, b: str) -> int:
    """Words of the longer sentence not matched in the other. Counted in words, not letters: a sentence full of
    names (Tram 3, Kunsthaus) is over 90 % similar letter by letter even with four words in dialect."""
    wa, wb = [_norm(w) for w in a.split() if _norm(w)], [_norm(w) for w in b.split() if _norm(w)]
    same = sum(block.size for block in SequenceMatcher(None, wa, wb).get_matching_blocks())
    return max(len(wa), len(wb)) - same


def _norm(t: str) -> str:
    return "".join(ch for ch in t.lower().replace("ß", "ss") if ch.isalnum())


@dataclass
class Session:
    scenario_id: str
    level: str = "B1"
    help_lang: str = "en"
    dialogue: str = field(default_factory=lambda: settings.dialogue)  # gsw | de, see config.py
    history: list[dict] = field(default_factory=list)   # chat messages without system prompt
    turns: list[Turn] = field(default_factory=list)

    @property
    def scenario(self) -> Scenario:
        return BY_ID[self.scenario_id]

    @property
    def done(self) -> bool:
        return bool(self.turns) and self.turns[-1].scene_done

    def _messages(self, new_user: str) -> list[dict]:
        hist = self.history[-MAX_HISTORY:]
        if hist and hist[0]["role"] == "assistant":
            hist = hist[1:]
        return ([{"role": "system", "content": system_prompt(self.scenario, self.level, self.dialogue)}]
                + hist + [{"role": "user", "content": new_user}])

    def _run(self, client: ApertusClient, user_msg: str, learner_text: str,
             translator: ApertusClient | None = None) -> Turn:
        data, reply = client.chat_json(self._messages(user_msg))
        turn = to_turn(data, reply.text, learner_text, reply.seconds)
        if not turn.reply_gsw:  # DIALOGUE=de: the Standard German line, translated (kept as is if that fails)
            turn.reply_gsw = to_zurich(translator or client, turn.reply_de) or turn.reply_de
        learner_turns = sum(1 for t in self.turns if t.learner_text) + (1 if learner_text else 0)
        if learner_turns >= MAX_TURNS:
            turn.scene_done = True
        self.history += [{"role": "user", "content": learner_text or "(Die Szene beginnt.)"},
                         {"role": "assistant", "content": self._said(turn)}]
        self.turns.append(turn)
        return turn

    def _said(self, turn: Turn) -> str:
        """The local's line as the model wrote it: in its history it sees its own Standard German in DIALOGUE=de."""
        return turn.reply_de if self.dialogue == "de" and turn.reply_de else turn.reply_gsw

    def start(self, client: ApertusClient, translator: ApertusClient | None = None) -> Turn:
        self.history, self.turns = [], []
        return self._run(client, opening_message(self.level, self.dialogue), "", translator)

    def open_with(self, opening: dict) -> Turn:
        """Start from a line generated in advance (data/openings.json): no waiting at the start of a scene."""
        lang = self.help_lang
        words = opening.get("words", {})
        turn = Turn(reply_gsw=opening["gsw"], reply_de=opening.get("de", ""),
                    reply_help=opening.get("help", {}).get(lang, "") if lang != "de" else "",
                    words=list(words.get(lang, words.get("en", []))), suggestions_de=list(opening.get("ideas", [])))
        self.history = [{"role": "user", "content": "(Die Szene beginnt.)"}, {"role": "assistant", "content": self._said(turn)}]
        self.turns = [turn]
        return turn

    def check_answer(self, client: ApertusClient, said: Turn | None, learner_text: str) -> tuple[bool | None, str, str]:
        """Did the learner understand `said`, and the learner's sentence in Zurich German (one call after the
        other): (understood, note, local version)."""
        understood, note = check_comprehension(client, said, learner_text, self.help_lang) if said else (None, "")
        return understood, note, local_version(client, learner_text)

    def reply(self, client: ApertusClient, learner_text: str, translator: ApertusClient | None = None) -> Turn:
        """The local's next line (the role-play call; with DIALOGUE=de then the translation)."""
        sc = self.scenario
        message = learner_message(learner_text, self.level, sc.role, sc.address, self.dialogue)
        return self._run(client, message, learner_text, translator)

    def respond(self, client: ApertusClient, learner_text: str) -> Turn:
        """check_answer() and reply() together; the app runs them itself to show the feedback first."""
        learner_text = learner_text.strip()
        if not learner_text:
            raise LLMError("empty learner input")
        # The checks run while the role-play turn is generated: never more than two requests at once
        # (Featherless allows two).
        with ThreadPoolExecutor(max_workers=1) as pool:
            checks = pool.submit(self.check_answer, client, self.turns[-1] if self.turns else None, learner_text)
            turn = self.reply(client, learner_text)
            turn.understood, turn.understood_note, turn.local_gsw = checks.result()
        return turn

    def words_seen(self) -> list[dict]:
        seen, out = set(), []
        for t in self.turns:
            for w in t.words:
                key = w["gsw"].lower()
                if key not in seen:
                    seen.add(key)
                    out.append(w)
        return out

    def to_dict(self) -> dict:
        return asdict(self)
