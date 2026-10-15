"""Conversation logic: per learner turn, the role-play turn and, in parallel, three short Apertus calls one
after the other (checks of the learner's Standard German and comprehension, then the learner's sentence in
Zurich German), combined into a validated Turn."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from .llm import ApertusClient, LLMError, extract_json
from .prompts import (TRANSLATE_TO_ZURICH, ZURICH_STYLE_GUIDE, comprehension_message, german_check_message,
                      learner_message, opening_message, system_prompt)
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
    feedback: dict | None = None
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
    if not data or not _s(data.get("reply_gsw")):
        # Model ignored the format: still show something usable instead of failing.
        text = raw.strip().strip("{}").strip() or "Exgüsi, chönd Sie das nomal säge?"
        return Turn(reply_gsw=text[:400], learner_text=learner_text, seconds=seconds, parse_ok=False)

    words = []
    for w in data.get("words") or []:
        if isinstance(w, dict) and _s(w.get("gsw")):
            words.append({"gsw": _s(w.get("gsw")), "de": _s(w.get("de")), "hint": _s(w.get("hint"))})

    feedback = parse_feedback(data.get("feedback"), learner_text)
    sugg =[_s(x) for x in (data.get("suggestions_de") or []) if _s(x)][:3]
    return Turn(
        reply_gsw=_s(data.get("reply_gsw")),
        reply_de=_s(data.get("reply_de")),
        reply_help=_s(data.get("reply_help")),
        words=words[:3],
        feedback=feedback,
        suggestions_de=sugg,
        scene_done=bool(_b(data.get("scene_done"))),
        learner_text=learner_text,
        seconds=seconds,
    )


def parse_feedback(fb, learner_text: str) -> dict | None:
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
    """Short, focused check of the learner's Standard German; None if it fails (the turn still goes on)."""
    try:
        reply = client.chat([{"role": "user", "content": german_check_message(learner_text, help_lang)}],
                            temperature=0.0, max_tokens=200)
    except LLMError as e:
        log.warning("Standard German check failed: %s", e)
        return None
    return parse_feedback(extract_json(reply.text), learner_text)


def local_version(client: ApertusClient, learner_text: str) -> str:
    """The learner's sentence the way a local would say it (the E3 prompt with the style guide); "" if it fails."""
    try:
        reply = client.chat([{"role": "system", "content": TRANSLATE_TO_ZURICH + "\n\n" + ZURICH_STYLE_GUIDE},
                             {"role": "user", "content": learner_text}], temperature=0.0, max_tokens=200)
    except LLMError as e:
        log.warning("Zurich version failed: %s", e)
        return ""
    return clean_local(reply.text, learner_text)


def clean_local(text: str, learner_text: str) -> str:
    line = next((ln for ln in text.splitlines() if ln.strip()), "").strip().strip('"„“«»').strip()
    if SequenceMatcher(None, _norm(line), _norm(learner_text)).ratio() >= 0.9:
        return ""  # the learner's sentence copied, at most a word changed: not a Zurich version
    return line


def _norm(t: str) -> str:
    return "".join(ch for ch in t.lower().replace("ß", "ss") if ch.isalnum())


@dataclass
class Session:
    scenario_id: str
    level: str = "B1"
    help_lang: str = "en"
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
        return ([{"role": "system", "content": system_prompt(self.scenario, self.level, self.help_lang)}]
                + hist + [{"role": "user", "content": new_user}])

    def _run(self, client: ApertusClient, user_msg: str, learner_text: str) -> Turn:
        data, reply = client.chat_json(self._messages(user_msg))
        turn = to_turn(data, reply.text, learner_text, reply.seconds)
        said = [t.learner_text for t in self.turns if t.learner_text] + ([learner_text] if learner_text else [])
        turn.suggestions_de = clean_suggestions(turn.suggestions_de, said)
        learner_turns = len(said)
        if learner_turns >= MAX_TURNS:
            turn.scene_done = True
        self.history += [{"role": "user", "content": learner_text or "(Die Szene beginnt.)"},
                         {"role": "assistant", "content": turn.reply_gsw}]
        self.turns.append(turn)
        return turn

    def start(self, client: ApertusClient) -> Turn:
        self.history, self.turns = [], []
        return self._run(client, opening_message(self.level), "")

    def respond(self, client: ApertusClient, learner_text: str) -> Turn:
        learner_text = learner_text.strip()
        if not learner_text:
            raise LLMError("empty learner input")
        # The two checks and the Zurich version run one after the other while the role-play turn is generated:
        # never more than two requests at once (Featherless allows two).
        with ThreadPoolExecutor(max_workers=1) as pool:
            said = self.turns[-1] if self.turns else None
            check = pool.submit(check_german, client, learner_text, self.help_lang)
            heard = pool.submit(check_comprehension, client, said, learner_text, self.help_lang) if said else None
            local = pool.submit(local_version, client, learner_text)
            turn = self._run(client, learner_message(learner_text, self.level, self.scenario.role), learner_text)
            turn.feedback, turn.local_gsw = check.result(), local.result()
            turn.understood, turn.understood_note = heard.result() if heard else (None, "")
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
