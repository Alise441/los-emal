"""Gradio web app.  Run:  python -m losemal.app   (or `make run` for Docker).

A scene is a chat: the local's lines arrive as voice messages (text and translation on request), the
learner replies in German by voice or typing and is told under the reply whether they understood.
Each scene's first line is prepared in advance (data/openings.json), so a scene starts at once. After a
reply, the next line and the comprehension check run in parallel; the local "types" until the line can be
heard, while its translations, words and reply ideas are prepared alongside (⏳ if asked for earlier).
Dialogs are kept while the page is open: going back to a scene continues it.
Open a scene directly with ?scene=<id>, e.g. ?scene=nachbarin."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import html
import json
import logging
import shutil
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import cache
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from .asr import ASR
from .config import DATA_DIR, settings
from .llm import ApertusClient, LLMError
from .prompts import scene_context
from .scenarios import BY_ID, SCENARIOS, Scenario
from .tts import TTS
from .tutor import Session, check_comprehension, hard_words, local_version, suggest_replies, translate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("losemal")

asr = ASR()
tts = TTS()
_client: ApertusClient | None = None
_translator: ApertusClient | None = None
AUDIO_DIR = Path(tempfile.mkdtemp(prefix="losemal-audio-")).resolve()  # synthesised lines, for the browser
atexit.register(shutil.rmtree, AUDIO_DIR, ignore_errors=True)


def client() -> ApertusClient:
    global _client
    if _client is None:
        _client = ApertusClient(fallback=fallback_model())
    return _client


def translator() -> ApertusClient:
    """Who turns Standard German into Zurich German (DIALOGUE=de, "how a local would say it"): a fine-tuned model
    behind GSW_BASE_URL (see E5 in the technical report), otherwise Apertus itself."""
    global _translator
    if _translator is None:
        if settings.gsw_base_url:
            _translator = ApertusClient(model=settings.gsw_model, base_url=settings.gsw_base_url,
                                        api_key=settings.gsw_api_key)
            _translator.fine_tuned = True
        else:
            _translator = client()
    return _translator


def fallback_model() -> str:
    """LLM_FALLBACK_NAME, or on Public AI, whose shared 70B quota ran out for hours during our tests, 8B."""
    if settings.llm_fallback_name:
        return settings.llm_fallback_name
    if "publicai" in settings.llm_base_url and settings.llm_name.lower().endswith("70b"):
        return "swiss-ai/apertus-v1.5-8b"
    return ""


def preload() -> None:
    """Load the speech models and synthesise every scene's first line in the background, so that neither
    the first turn nor the start of a scene waits."""
    for part in (tts, asr):
        try:
            part.load()
        except Exception:  # noqa: BLE001 - errors are kept in part.error and shown in the UI
            pass
    for sc in SCENARIOS:
        if sc.id in openings():
            make_clip(openings()[sc.id]["gsw"], sc.voice, warn=False)


# ---------------------------------------------------------------- texts

T = {  # all UI texts; the language menu switches them, and Apertus explains in the same language.
       # Another language only needs an entry here and in the language maps of prompts.py.
    "en": {"eyebrow": "Everyday conversations", "choose": "Where shall we practise?",
           "intro": "Listen to Zurich German. Reply in German. Choose a moment from everyday life.",
           "note": "📖 Every scene comes with listening, useful words and reply hints.", "going": "In progress",
           "all": "‹ All scenarios", "goal": "Your goal", "restart": "↻ Restart scenario", "zurich": "Zurich German",
           "scene_words": "📖 Scene words", "words_title": "A little Zurich German",
           "words_sub": "Words you'll hear in this scene.", "show": "Show text", "translate": "Translate",
           "coming": "⏳ The translation is on its way…", "notr": "No translation this time.", "you": "You", "checking": "Checking whether you understood…",
           "understood": "You understood the message.", "misunderstood": "Looks like a misunderstanding",
           "local": "In Zurich you'd hear", "exp": "experimental", "listen_reply": "Listen, then reply in German.",
           "rec": "Record", "stop": "Stop", "resume": "Continue",
           "hint": "💡 Need a hint?", "ideas": "You could answer", "noideas": "No ideas for this line.",
           "ideas_wait": "⏳ Preparing ideas…",
           "placeholder": "Type a reply in German…", "send": "Send",
           "done": "🎉 Scene complete — well done!", "another": "Choose another scene",
           "settings": "⚙️ Settings", "settings_title": "Settings", "auto_show": "Show the text of every line right away",
           "wait": "Wait for the answer, then record again.",
           "powered": "Powered by Apertus"},
    "de": {"eyebrow": "Alltagsgespräche", "choose": "Wo üben wir heute?",
           "intro": "Hör Zürichdeutsch. Antworte auf Deutsch. Wähl einen Moment aus dem Alltag.",
           "note": "📖 Jede Szene mit Hörtext, nützlichen Wörtern und Antworttipps.", "going": "Angefangen",
           "all": "‹ Alle Situationen", "goal": "Dein Ziel", "restart": "↻ Szene neu starten", "zurich": "Züritüütsch",
           "scene_words": "📖 Wörter der Szene", "words_title": "Ein bisschen Züritüütsch",
           "words_sub": "Wörter, die du in dieser Szene hörst.", "show": "Text zeigen", "translate": "Übersetzung",
           "coming": "⏳ Die Übersetzung kommt gleich…", "notr": "Diesmal keine Übersetzung.", "you": "Du", "checking": "Ich prüfe, ob du richtig verstanden hast…",
           "understood": "Richtig verstanden.", "misunderstood": "Da gab es wohl ein Missverständnis",
           "local": "In Zürich hörst du", "exp": "experimentell", "listen_reply": "Hör zu und antworte auf Deutsch.",
           "rec": "Aufnehmen", "stop": "Stopp", "resume": "Weiter",
           "hint": "💡 Ein Tipp?", "ideas": "Du könntest antworten", "noideas": "Keine Ideen für diese Äusserung.",
           "ideas_wait": "⏳ Ideen werden vorbereitet…",
           "placeholder": "Antworte auf Deutsch…", "send": "Senden",
           "done": "🎉 Szene geschafft — super!", "another": "Andere Situation wählen",
           "settings": "⚙️ Einstellungen", "settings_title": "Einstellungen", "auto_show": "Text jeder Äusserung gleich zeigen",
           "wait": "Warte auf die Antwort und nimm dann nochmals auf.",
           "powered": "Mit Apertus"},
}
LANGS = [("English", "en"), ("Deutsch", "de")]


# ---------------------------------------------------------------- rendering

def e(x: str) -> str:
    return html.escape(x or "")


@cache
def avatar_svg(scenario_id: str) -> str:
    """The character's drawing, data/avatars/<scene>.svg, to inline; "" if there is none."""
    path = DATA_DIR / "avatars" / f"{scenario_id}.svg"
    return path.read_text(encoding="utf-8") if path.exists() else ""


@cache
def glossary() -> dict:
    """The reviewed vocabulary of each scene (data/glossary.json, made with make_glossary.py)."""
    path = DATA_DIR / "glossary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@cache
def openings() -> dict:
    """Each scene's first line with translations, words and reply ideas (made with make_openings.py)."""
    path = DATA_DIR / "openings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def avatar(sc: Scenario) -> str:
    return avatar_svg(sc.id) or sc.avatar


def gallery_html(lang: str) -> str:
    tx = T[lang]
    return f"<div class='le-catalog'><div class='le-eyebrow'>{tx['eyebrow']}</div><h1>{tx['choose']}</h1><p>{tx['intro']}</p></div>"


def in_progress(state: dict) -> bool:
    """Begun (the learner has answered) and not finished."""
    sess: Session = state["session"]
    return (not state.get("discarded") and not sess.done
            and (bool(state["pending"]) or any(t.learner_text for t in sess.turns)))


def note_html(lang: str, scenes: dict | None = None) -> str:
    """The note under the cards, and a badge on the cards of scenes in progress (by CSS: cards are buttons)."""
    going = [sid for sid, state in (scenes or {}).items() if in_progress(state)]
    badges = (f"<style>{', '.join(f'button.le-card.le-ic-{sid}::after' for sid in going)}"
              f" {{ content: \"{T[lang]['going']}\" !important; }}</style>") if going else ""  # beats Gradio's scoped copy
    return f"<div class='le-note'>{T[lang]['note']}</div>{badges}"


def meta_html(lang: str) -> str:
    """The line above the answer box, and the labels of the recording buttons: Gradio writes those in the
    browser's language (a Russian browser showed «Запись»), the app in its own."""
    tx = T[lang]
    labels = ";".join(f'--le-{k}: "{tx[k]}" !important' for k in ("rec", "stop", "resume"))
    return f"<span class='le-meta-text'>{tx['listen_reply']}</span><style>:root {{ {labels}; }}</style>"


def card_label(sc: Scenario, lang: str) -> str:
    return f"{sc.title[lang]}\n{sc.goal[lang]}"


def heading_html(sc: Scenario, lang: str) -> str:
    return (f"<div class='le-heading'><h1>{e(sc.title[lang])}</h1>"
            f"<p><b>{T[lang]['goal']}:</b> {e(sc.goal[lang])}</p></div>")


def chat_head_html(sc: Scenario, lang: str) -> str:
    return (f"<div class='le-contact'><div class='le-ava le-ava-lg'>{avatar(sc)}</div><div>"
            f"<div class='le-contact-name'>{e(sc.who.get(lang, sc.role))}</div>"
            f"<div class='le-contact-sub'>{T[lang]['zurich']}</div></div></div>")


def words_html(sc: Scenario, lang: str, show: bool = True) -> str:
    """The scene's word list; "" closes the panel (CSS hides it when empty, as for the hint)."""
    if not show:
        return ""
    tx = T[lang]
    rows = "".join(f"<div class='le-wrow'><b>{e(w['gsw'])}</b><span>{e(w['de'])}"
                   + (f" · {e(w['en'])}" if lang == "en" else "") + "</span></div>"
                   for w in glossary().get(sc.id, []))
    return f"<div class='le-words'><h3>{tx['words_title']}</h3><p>{tx['words_sub']}</p>{rows}</div>"


def fmt(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def player_html(clip: dict) -> str:
    """A voice message: play, static waveform (played part darker, with a playhead), time, speed. PLAYER_JS
    plays it in one Audio object outside the page, so re-rendering the chat does not stop it."""
    bars = "".join(f"<span style='--h:{h}%'></span>" for h in clip["bars"])
    return (f"<div class='le-audio' data-src='{clip['url']}' data-dur='{clip['dur']:.2f}'>"
            "<button class='le-play' type='button' aria-label='Play'></button>"
            f"<div class='le-wave'>{bars}<i class='le-head'></i></div>"
            f"<span class='le-time'>0:00 / {fmt(clip['dur'])}</span>"
            "<button class='le-speed' type='button'>1×</button></div>")


def incoming_html(st: dict, i: int) -> str:
    sess: Session = st["session"]
    sc, turn, lang = sess.scenario, sess.turns[i], sess.help_lang
    tx, key = T[lang], f"{st['key']}-{i}"
    clip = st["audio"].get(i)
    player = player_html(clip) if clip else ""
    tr = []
    if turn.reply_de:
        tr.append(f"<div class='le-tr'><span>DE</span>{e(turn.reply_de)}</div>")
    if turn.reply_help and lang != "de":
        tr.append(f"<div class='le-tr'><span>{lang.upper()}</span>{e(turn.reply_help)}</div>")
    if turn.words:
        tr.append("<div class='le-chips'>" + "".join(
            f"<span class='le-chip'><b>{e(w['gsw'])}</b> = {e(w['de'])}"
            + (f"<i> · {e(w['hint'])}</i>" if w.get("hint") else "") + "</span>" for w in turn.words) + "</div>")
    if st["busy"].get(i, set()) & {"translation", "words"}:
        tr.append(f"<div class='le-muted'>{tx['coming']}</div>")
    elif not tr:
        tr.append(f"<div class='le-muted'>{tx['notr']}</div>")
    show = " checked" if st["auto_show"] or not clip else ""
    return (f"<div class='le-msg le-in'><div class='le-ava'>{avatar(sc)}</div><div class='le-body'>"
            f"<div class='le-label'>{e(sc.who.get(lang, sc.role))}</div>{player}<div class='le-actions'>"
            f"<input type='checkbox' class='le-tg' id='{key}-t'{show}><label for='{key}-t'>📄 {tx['show']}</label>"
            f"<input type='checkbox' class='le-tg' id='{key}-r'><label for='{key}-r'>📖 {tx['translate']}</label>"
            f"<div class='le-pane le-pane-t'>{e(turn.reply_gsw)}</div><div class='le-pane le-pane-r'>{''.join(tr)}</div>"
            "</div></div></div>")


def new_feedback() -> dict:
    """What the app finds out about one answer; run() fills it in as the checks come back."""
    return {"checked": False, "understood": None, "note": "", "local": "", "local_done": False}


DONE = {"checked": True, "understood": None, "note": "", "local": "", "local_done": True}


def outgoing_html(text: str, fb: dict, lang: str, clip: dict | None = None) -> str:
    """The learner's answer, typed or as a voice message (with what was heard), and under it whether they
    understood the local and how a local would say it; ⏳ while these are being worked out."""
    tx, rows = T[lang], []
    if text and not fb["checked"]:
        rows.append(f"<div class='le-fbline le-muted'>⏳ {tx['checking']}</div>")
    elif fb["understood"] is False:
        rows.append(f"<div class='le-fbline le-bad'>❓ {tx['misunderstood']}: {e(fb['note'])}</div>")
    elif fb["understood"] is True:
        rows.append(f"<div class='le-fbline le-good'>✓ {tx['understood']}</div>")
    if text and not fb["local_done"]:
        rows.append(f"<div class='le-fbline le-muted'>🏔️ {tx['local']}: ⏳</div>")
    elif fb["local"]:
        rows.append(f"<div class='le-fbline'>🏔️ {tx['local']}: «{e(fb['local'])}» <span class='le-tag'>{tx['exp']}</span></div>")
    if clip:
        body = (f"<div class='le-bubble le-voice'>{player_html(clip)}"
                f"<div class='le-heard'>{f'«{e(text)}»' if text else '⏳'}</div></div>")
    else:
        body = f"<div class='le-bubble'>{e(text)}</div>"
    return f"<div class='le-msg le-out'><div class='le-label'>{tx['you']}</div>{body}{''.join(rows)}</div>"


def typing_html(sc: Scenario, lang: str) -> str:
    return (f"<div class='le-msg le-in'><div class='le-ava'>{avatar(sc)}</div><div class='le-body'>"
            f"<div class='le-label'>{e(sc.who.get(lang, sc.role))}</div>"
            "<div class='le-typing'><i></i><i></i><i></i></div></div></div>")


def log_html(st: dict | None) -> str:
    """The chat so far: the lines already presented (with their audio), a reply being checked, typing dots."""
    if not st:
        return ""
    sess: Session = st["session"]
    lang = sess.help_lang
    parts = []
    for i, turn in enumerate(sess.turns[:st["shown"]]):
        if turn.learner_text:
            parts.append(outgoing_html(turn.learner_text, st["fbs"].get(i, DONE), lang, st["voice"].get(i)))
        parts.append(incoming_html(st, i))
    if st["pending"]:  # an answer whose reply is not there yet
        pending = st["pending"]
        parts.append(outgoing_html(pending["text"], pending["fb"], lang, pending["clip"]))
    if st["typing"]:
        parts.append(typing_html(sess.scenario, lang))
    finished = " le-finished" if sess.done and st["shown"] == len(sess.turns) else ""
    return f"<div class='le-log{finished}' data-count='{st['key']}-{len(parts)}'>{''.join(parts)}</div>"


def hint_html(st: dict | None) -> str:
    """Ideas for the learner's answer when the hint is open; "" closes the panel (CSS hides it when empty:
    Gradio 6 sometimes dropped visibility updates, and a scene opened with the previous scene's hint)."""
    if not st or not st["hint"]:
        return ""
    tx = T[st["session"].help_lang]
    if st and (st["typing"] or "ideas" in st["busy"].get(st["shown"] - 1, set())):
        return f"<div class='le-hint'>{tx['ideas_wait']}</div>"
    ideas = st["session"].turns[-1].suggestions_de if st["session"].turns else []
    if not ideas:
        return f"<div class='le-hint'>{tx['noideas']}</div>"
    return f"<div class='le-hint'><b>{tx['ideas']}:</b><ul>{''.join(f'<li>{e(x)}</li>' for x in ideas)}</ul></div>"


def status_md() -> str:
    def part(name, obj):
        if obj.backend == "none":
            return f"{name}: off"
        if obj.error:
            return f"{name}: ⚠️ unavailable"
        return f"{name}: {'ready' if obj._model is not None else 'loading…'}"
    model = _client.model if _client else settings.llm_name  # shows a switch to the fallback model
    zurich = f"<code>{e(settings.gsw_model)}</code> (local)" if settings.gsw_base_url else "Apertus"
    return (f"<small>Apertus: <code>{e(model)}</code> · lines written in {'Standard German' if settings.dialogue == 'de' else 'Zurich German'}"
            f" · Zurich German by {zurich} · {part('ASR', asr)} · {part('TTS', tts)}</small>")


def waveform(wav: np.ndarray, n: int = 52) -> list[int]:
    """Bar heights (%) of a static waveform."""
    chunks = np.array_split(np.abs(np.asarray(wav, dtype=np.float32)), n)
    peaks = np.array([c.max() if len(c) else 0.0 for c in chunks])
    peaks = peaks / (peaks.max() or 1.0)
    return [int(16 + 84 * p ** 0.7) for p in peaks]


_clips: dict[str, dict] = {}


def cached_clip(text: str, voice: str) -> dict | None:
    return _clips.get(hashlib.sha1(f"{voice}|{text}".encode()).hexdigest())


def make_clip(text: str, voice: str, warn: bool = True) -> dict | None:
    """Synthesise a line into a wav the browser can fetch (once per voice and text); None without speech."""
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()
    if key in _clips or not tts.enabled:
        return _clips.get(key)
    try:
        t0 = time.time()
        sr, wav = tts.synth(text, voice=voice)
        log.info("TTS %.1f s for %.1f s of audio", time.time() - t0, len(wav) / sr)
    except Exception as ex:  # noqa: BLE001
        log.exception("TTS failed")
        if warn:
            gr.Warning(f"Speech synthesis failed: {ex}")
        return None
    path = AUDIO_DIR / f"{key}.wav"
    sf.write(path, wav, sr)
    _clips[key] = {"url": f"/gradio_api/file={path}", "dur": len(wav) / sr, "bars": waveform(wav)}
    return _clips[key]


def voice_clip(path: str) -> dict:
    """The learner's recording as a voice message, copied next to the synthesised lines."""
    dest = AUDIO_DIR / f"you-{hashlib.sha1(Path(path).read_bytes()).hexdigest()}{Path(path).suffix or '.wav'}"
    shutil.copyfile(path, dest)
    try:
        wav, sr = sf.read(dest, dtype="float32", always_2d=True)
        dur, bars = len(wav) / sr, waveform(wav.mean(axis=1))
    except Exception:  # noqa: BLE001 - a format soundfile cannot read: the browser finds the length itself
        dur, bars = 0.0, [50] * 52
    return {"url": f"/gradio_api/file={dest}", "dur": dur, "bars": bars}


# ---------------------------------------------------------------- layout and events

PLAYER_JS = """
<script>
(() => {
  const audio = new Audio();                // plays every voice message, outside the re-rendered chat
  let current = null;
  const rates = {}, checked = new Set();
  let lastCount = null, lastTop = 0, atBottom = true;
  const fmt = s => { s = Math.max(0, Math.floor(s || 0)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };
  function paint() {   // time, speed, and the played part of the waveform up to the playhead
    document.querySelectorAll('.le-audio').forEach(p => {
      const mine = p.dataset.src === current, t = mine ? audio.currentTime : 0;
      const dur = parseFloat(p.dataset.dur) || (mine && isFinite(audio.duration) ? audio.duration : 0);
      const frac = dur ? Math.min(1, t / dur) : 0;
      p.classList.toggle('le-playing', mine && !audio.paused);
      p.classList.toggle('le-started', mine && t > 0);
      const time = fmt(t) + ' / ' + fmt(dur), label = (rates[p.dataset.src] || 1) + '×';
      const tEl = p.querySelector('.le-time'), sEl = p.querySelector('.le-speed');
      if (tEl.textContent !== time) tEl.textContent = time;
      if (sEl.textContent !== label) sEl.textContent = label;
      p.style.setProperty('--progress', (100 * frac) + '%');
      const bars = p.querySelectorAll('.le-wave span'), lit = Math.round(frac * bars.length);
      bars.forEach((b, k) => b.classList.toggle('on', k < lit));
    });
  }
  let raf = null;   // repaint every frame while playing, so the playhead moves smoothly
  function tick() { paint(); raf = audio.paused ? null : requestAnimationFrame(tick); }
  function play(p, at) {
    const src = p.dataset.src;
    if (current !== src) { audio.src = src; current = src; }
    if (at !== undefined) audio.currentTime = at;
    audio.playbackRate = rates[src] || 1;
    audio.play().catch(() => {});
  }
  ['pause', 'loadedmetadata', 'seeked'].forEach(ev => audio.addEventListener(ev, paint));
  audio.addEventListener('play', () => { if (!raf) raf = requestAnimationFrame(tick); });
  audio.addEventListener('ended', () => { audio.currentTime = 0; paint(); });
  document.addEventListener('click', ev => {
    const p = ev.target.closest('.le-audio');
    if (!p) return;
    if (ev.target.closest('.le-play')) {
      if (current === p.dataset.src && !audio.paused) audio.pause(); else play(p);
    } else if (ev.target.closest('.le-speed')) {
      const r = {1: 0.75, 0.75: 0.5, 0.5: 1}[rates[p.dataset.src] || 1];
      rates[p.dataset.src] = r;
      if (current === p.dataset.src) audio.playbackRate = r;
      paint();
    } else if (ev.target.closest('.le-wave')) {
      const box = p.querySelector('.le-wave').getBoundingClientRect();
      play(p, Math.max(0, (ev.clientX - box.left) / box.width) * (parseFloat(p.dataset.dur) || 0));
    }
  });
  document.addEventListener('change', ev => {
    if (ev.target.matches && ev.target.matches('input.le-tg')) {
      (ev.target.checked ? checked.add(ev.target.id) : checked.delete(ev.target.id));
      if (!ev.target.checked) checked.add('-' + ev.target.id);
      else checked.delete('-' + ev.target.id);
    }
  });
  document.addEventListener('scroll', ev => {     // remembered, as every re-render replaces the chat
    const l = ev.target;
    if (l.classList && l.classList.contains('le-log')) {
      lastTop = l.scrollTop; atBottom = l.scrollHeight - l.scrollTop - l.clientHeight < 40;
    }
  }, true);
  function onRender() {
    document.querySelectorAll('input.le-tg').forEach(c => {
      if (checked.has(c.id)) c.checked = true; else if (checked.has('-' + c.id)) c.checked = false;
    });
    const log = document.querySelector('.le-log');
    if (log && !log.dataset.placed) {             // a new message, or the reader was at the end: follow it
      log.dataset.placed = '1';
      if (log.dataset.count !== lastCount || atBottom) log.scrollTop = log.scrollHeight; else log.scrollTop = lastTop;
      lastCount = log.dataset.count;
    }
    paint();
  }
  let timer = null;
  new MutationObserver(ms => {
    if (ms.every(m => m.target.closest && m.target.closest('.le-time, .le-speed'))) return;
    clearTimeout(timer); timer = setTimeout(onRender, 30);
  }).observe(document.documentElement, {childList: true, subtree: true});
})();
</script>
"""

# Gradio follows the system's dark mode; this light design is meant to stay light.
LIGHT = """() => { const light = () => document.body.classList.remove('dark'); light();
  new MutationObserver(light).observe(document.body, {attributes: true, attributeFilter: ['class']}); }"""

CSS = """
:root { --le-red: #df2635; --le-ink: #202321; --le-muted: #727670; --le-line: #e4e5df; --le-soft: #f3f4f1;
  --le-bg: #f8f8f6; --le-reply: #f0f2ee; --le-green: #31785a; }
body, gradio-app, .gradio-container { background: var(--le-bg) !important; }
.gradio-container { max-width: 1240px !important; margin: auto; }
.le-top { align-items: center !important; border-bottom: 1px solid var(--le-line); padding-bottom: 10px; }
.le-logo { font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; color: var(--le-ink); }
.le-logo span { color: var(--le-red); }
button.le-link { background: none !important; border: none !important; box-shadow: none !important;
  color: var(--le-ink) !important; font-weight: 500 !important; padding: 4px 6px !important; min-width: 0 !important;
  width: auto !important; flex: 0 0 auto !important; font-size: .9rem !important; }
button.le-link:hover { color: var(--le-red) !important; }
button.le-back { color: var(--le-muted) !important; align-self: flex-start; }
/* catalog */
.le-catalog { max-width: 680px; margin: 26px 0 10px; }
.le-eyebrow { font-size: .85rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 600; color: #818a77; }
.le-catalog h1 { font-size: 2.3rem; margin: 10px 0; color: var(--le-ink); letter-spacing: -0.02em; }
.le-catalog p { color: var(--le-muted); margin: 0; font-size: 1.12rem; }
.le-grid { gap: 18px !important; }
button.le-card { display: block !important; position: relative; text-align: left !important; white-space: pre-line !important;
  background: #fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 16'%3E%3Cpath d='M2 2l6 6-6 6' fill='none' stroke='%23536b40' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 24px bottom 22px / 8px 13px !important; border: 1px solid var(--le-line) !important; border-radius: 18px !important;
  padding: 92px 24px 22px !important; min-height: 196px; font-weight: 400 !important; color: var(--le-muted) !important;
  font-size: .9rem !important; line-height: 1.45 !important; box-shadow: none !important; }
button.le-card::first-line { font-size: 1.15rem; font-weight: 700; color: var(--le-ink); line-height: 2.1; }
button.le-card::before { position: absolute; top: 24px; left: 24px; width: 52px; height: 52px; border-radius: 14px;
  background: #f1f1e8; display: grid; place-items: center; font-size: 1.6rem; }
button.le-card::after { content: none; position: absolute; top: 26px; right: 22px; font-size: .7rem; font-weight: 600;
  color: #3d5a32; background: #eef3ea; border-radius: 6px; padding: 3px 9px; line-height: 1.35; }  /* badge: note_html */
button.le-card:hover { border-color: #c4c9ba !important; }
.le-note { text-align: center; color: #959c8e; font-size: .88rem; margin-top: 10px; }
/* lesson */
.le-heading h1 { font-size: 2rem; margin: 2px 0 6px; color: var(--le-ink); letter-spacing: -0.02em; }
.le-heading p { color: var(--le-muted); margin: 0; font-size: 1.15rem; }
.le-heading b { color: var(--le-ink); }
.le-scene-head { align-items: flex-start !important; }
button.le-ghost, button.le-pill { white-space: nowrap !important; width: auto !important; flex: 0 0 auto !important; }
button.le-ghost { border: 1px solid var(--le-line) !important; background: #fff !important; color: var(--le-muted) !important;
  border-radius: 9px !important; box-shadow: none !important; font-weight: 500 !important; }
.le-practice { gap: 20px !important; align-items: flex-start !important; }
.le-chat { background: #fff !important; border: 1px solid #dedfd8 !important; border-radius: 18px !important;
  padding: 0 !important; gap: 0 !important; overflow: hidden; }
.le-chat-head { padding: 14px 22px !important; border-bottom: 1px solid var(--le-line); align-items: center !important; }
button.le-pill { background: #eef3ea !important; color: #3d5a32 !important; border: none !important; border-radius: 9px !important;
  box-shadow: none !important; font-weight: 500 !important; }
.le-contact { display: flex; align-items: center; gap: 12px; }
.le-contact-name { font-weight: 650; font-size: 1.05rem; color: var(--le-ink); }
.le-contact-sub { font-size: .86rem; color: var(--le-muted); }
.le-ava { width: 38px; height: 38px; border-radius: 50%; overflow: hidden; flex: 0 0 auto; font-size: 1.6rem;
  display: flex; align-items: center; justify-content: center; background: #f1ede7; }
.le-ava-lg { width: 48px; height: 48px; font-size: 2rem; }
.le-ava svg { width: 100%; height: 100%; display: block; }
.le-log { padding: 22px 24px; display: flex; flex-direction: column; gap: 20px; min-height: 380px; max-height: 560px;
  overflow-y: auto; }
.le-msg { display: flex; gap: 12px; align-items: flex-start; }
.le-in .le-ava { margin-top: 20px; }
.le-body { width: min(580px, 100%); }
.le-label { font-size: .8rem; font-weight: 600; color: #7b8077; margin-bottom: 6px; }
.le-audio { --progress: 0%; display: flex; align-items: center; gap: 13px; border-radius: 13px; padding: 12px 14px;
  background: var(--le-soft); }
.le-play { width: 40px; height: 40px; border-radius: 50%; background: #30372e; flex: 0 0 auto; border: none; padding: 0;
  cursor: pointer; display: grid; place-items: center; }
/* drawn, not a ▶ glyph, so that it sits in the middle (shifted right a little: a triangle looks centred so) */
.le-play::before { content: ""; width: 0; height: 0; margin-left: 3px; border-style: solid; border-width: 7px 0 7px 12px;
  border-color: transparent transparent transparent #fff; }
.le-playing .le-play::before { width: 11px; height: 13px; margin-left: 0; border-width: 0 4px; border-color: #fff;
  box-sizing: border-box; }
.le-wave { flex: 1; height: 32px; display: flex; align-items: center; gap: 3px; cursor: pointer; position: relative; }
.le-wave span { flex: 1 1 0; min-width: 2px; max-width: 4px; border-radius: 4px; height: var(--h); background: #b4bbae; }
.le-wave span.on { background: #30372e; }
.le-head { position: absolute; top: -5px; bottom: -5px; left: var(--progress); width: 2px; margin-left: -1px; border-radius: 2px;
  background: var(--le-red); opacity: 0; pointer-events: none; }
.le-started .le-head { opacity: 1; }
.le-time { font-size: .8rem; color: #737b6b; font-variant-numeric: tabular-nums; white-space: nowrap; }
.le-speed { font-size: .8rem; border: none; border-radius: 7px; background: #e5e8e1; padding: 6px 9px; cursor: pointer; color: var(--le-ink); }
.le-actions { display: flex; flex-wrap: wrap; gap: 4px 18px; margin-top: 8px; padding-left: 4px; }
.le-tg { display: none; }
.le-actions label { font-size: .88rem; color: #4e544b; cursor: pointer; user-select: none; }
.le-actions label:hover { color: var(--le-red); }
.le-tg:checked + label { color: var(--le-ink); font-weight: 600; }
.le-pane { display: none; flex-basis: 100%; font-size: .98rem; line-height: 1.6; color: #3f463b; padding: 10px 14px;
  border-left: 2px solid #d1d9cb; background: #fafbf9; border-radius: 0 8px 8px 0; }
.le-tg:nth-of-type(1):checked ~ .le-pane-t, .le-tg:nth-of-type(2):checked ~ .le-pane-r { display: block; }
.le-pane-t { font-weight: 600; color: var(--le-ink); font-size: 1rem; }
.le-tr { display: flex; gap: 10px; align-items: baseline; margin: 2px 0 6px; }
.le-tr span { font-size: .7rem; font-weight: 700; color: #fff; background: #8a8f96; border-radius: 6px; padding: 1px 6px; }
.le-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.le-chip { background: #fff; border: 1px solid var(--le-line); border-radius: 10px; padding: 4px 11px; font-size: .88rem; }
.le-chip b { color: var(--le-red); }
.le-out { flex-direction: column; align-self: flex-end; width: min(76%, 600px); gap: 0; }
.le-out .le-label { align-self: flex-start; }
.le-bubble { padding: 12px 16px; border-radius: 14px 14px 3px 14px; background: var(--le-reply); font-size: .95rem;
  line-height: 1.6; color: var(--le-ink); width: 100%; box-sizing: border-box; }
.le-voice { padding: 4px 10px 10px; }
.le-voice .le-audio { background: transparent; padding: 8px 4px; }
.le-voice .le-wave span { background: #c3c9bc; } .le-voice .le-wave span.on { background: #30372e; }
.le-heard { font-size: .9rem; color: var(--le-muted); font-style: italic; padding: 0 6px; }
.le-fbline { font-size: .9rem; line-height: 1.5; padding: 7px 2px 0; color: #59724e; }
.le-fbline:first-of-type { border-top: 1px solid #e5e9e0; margin-top: 7px; }
.le-good { color: var(--le-green); } .le-bad { color: #9a6700; } .le-muted { color: var(--le-muted); }
.le-tag { font-size: .72rem; background: #eee; border-radius: 6px; padding: 1px 6px; color: var(--le-muted); }
.le-typing { display: flex; align-items: center; gap: 5px; height: 44px; padding: 0 17px; background: var(--le-soft);
  border-radius: 13px; width: 70px; box-sizing: border-box; }
.le-typing i { width: 6px; height: 6px; border-radius: 50%; background: #9ba592; animation: le-dot 1s infinite ease-in-out; }
.le-typing i:nth-child(2) { animation-delay: .15s; } .le-typing i:nth-child(3) { animation-delay: .3s; }
@keyframes le-dot { 0%, 80%, 100% { opacity: .35; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-4px); } }
.le-composer { border-top: 1px solid var(--le-line); padding: 12px 20px 16px !important; gap: 10px !important; }
.le-meta-text { font-size: .88rem; color: #858b7e; }
.le-hint { background: #f6f7f2; border: 1px solid #e5e8df; border-radius: 10px; padding: 10px 14px; font-size: .92rem; color: #4f5b45; }
.le-hint ul { margin: 4px 0 0 18px; }
.le-hintbox:not(:has(.le-hint)), .le-words-col:not(:has(.le-words)) { display: none !important; }
.le-reply { align-items: flex-end !important; gap: 14px !important; }
.le-textcol { gap: 8px !important; }
.le-sidecol { align-items: center !important; gap: 6px !important; }  /* the hint above the record button */
.le-sidecol button.le-link { align-self: center; }
/* recording: the recorder takes the place of the text box */
.le-reply:has(.le-mic .timestamps) .le-textcol { display: none !important; }
.le-reply:has(.le-mic .timestamps) .le-sidecol { flex: 1 1 auto !important; }
.le-mic .component-wrapper { padding: 0 !important; }  /* so the button lines up with the text box */
.le-mic .component-wrapper:has(.timestamps) { display: flex; align-items: center; gap: 14px; width: 100%; }
.le-mic .component-wrapper:has(.timestamps) > * { flex: 0 0 auto; }
.le-mic .component-wrapper:has(.timestamps) [data-testid=recording-waveform] { display: none; }
.le-mic .component-wrapper:has(.timestamps) .microphone { flex: 1 1 0 !important; min-width: 0; background: var(--le-soft);
  border-radius: 10px; padding: 4px 14px; }
.le-mic .timestamps { order: 2; margin: 0 !important; width: auto !important; font-variant-numeric: tabular-nums;
  color: var(--le-muted); }
.le-mic .timestamps .time { display: none; }
.le-mic .component-wrapper:has(.timestamps) .controls { order: 3; width: auto !important; }
.le-mic .pause-button, .le-mic .resume-button, .le-mic .stop-button-paused { display: none !important; }
.le-reply textarea, .le-reply input { border: 1px solid #d9ded2 !important; border-radius: 10px !important; background: #fdfefc !important;
  padding: 13px 14px !important; min-height: 50px; }
.le-reply:has(textarea:placeholder-shown) button.le-send, .le-reply:has(input:placeholder-shown) button.le-send { display: none !important; }
.le-reply:has(textarea:not(:placeholder-shown)) .le-mic, .le-reply:has(input:not(:placeholder-shown)) .le-mic { display: none !important; }
button.le-send { border-radius: 10px !important; min-height: 50px; }
.le-mic { border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; }
.le-mic .mic-select, .le-mic .device-select-large, .le-mic .device-selection-wrapper,
.le-mic .icon-button-wrapper { display: none !important; }
.le-mic .controls, .le-mic .controls .wrapper { justify-content: center !important; width: 100%; margin: 0 !important; }
.le-mic .record-button, .le-mic .stop-button { background: var(--le-red) !important; color: #fff !important; border: none !important;
  border-radius: 10px !important; min-height: 50px; padding: 0 20px !important; font-size: .95rem !important; font-weight: 600 !important;
  white-space: nowrap; }
.le-mic .record-button, .le-mic .stop-button, .le-mic .stop-button-paused, .le-mic .resume-button {
  font-size: 0 !important; }  /* Gradio's own label; ours below, from meta_html */
.le-mic .record-button::before, .le-mic .stop-button::before, .le-mic .stop-button-paused::before,
.le-mic .resume-button::before { font-size: .95rem; width: auto !important; height: auto !important; margin: 0 !important;
  background: none !important; border: none !important; border-radius: 0 !important; animation: none !important; }
.le-mic .record-button::before { content: "🎙  " var(--le-rec); }
.le-mic .stop-button::before, .le-mic .stop-button-paused::before { content: "■  " var(--le-stop); }
.le-mic .resume-button::before { content: var(--le-resume); }
.le-done { padding: 18px 22px !important; border-top: 1px solid var(--le-line); align-items: center !important; }
.le-chat:not(:has(.le-finished)) .le-done, .le-chat:has(.le-finished) .le-composer { display: none !important; }
.le-done-text { font-weight: 700; color: var(--le-green); }
.le-words-col { background: #fff !important; border: 1px solid var(--le-line) !important; border-radius: 18px !important;
  padding: 18px 20px !important; position: sticky; top: 16px; }
.le-words h3 { margin: 0; font-size: 1.15rem; color: var(--le-ink); }
.le-words p { color: var(--le-muted); font-size: .86rem; margin: 6px 0 8px; }
.le-wrow { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr); gap: 10px; padding: 10px 0;
  border-top: 1px solid var(--le-line); font-size: .88rem; }
.le-wrow b { color: var(--le-ink); font-weight: 600; } .le-wrow span { color: var(--le-muted); text-align: right; }
.le-words-close { align-self: flex-end; }
.le-footer { text-align: center; color: #969c8e; font-size: .86rem; margin-top: 14px; }
/* settings: a window over the page */
.le-modal { position: fixed !important; inset: 0 !important; z-index: 1000 !important; background: rgba(32, 35, 33, .38) !important;
  justify-content: center !important; align-items: center !important; margin: 0 !important; padding: 16px !important;
  max-width: none !important; }
.le-modal-card { background: #fff !important; border-radius: 16px !important; padding: 18px 24px 24px !important;
  width: min(460px, 100%) !important; flex: 0 0 auto !important; box-shadow: 0 20px 60px rgba(0, 0, 0, .2); gap: 14px !important; }
.le-modal-head { align-items: center !important; justify-content: space-between !important; }
.le-modal-head h3 { margin: 0; font-size: 1.25rem; color: var(--le-ink); }
button.le-close { font-size: 1.2rem !important; color: var(--le-muted) !important; }
.le-top input { cursor: pointer !important; caret-color: transparent; }
""" + "".join(f'button.le-ic-{s.id}::before {{ content: "{s.emoji}"; }}\n' for s in SCENARIOS)


def build() -> gr.Blocks:
    en = T["en"]
    with gr.Blocks(title="Los emal!", analytics_enabled=False) as demo:
        st = gr.State(None)       # the scene on screen
        saved = gr.State({})      # every scene opened in this tab, {scenario id: state}: back to where you left it
        settings_open = gr.State(False)

        with gr.Row(elem_classes="le-top"):
            gr.HTML("<div class='le-logo'>Los emal<span>!</span></div>")
            level = gr.Dropdown(["A2", "B1", "B2"], value="B1", show_label=False, container=False, scale=0,
                                min_width=90, filterable=False)
            lang = gr.Dropdown(LANGS, value="en", show_label=False, container=False, scale=0, min_width=130,
                               filterable=False)
            b_settings = gr.Button(en["settings"], scale=0, min_width=150, elem_classes="le-link")
        with gr.Column(visible=False, elem_classes="le-modal") as settings_panel:  # a window over the page
            with gr.Column(elem_classes="le-modal-card"):
                with gr.Row(elem_classes="le-modal-head"):
                    settings_title = gr.HTML(f"<h3>{en['settings_title']}</h3>")
                    b_close = gr.Button("✕", scale=0, min_width=40, elem_classes="le-link le-close")
                auto_show = gr.Checkbox(False, label=en["auto_show"])
                status = gr.HTML(status_md())

        with gr.Column() as gallery:
            gallery_head = gr.HTML(gallery_html("en"))
            cards = []
            for i in range(0, len(SCENARIOS), 3):
                with gr.Row(elem_classes="le-grid", equal_height=True):
                    for sc in SCENARIOS[i:i + 3]:
                        cards.append(gr.Button(card_label(sc, "en"), elem_classes=["le-card", f"le-ic-{sc.id}"]))
                    for _ in range(3 - len(SCENARIOS[i:i + 3])):
                        gr.HTML("")
            gallery_note = gr.HTML(note_html("en"))

        with gr.Column(visible=False) as lesson:
            b_back = gr.Button(en["all"], size="sm", elem_classes="le-link le-back")
            with gr.Row(elem_classes="le-scene-head"):
                heading = gr.HTML()
                b_restart = gr.Button(en["restart"], scale=0, min_width=180, elem_classes="le-ghost")
            with gr.Row(elem_classes="le-practice", equal_height=False):
                with gr.Column(scale=3, min_width=480, elem_classes="le-chat"):
                    with gr.Row(elem_classes="le-chat-head"):
                        chat_head = gr.HTML()
                        b_words = gr.Button(en["scene_words"], scale=0, min_width=160, elem_classes="le-pill")
                    chat = gr.HTML()
                    with gr.Column(elem_classes="le-composer"):
                        hint = gr.HTML("", elem_classes="le-hintbox")
                        with gr.Row(elem_classes="le-reply", equal_height=False):
                            with gr.Column(scale=5, elem_classes="le-textcol"):
                                meta = gr.HTML(meta_html("en"))
                                typed = gr.Textbox(placeholder=en["placeholder"], show_label=False, container=False,
                                                   lines=1, max_lines=4)
                            with gr.Column(scale=0, min_width=210, elem_classes="le-sidecol"):  # hint over the mic
                                b_hint = gr.Button(en["hint"], elem_classes="le-link")
                                send = gr.Button(en["send"], variant="primary", elem_classes="le-send")
                                mic = gr.Audio(sources=["microphone"], type="filepath", show_label=False, buttons=[],
                                               editable=False, elem_classes="le-mic")
                    with gr.Row(elem_classes="le-done"):  # shown instead of the composer at the end (CSS)
                        done_text = gr.HTML(f"<span class='le-done-text'>{en['done']}</span>")
                        b_another = gr.Button(en["another"], variant="primary", scale=0, min_width=220)
                with gr.Column(scale=1, min_width=250, elem_classes="le-words-col"):
                    words = gr.HTML()
        footer = gr.HTML(f"<div class='le-footer'>{en['powered']}</div>")

        outs = [st, gallery, lesson, heading, chat_head, chat, words, hint, typed, mic, status]

        def keep(*components) -> dict:
            """Leave these outputs as they are (an empty dict would be taken for a single value)."""
            return {c: gr.skip() for c in components}

        def render(state: dict) -> str:
            """The chat as now shown (remembered, so run() only sends real changes: each new chat restarts
            the typing dots)."""
            state["rendered"] = log_html(state)
            return state["rendered"]

        def prepare(pool: ThreadPoolExecutor, state: dict, i: int) -> dict:
            """Start what line i still lacks: its audio, translations, ideas for the answer and hard words;
            returns {job: (kind, i)}. The model calls queue in the client (two at a time) in this order; the
            audio is synthesised locally alongside."""
            sess: Session = state["session"]
            turn, sc, lang_ = sess.turns[i], sess.scenario, sess.help_lang
            jobs = {}
            if i not in state["audio"] and tts.enabled:
                jobs[pool.submit(make_clip, turn.reply_gsw, sc.voice)] = ("audio", i)
            if not turn.reply_de or (lang_ != "de" and not turn.reply_help):
                jobs[pool.submit(translate, client(), turn, lang_, sc)] = ("translation", i)
            if not turn.suggestions_de and not sess.done:
                jobs[pool.submit(suggest_replies, client(), sess)] = ("ideas", i)
            if not turn.words:
                jobs[pool.submit(hard_words, client(), turn.reply_gsw, lang_, scene_context(sc))] = ("words", i)
            state["busy"][i] = {kind for kind, _ in jobs.values()}
            return jobs

        def run(state: dict, steps: dict, text: str = ""):
            """Show each result when it is ready: under the learner's answer, whether they understood ("check")
            and how a local would say it ("local"); the next line ("line") once it can be heard (until then the
            local is typing), its translations, reply ideas and words being prepared meanwhile. A scene that is
            not on screen (the learner went to another one) gets its results all the same, without touching the
            page, so it is complete when the learner comes back. Ends with an update that changes nothing: when
            a generator finishes, Gradio sends its last values once more, which would put back an older chat
            (or another scene's) over the page."""
            sess: Session = state["session"]
            fb = state["pending"]["fb"] if state["pending"] else new_feedback()  # on the answer `text`
            line = None  # index of the new line, once generated
            with ThreadPoolExecutor(max_workers=6) as pool:
                jobs = {pool.submit(fn, *args): (kind, None) for kind, (fn, args) in steps.items()}
                if "line" not in steps:  # a prepared opening: perhaps its audio is still being synthesised
                    line = len(sess.turns) - 1
                    jobs.update(prepare(pool, state, line))
                while jobs:
                    done, _ = wait(list(jobs), return_when=FIRST_COMPLETED)
                    if state.get("discarded"):  # the scene was restarted: its old dialog is gone
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    update = {}
                    for job in done:
                        kind, i = jobs.pop(job)
                        try:
                            value = job.result()
                        except LLMError as ex:
                            if kind != "line":  # the helper calls return ""/[] themselves; just in case
                                log.warning("%s failed: %s", kind, ex)
                                value = None
                            else:
                                state.update(typing=False, pending=None)
                                if not sess.turns:  # a scene that could not even start: start it anew next time
                                    state["discarded"] = True
                                    update.update({gallery: gr.update(visible=True), lesson: gr.update(visible=False)})
                                if state["visible"]:
                                    gr.Warning(f"Apertus is not reachable: {ex}")
                                    yield {**update, chat: render(state), typed: text, status: status_md()}
                                yield keep(st)
                                return
                        except Exception:  # noqa: BLE001 - a missing translation or audio must not end the scene
                            log.exception("%s failed", kind)
                            value = None
                        if kind == "line":
                            log.info("Apertus %.1f s (parse_ok=%s)", value.seconds, value.parse_ok)
                            line = len(sess.turns) - 1
                            if text:  # the answer's feedback (and recording) stay with the line it led to
                                state["fbs"][line] = fb
                                if state["pending"] and state["pending"]["clip"]:
                                    state["voice"][line] = state["pending"]["clip"]
                            jobs.update(prepare(pool, state, line))
                        elif kind == "check":
                            fb.update(understood=value[0] if value else None, note=value[1] if value else "",
                                      checked=True)
                        elif kind == "local":
                            fb.update(local=value or "", local_done=True)
                        else:
                            if kind == "audio" and value:
                                state["audio"][i] = value
                            elif kind == "ideas":
                                sess.turns[i].suggestions_de = value or []
                            elif kind == "words":
                                sess.turns[i].words = value or []
                            state["busy"].get(i, set()).discard(kind)
                    if line is not None and state["shown"] <= line and "audio" not in state["busy"].get(line, ()):
                        state.update(shown=line + 1, typing=False, pending=None)  # it can be heard: no more typing
                        update[status] = status_md()
                    if state["visible"]:
                        if log_html(state) != state.get("rendered"):
                            update[chat] = render(state)
                        if state["hint"]:
                            update[hint] = hint_html(state)
                        if update:
                            yield update
            yield keep(st)

        def page(state: dict, lang_: str) -> dict:
            """The lesson page of a scene, as the state has it."""
            sc = state["session"].scenario
            return {st: state, gallery: gr.update(visible=False), lesson: gr.update(visible=True),
                    heading: heading_html(sc, lang_), chat_head: chat_head_html(sc, lang_), chat: render(state),
                    words: words_html(sc, lang_, state["words_open"]),
                    hint: hint_html(state), typed: "", mic: None, status: status_md()}

        def start(scenario_id: str, level_: str, lang_: str, show_text: bool, scenes: dict):
            """A scene from its beginning; an earlier dialog of it is dropped."""
            if scenario_id in scenes:
                scenes[scenario_id].update(visible=False, discarded=True)
            sc = BY_ID[scenario_id]
            sess = Session(scenario_id=scenario_id, level=level_, help_lang=lang_)
            state = {"session": sess, "audio": {}, "shown": 0, "pending": None, "typing": True, "fbs": {}, "voice": {},
                     "busy": {}, "auto_show": bool(show_text), "hint": False, "words_open": True, "visible": True,
                     "key": f"{scenario_id}{int(time.time() * 1000)}"}
            scenes[scenario_id] = state
            opening = openings().get(scenario_id)
            if opening:  # prepared in advance: no waiting for the model
                sess.open_with(opening)
                clip = cached_clip(opening["gsw"], sc.voice)
                if clip:
                    state["audio"][0] = clip
                if clip or not tts.enabled:  # otherwise the local types until the audio is synthesised
                    state.update(shown=1, typing=False)
            yield page(state, lang_)
            yield from run(state, {} if opening else {"line": (sess.start, (client(), translator()))})

        def opener(scenario_id: str):
            def go(level_, lang_, show_text, current, scenes):
                """Back to the scene where the learner left it (with the settings chosen meanwhile), or start it."""
                if current:
                    current["visible"] = False
                state = scenes.get(scenario_id)
                if not state or state.get("discarded"):
                    yield from start(scenario_id, level_, lang_, show_text, scenes)
                    return
                state["session"].level, state["session"].help_lang = level_, lang_
                state.update(visible=True, auto_show=bool(show_text), hint=False)
                yield page(state, lang_)
                yield keep(st)  # see run()
            return go

        for sc, card in zip(SCENARIOS, cards):
            card.click(opener(sc.id), [level, lang, auto_show, st, saved], outs, show_progress="hidden",
                       trigger_mode="multiple")

        def restart(level_, lang_, show_text, state, scenes):
            if not state:
                yield keep(st)
                return
            yield from start(state["session"].scenario_id, level_, lang_, show_text, scenes)

        b_restart.click(restart, [level, lang, auto_show, st, saved], outs, show_progress="hidden",
                        trigger_mode="multiple")

        def on_load(level_, lang_, show_text, scenes, request: gr.Request):
            scenario_id = request.query_params.get("scene") if request else None
            if scenario_id in BY_ID:
                yield from start(scenario_id, level_, lang_, show_text, scenes)
            else:
                yield {status: status_md()}

        demo.load(on_load, [level, lang, auto_show, saved], outs, show_progress="hidden")

        def to_gallery(state, scenes, lang_):
            if state:  # the dialog stays in `saved`; its background work goes on without touching the page
                state["visible"] = False
            return {gallery: gr.update(visible=True), lesson: gr.update(visible=False),
                    gallery_note: note_html(lang_, scenes)}

        b_back.click(to_gallery, [st, saved, lang], [gallery, lesson, gallery_note], show_progress="hidden")
        b_another.click(to_gallery, [st, saved, lang], [gallery, lesson, gallery_note], show_progress="hidden")

        def answer(text, state):
            text = (text or "").strip()
            if not state or state["typing"]:
                yield keep(st)
                return
            sess: Session = state["session"]
            if sess.done:
                gr.Warning("This scene is finished — choose another one.")
                yield keep(st)
                return
            if not text:
                gr.Warning("Record or type your answer first.")
                yield keep(st)
                return
            state.update(pending={"text": text, "clip": None, "fb": new_feedback()}, typing=True, hint=False)
            yield {st: state, chat: render(state), typed: "", hint: ""}
            yield from run(state, answer_steps(sess, text), text)

        def answer_steps(sess: Session, text: str) -> dict:
            """The next line and, alongside, the two checks of the answer."""
            return {"line": (sess.reply, (client(), text, translator())),
                    "check": (check_comprehension, (client(), sess.turns[-1], text, sess.help_lang)),
                    "local": (local_version, (translator(), text))}

        # "multiple": the previous line's event may still be preparing its translations in the background;
        # with Gradio's default ("once") an answer sent meanwhile was silently dropped.
        send.click(answer, [typed, st], outs, show_progress="hidden", trigger_mode="multiple")
        typed.submit(answer, [typed, st], outs, show_progress="hidden", trigger_mode="multiple")

        def answer_voice(path, state):
            """A recording is sent as a voice message: shown at once, then transcribed and answered like text."""
            if not path:
                yield keep(st)
                return
            if not state or state["typing"] or state["session"].done:
                if state and state["typing"]:
                    gr.Warning(T[state["session"].help_lang]["wait"])
                yield {mic: None}
                return
            sess: Session = state["session"]
            state.update(pending={"text": "", "clip": voice_clip(path), "fb": new_feedback()}, typing=True, hint=False)
            yield {st: state, chat: render(state), mic: None, hint: ""}
            try:
                t0 = time.time()
                text = asr.transcribe(path)
                log.info("ASR %.1f s", time.time() - t0)
            except Exception as ex:  # noqa: BLE001
                gr.Warning(f"Speech recognition failed ({ex}). Please type your answer.")
                text = None
            if not text:
                if text is not None:
                    gr.Warning("I could not hear anything — try again or type your answer.")
                state.update(pending=None, typing=False)
                yield {chat: render(state)}
                yield keep(st)
                return
            state["pending"]["text"] = text
            yield {chat: render(state)}
            yield from run(state, answer_steps(sess, text), text)

        mic.stop_recording(answer_voice, [mic, st], outs, show_progress="hidden", trigger_mode="multiple")

        def toggle_hint(state):
            if not state:
                return keep(st, hint)
            state["hint"] = not state["hint"]
            return {st: state, hint: hint_html(state)}

        b_hint.click(toggle_hint, st, [st, hint], show_progress="hidden")

        def toggle_words(state):
            if not state:
                return keep(st, words)
            state["words_open"] = not state.get("words_open", True)
            sess: Session = state["session"]
            return {st: state, words: words_html(sess.scenario, sess.help_lang, state["words_open"])}

        b_words.click(toggle_words, st, [st, words], show_progress="hidden")

        def toggle_settings(is_open):
            return not is_open, gr.update(visible=not is_open)

        b_settings.click(toggle_settings, settings_open, [settings_open, settings_panel], show_progress="hidden")
        b_close.click(lambda: (False, gr.update(visible=False)), None, [settings_open, settings_panel],
                      show_progress="hidden")

        def set_auto_show(show_text, state):
            if not state:
                return keep(st, chat)
            state["auto_show"] = bool(show_text)
            return {st: state, chat: render(state)}

        auto_show.change(set_auto_show, [auto_show, st], [st, chat], show_progress="hidden")

        def set_level(level_, state):
            if state:  # applies from the next line on
                state["session"].level = level_
            return state

        level.change(set_level, [level, st], st, show_progress="hidden")

        def switch_language(lang_, state, scenes):
            """Relabel the whole interface; in a running scene, Apertus also explains in the new language."""
            tx = T[lang_]
            updates = {gallery_head: gallery_html(lang_), gallery_note: note_html(lang_, scenes),
                       b_settings: tx["settings"], settings_title: f"<h3>{tx['settings_title']}</h3>",
                       auto_show: gr.update(label=tx["auto_show"]), b_back: tx["all"],
                       b_restart: tx["restart"], b_words: tx["scene_words"], b_hint: tx["hint"],
                       meta: meta_html(lang_),
                       typed: gr.update(placeholder=tx["placeholder"]), send: tx["send"],
                       done_text: f"<span class='le-done-text'>{tx['done']}</span>", b_another: tx["another"],
                       footer: f"<div class='le-footer'>{tx['powered']}</div>"}
            updates.update({card: card_label(sc, lang_) for sc, card in zip(SCENARIOS, cards)})
            if not state:
                return {**updates, **keep(st, heading, chat_head, chat, words, hint)}
            sess: Session = state["session"]
            sess.help_lang = lang_
            return {**updates, st: state, heading: heading_html(sess.scenario, lang_),
                    chat_head: chat_head_html(sess.scenario, lang_), chat: render(state),
                    words: words_html(sess.scenario, lang_, state["words_open"]), hint: hint_html(state)}

        lang.change(switch_language, [lang, st, saved],
                    [gallery_head, gallery_note, b_settings, settings_title, auto_show, b_back, b_restart, b_words,
                     b_hint, meta, typed,
                     send, done_text, b_another, footer, *cards, st, heading, chat_head, chat, words, hint],
                    show_progress="hidden")
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=settings.host)
    ap.add_argument("--port", type=int, default=settings.port)
    ap.add_argument("--no-preload", action="store_true")
    args = ap.parse_args()
    if not args.no_preload:
        threading.Thread(target=preload, daemon=True).start()
    theme = gr.themes.Soft(primary_hue="red", neutral_hue="stone")
    build().queue(default_concurrency_limit=4).launch(
        server_name=args.host, server_port=args.port, theme=theme, css=CSS, js=LIGHT, head=PLAYER_JS,
        allowed_paths=[str(AUDIO_DIR)], show_error=True, footer_links=[])


if __name__ == "__main__":
    main()
