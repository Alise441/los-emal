"""Gradio web app.  Run:  python -m losemal.app   (or `make run` for Docker)."""
from __future__ import annotations

import argparse
import html
import logging
import threading
import time

import gradio as gr

from .asr import ASR
from .config import settings
from .llm import ApertusClient, LLMError
from .scenarios import BY_ID, SCENARIOS, label
from .tts import TTS
from .tutor import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("losemal")

asr = ASR()
tts = TTS()
_client: ApertusClient | None = None


def client() -> ApertusClient:
    global _client
    if _client is None:
        _client = ApertusClient(fallback=fallback_model())
    return _client


def fallback_model() -> str:
    """LLM_FALLBACK_NAME, or on Public AI, whose shared 70B quota ran out for hours during our tests, 8B."""
    if settings.llm_fallback_name:
        return settings.llm_fallback_name
    if "publicai" in settings.llm_base_url and settings.llm_name.lower().endswith("70b"):
        return "swiss-ai/apertus-v1.5-8b"
    return ""


def preload() -> None:
    """Load speech models in the background so the first turn is not slow."""
    for part in (tts, asr):
        try:
            part.load()
        except Exception:  # noqa: BLE001 - errors are kept in part.error and shown in the UI
            pass


# ---------------------------------------------------------------- rendering

T = {  # all UI texts; the language radio switches them, and Apertus explains in the same language
    "en": {"subtitle": "Understand Zurich German. Locals speak <i>Züritüütsch</i>, you answer in Standard German "
                       "— powered by Apertus.",
           "intro": "Choose a situation and press <b>Start the scene</b>. Someone from Zurich will talk to you in "
                    "<i>Züritüütsch</i>. Listen, then answer in Standard German — by voice or by typing.",
           "situation": "Situation", "level": "Your German level", "lang": "Language",
           "listen_first": "Listen first (hide the text until I ask)", "start": "▶ Start the scene",
           "again": "🔁 Again", "slower": "🐢 Slower", "show": "👀 Show text", "translate": "🌍 Translate",
           "hint": "💡 Help me answer", "audio": "Listen", "mic": "Answer in Standard German 🎙️",
           "placeholder": "…or type your answer and press Enter", "send": "Send", "conversation": "Whole conversation",
           "listen": "🎧 Listen first. Press <b>Show text</b> if you need it.", "said": "You said",
           "understood": "You understood correctly.", "misunderstood": "Looks like a misunderstanding",
           "correct": "Your German is correct.", "better": "Better", "local": "In Zurich you'd hear",
           "exp": "experimental", "ideas": "You could answer", "done": "Scene complete — well done! Pick another one.",
           "words": "Zurich words from this scene", "nowords": "Words will appear here as you play.",
           "goal": "Your goal"},
    "de": {"subtitle": "Zürichdeutsch verstehen. Einheimische sprechen <i>Züritüütsch</i>, du antwortest auf "
                       "Hochdeutsch — mit Apertus.",
           "intro": "Wähl eine Situation und drück <b>Szene starten</b>. Jemand aus Zürich spricht mit dir "
                    "<i>Züritüütsch</i>. Hör zu und antworte auf Hochdeutsch — per Stimme oder getippt.",
           "situation": "Situation", "level": "Dein Deutschniveau", "lang": "Sprache",
           "listen_first": "Zuerst zuhören (Text erst auf Wunsch zeigen)", "start": "▶ Szene starten",
           "again": "🔁 Nochmal", "slower": "🐢 Langsamer", "show": "👀 Text zeigen", "translate": "🌍 Übersetzen",
           "hint": "💡 Hilf mir antworten", "audio": "Hören", "mic": "Antworte auf Hochdeutsch 🎙️",
           "placeholder": "…oder tipp deine Antwort und drück Enter", "send": "Senden", "conversation": "Ganzes Gespräch",
           "listen": "🎧 Zuerst zuhören. Drück <b>Text zeigen</b>, wenn du den Text brauchst.", "said": "Du hast gesagt",
           "understood": "Richtig verstanden.", "misunderstood": "Da gab es wohl ein Missverständnis",
           "correct": "Dein Deutsch ist korrekt.", "better": "Besser", "local": "In Zürich hörst du",
           "exp": "experimentell", "ideas": "Du könntest antworten", "done": "Szene geschafft — super! Wähl die nächste.",
           "words": "Zürcher Wörter aus dieser Szene", "nowords": "Hier erscheinen die Wörter aus dem Gespräch.",
           "goal": "Dein Ziel"},
    "ru": {"subtitle": "Понимай цюрихский немецкий. Местные говорят на <i>Züritüütsch</i>, ты отвечаешь на "
                       "литературном немецком — с Apertus.",
           "intro": "Выбери ситуацию и нажми <b>Начать сцену</b>. Житель Цюриха заговорит с тобой на "
                    "<i>Züritüütsch</i>. Послушай и ответь на литературном немецком — голосом или текстом.",
           "situation": "Ситуация", "level": "Твой уровень немецкого", "lang": "Язык",
           "listen_first": "Сначала слушать (текст — только по запросу)", "start": "▶ Начать сцену",
           "again": "🔁 Ещё раз", "slower": "🐢 Медленнее", "show": "👀 Показать текст", "translate": "🌍 Перевести",
           "hint": "💡 Помоги ответить", "audio": "Слушать", "mic": "Ответь на литературном немецком 🎙️",
           "placeholder": "…или напечатай ответ и нажми Enter", "send": "Отправить", "conversation": "Весь разговор",
           "listen": "🎧 Сначала послушай. Нажми <b>Показать текст</b>, если нужен текст.", "said": "Твой ответ",
           "understood": "Всё понято верно.", "misunderstood": "Похоже, тут недопонимание",
           "correct": "Твой немецкий правильный.", "better": "Лучше так", "local": "В Цюрихе скажут",
           "exp": "экспериментально", "ideas": "Можно ответить так", "done": "Сцена пройдена — отлично! Выбери следующую.",
           "words": "Цюрихские слова из этой сцены", "nowords": "Здесь появятся слова из разговора.",
           "goal": "Твоя цель"},
}


def header_html(lang: str) -> str:
    return f"<div class='le-head'><h1>Los emal<span>!</span></h1><p>{T[lang]['subtitle']}</p></div>"


def e(x: str) -> str:
    return html.escape(x or "")


def goal_md(scenario_id: str, lang: str) -> str:
    s = BY_ID[scenario_id]
    return f"**{T[lang]['goal']}:** {s.goal.get(lang, s.goal['en'])}"


def render_card(st: dict | None, lang: str = "en") -> str:
    if not st:
        return f"<div class='le-card le-empty'><div class='le-big'>Grüezi! 👋</div><p>{T[lang]['intro']}</p></div>"
    sess: Session = st["session"]
    turn = sess.turns[-1]
    lang = sess.help_lang
    tx = T[lang]
    sc = sess.scenario
    show_text = st["reveal_text"] or not st["listen_first"]
    parts = [f"<div class='le-card'><div class='le-role'>{sc.emoji} {e(sc.role)}</div>"]
    if show_text:
        parts.append(f"<div class='le-line'>{e(turn.reply_gsw)}</div>")
    else:
        parts.append(f"<div class='le-line le-masked'>{tx['listen']}</div>")
    if st["reveal_tr"]:
        if turn.reply_de:
            parts.append(f"<div class='le-tr'><span>DE</span>{e(turn.reply_de)}</div>")
        if turn.reply_help and lang != "de":
            parts.append(f"<div class='le-tr'><span>{lang.upper()}</span>{e(turn.reply_help)}</div>")
    if (show_text or st["reveal_tr"]) and turn.words:
        chips = "".join(
            f"<span class='le-chip'><b>{e(w['gsw'])}</b> = {e(w['de'])}"
            + (f"<i> · {e(w['hint'])}</i>" if w.get("hint") else "") + "</span>" for w in turn.words)
        parts.append(f"<div class='le-chips'>{chips}</div>")
    if st["reveal_hint"] and turn.suggestions_de:
        items = "".join(f"<li>{e(x)}</li>" for x in turn.suggestions_de)
        parts.append(f"<div class='le-hint'><b>💡 {tx['ideas']}:</b><ul>{items}</ul></div>")
    if turn.scene_done:
        parts.append(f"<div class='le-done'>🎉 {tx['done']}</div>")
    if not turn.parse_ok:
        parts.append("<div class='le-warn'>Apertus answered in an unexpected format; showing the raw reply.</div>")
    parts.append("</div>")
    return "".join(parts)


def render_feedback(st: dict | None) -> str:
    if not st:
        return ""
    sess: Session = st["session"]
    turn = sess.turns[-1]
    if not turn.learner_text:
        return ""
    tx = T[sess.help_lang]
    rows = [f"<div class='le-fb'><div class='le-said'>🗣️ {tx['said']}: «{e(turn.learner_text)}»</div>"]
    if turn.understood is False:
        rows.append(f"<div class='le-row le-bad'>❓ {tx['misunderstood']}: {e(turn.understood_note)}</div>")
    elif turn.understood is True:
        rows.append(f"<div class='le-row le-good'>👂 {tx['understood']}</div>")
    fb = turn.feedback
    if fb:
        if fb["ok"]:
            rows.append(f"<div class='le-row le-good'>✅ {tx['correct']}</div>")
        else:
            rows.append(f"<div class='le-row le-fix'>✏️ {tx['better']}: «{e(fb['corrected_de'])}»"
                        + (f"<div class='le-tip'>{e(fb['tip'])}</div>" if fb.get("tip") else "") + "</div>")
    if turn.local_gsw:
        rows.append(f"<div class='le-row le-local'>🏔️ {tx['local']}: «{e(turn.local_gsw)}» "
                    f"<span class='le-tag'>{tx['exp']}</span></div>")
    rows.append("</div>")
    return "".join(rows)


def render_words(st: dict | None, lang: str = "en") -> str:
    if not st:
        return f"<div class='le-words'><div class='le-wtitle'>{T[lang]['words']}</div><p>{T[lang]['nowords']}</p></div>"
    sess: Session = st["session"]
    tx = T[sess.help_lang]
    words = sess.words_seen()
    if not words:
        return f"<div class='le-words'><div class='le-wtitle'>{tx['words']}</div><p>{tx['nowords']}</p></div>"
    items = "".join(f"<li><b>{e(w['gsw'])}</b> — {e(w['de'])}</li>" for w in words)
    return f"<div class='le-words'><div class='le-wtitle'>{tx['words']}</div><ul>{items}</ul></div>"


def chat_messages(st: dict | None) -> list[dict]:
    if not st:
        return []
    msgs = []
    for t in st["session"].turns:
        if t.learner_text:
            msgs.append({"role": "user", "content": t.learner_text})
        msgs.append({"role": "assistant", "content": t.reply_gsw})
    return msgs


def status_md() -> str:
    def part(name, obj):
        if obj.backend == "none":
            return f"{name}: off"
        if obj.error:
            return f"{name}: ⚠️ unavailable"
        return f"{name}: {'ready' if obj._model is not None else 'loading…'}"
    model = _client.model if _client else settings.llm_name  # shows a switch to the fallback model
    return f"<small>Apertus: <code>{e(model)}</code> · {part('ASR', asr)} · {part('TTS', tts)}</small>"


def speak(st: dict, slow: bool = False):
    sess: Session = st["session"]
    text = sess.turns[-1].reply_gsw
    if not tts.enabled:
        return None
    try:
        t0 = time.time()
        sr, wav = tts.synth(text, voice=sess.scenario.voice, slow=slow)
        log.info("TTS %.1f s for %.1f s of audio%s", time.time() - t0, len(wav) / sr, " (slower)" if slow else "")
        return sr, wav
    except Exception as ex:  # noqa: BLE001
        log.exception("TTS failed")
        gr.Warning(f"Speech synthesis failed: {ex}")
        return None


# ---------------------------------------------------------------- events

def start_scene(scenario_id, level, help_lang, listen_first):
    sess = Session(scenario_id=scenario_id, level=level, help_lang=help_lang)
    try:
        turn = sess.start(client())
    except LLMError as ex:
        gr.Warning(f"Apertus is not reachable: {ex}")
        return (gr.skip(),) * 6 + (status_md(),)
    log.info("Apertus %.1f s (scene start, parse_ok=%s)", turn.seconds, turn.parse_ok)
    st = {"session": sess, "listen_first": bool(listen_first),
          "reveal_text": False, "reveal_tr": False, "reveal_hint": False}
    return (st, render_card(st), speak(st), render_feedback(st), chat_messages(st), render_words(st),
            status_md())


def _keep(message: str):
    """Show a toast and leave every output untouched (raising gr.Error would badge all outputs)."""
    gr.Warning(message)
    return (gr.skip(),) * 6 + (status_md(), gr.skip(), gr.skip())


def answer(mic_path, typed, st):
    if not st:
        return _keep("Start a scene first.")
    sess: Session = st["session"]
    if sess.done:
        return _keep("This scene is finished — start a new one.")
    text = (typed or "").strip()
    if not text and mic_path:
        try:
            t0 = time.time()
            text = asr.transcribe(mic_path)
            log.info("ASR %.1f s", time.time() - t0)
        except Exception as ex:  # noqa: BLE001
            return _keep(f"Speech recognition failed ({ex}). Please type your answer.")
    if not text:
        return _keep("I could not hear anything — try again or type your answer.")
    try:
        turn = sess.respond(client(), text)
    except LLMError as ex:
        return _keep(f"Apertus is not reachable: {ex}")
    log.info("Apertus %.1f s (parse_ok=%s)", turn.seconds, turn.parse_ok)
    st.update(reveal_text=False, reveal_tr=False, reveal_hint=False)
    return (st, render_card(st), speak(st), render_feedback(st), chat_messages(st), render_words(st),
            status_md(), None, "")


def replay(st, slow=False):
    if not st:
        return None
    return speak(st, slow=slow)


def toggle(st, key):
    if not st:
        return st, render_card(st)
    st[key] = not st[key]
    return st, render_card(st)


# ---------------------------------------------------------------- layout

CSS = """
:root { --le-red: #d52b1e; --le-ink: #1f2328; --le-soft: #f6f4f1; }
.gradio-container { max-width: 1100px !important; margin: auto; }
.le-head h1 { font-size: 2.1rem; margin: 0; letter-spacing: -0.02em; }
.le-head h1 span { color: var(--le-red); }
.le-head p { margin: 4px 0 0; color: #6b6f76; }
.le-card { background: var(--le-soft); border-radius: 18px; padding: 22px 24px; border: 1px solid #ebe7e1; }
.le-empty p { color: #4b4f56; line-height: 1.5; }
.le-big { font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; }
.le-role { font-size: .85rem; text-transform: uppercase; letter-spacing: .06em; color: #8a8f96; margin-bottom: 10px; }
.le-line { font-size: 1.45rem; line-height: 1.35; font-weight: 600; color: var(--le-ink); }
.le-masked { font-size: 1.05rem; font-weight: 400; color: #6b6f76; }
.le-tr { margin-top: 10px; color: #3b3f45; display: flex; gap: 10px; align-items: baseline; }
.le-tr span { font-size: .7rem; font-weight: 700; color: #fff; background: #8a8f96; border-radius: 6px; padding: 1px 6px; }
.le-chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.le-chip { background: #fff; border: 1px solid #e3ded7; border-radius: 999px; padding: 4px 12px; font-size: .9rem; }
.le-chip b { color: var(--le-red); }
.le-hint { margin-top: 14px; background: #fff; border-radius: 12px; padding: 10px 14px; }
.le-hint ul { margin: 6px 0 0 18px; }
.le-done { margin-top: 14px; font-weight: 700; color: #1a7f37; }
.le-warn { margin-top: 10px; color: #9a6700; font-size: .85rem; }
.le-fb { border-radius: 16px; border: 1px solid #ebe7e1; padding: 14px 18px; }
.le-said { color: #4b4f56; margin-bottom: 8px; }
.le-row { padding: 6px 0; }
.le-good { color: #1a7f37; } .le-bad { color: #9a6700; } .le-fix { color: var(--le-ink); }
.le-tip { color: #6b6f76; font-size: .92rem; margin-top: 2px; }
.le-local { color: #3b3f45; }
.le-tag { font-size: .7rem; background: #eee; border-radius: 6px; padding: 1px 6px; color: #6b6f76; }
.le-words { background: var(--le-soft); border-radius: 14px; padding: 12px 16px; }
.le-wtitle { font-weight: 700; margin-bottom: 4px; }
.le-words ul { margin: 0 0 0 16px; } .le-words p { color: #8a8f96; margin: 0; font-size: .9rem; }
@media (prefers-color-scheme: dark) {
  :root { --le-ink: #f0f0f0; --le-soft: #24262b; }
  .le-card, .le-fb { border-color: #34373d; } .le-chip, .le-hint { background: #1b1d21; border-color: #34373d; }
  .le-tr { color: #d0d3d8; } .le-local { color: #d0d3d8; } .le-said { color: #b8bcc2; }
}
"""

def switch_language(lang, scenario_id, st):
    """Relabel the whole interface; in a running scene, Apertus also explains in the new language from now on."""
    tx = T[lang]
    if st:
        st["session"].help_lang = lang
    choices = [(label(s, lang), s.id) for s in SCENARIOS]
    return (header_html(lang), gr.update(choices=choices, value=scenario_id, label=tx["situation"]),
            goal_md(scenario_id, lang), gr.update(label=tx["level"]), gr.update(label=tx["lang"]),
            gr.update(label=tx["listen_first"]), gr.update(value=tx["start"]), gr.update(label=tx["audio"]),
            gr.update(value=tx["again"]), gr.update(value=tx["slower"]), gr.update(value=tx["show"]),
            gr.update(value=tx["translate"]), gr.update(value=tx["hint"]), gr.update(label=tx["mic"]),
            gr.update(placeholder=tx["placeholder"]), gr.update(value=tx["send"]), gr.update(label=tx["conversation"]),
            render_card(st, lang), render_feedback(st), render_words(st, lang), st)


def build() -> gr.Blocks:
    tx = T["en"]
    with gr.Blocks(title="Los emal!", analytics_enabled=False) as demo:
        st = gr.State(None)
        header = gr.HTML(header_html("en"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                scenario = gr.Dropdown([(label(s, "en"), s.id) for s in SCENARIOS], value=SCENARIOS[0].id,
                                       label=tx["situation"])
                goal = gr.Markdown(goal_md(SCENARIOS[0].id, "en"))
                level = gr.Radio(["A2", "B1", "B2"], value="B1", label=tx["level"])
                help_lang = gr.Radio([("English", "en"), ("Deutsch", "de"), ("Русский", "ru")], value="en",
                                     label=tx["lang"])
                listen_first = gr.Checkbox(True, label=tx["listen_first"])
                start = gr.Button(tx["start"], variant="primary")
                words = gr.HTML(render_words(None))
                status = gr.HTML(status_md())
            with gr.Column(scale=2):
                feedback = gr.HTML()
                card = gr.HTML(render_card(None))
                voice = gr.Audio(label=tx["audio"], type="numpy", autoplay=True, interactive=False)
                with gr.Row():
                    b_again = gr.Button(tx["again"], size="sm")
                    b_slow = gr.Button(tx["slower"], size="sm")
                    b_text = gr.Button(tx["show"], size="sm")
                    b_tr = gr.Button(tx["translate"], size="sm")
                    b_hint = gr.Button(tx["hint"], size="sm")
                mic = gr.Audio(sources=["microphone"], type="filepath", label=tx["mic"])
                with gr.Row():
                    typed = gr.Textbox(placeholder=tx["placeholder"], show_label=False, scale=5)
                    send = gr.Button(tx["send"], variant="primary", scale=1)
                with gr.Accordion(tx["conversation"], open=False) as conversation:
                    chat = gr.Chatbot(height=320, show_label=False)

        scenario.change(goal_md, [scenario, help_lang], goal)
        help_lang.change(switch_language, [help_lang, scenario, st],
                         [header, scenario, goal, level, help_lang, listen_first, start, voice, b_again, b_slow, b_text,
                          b_tr, b_hint, mic, typed, send, conversation, card, feedback, words, st])

        outs = [st, card, voice, feedback, chat, words, status]
        start.click(start_scene, [scenario, level, help_lang, listen_first], outs)
        answer_outs = outs + [mic, typed]
        send.click(answer, [mic, typed, st], answer_outs)
        typed.submit(answer, [mic, typed, st], answer_outs)
        mic.stop_recording(answer, [mic, typed, st], answer_outs)
        b_again.click(lambda s: replay(s, False), st, voice)
        b_slow.click(lambda s: replay(s, True), st, voice)
        b_text.click(lambda s: toggle(s, "reveal_text"), st, [st, card])
        b_tr.click(lambda s: toggle(s, "reveal_tr"), st, [st, card])
        b_hint.click(lambda s: toggle(s, "reveal_hint"), st, [st, card])
        demo.load(status_md, None, status)
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
        server_name=args.host, server_port=args.port, theme=theme, css=CSS, show_error=True)


if __name__ == "__main__":
    main()
