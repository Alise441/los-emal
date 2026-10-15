"""Prompts. The Zurich style guide is shared with the evaluation (src/eval/e3_generation.py)."""
from __future__ import annotations

from .scenarios import Scenario

# Stadtzürcher forms vs. forms from other regions that Apertus tends to mix in
# (observed in our pilot: het, dr, gsy, nid ...). Kept short on purpose.
ZURICH_STYLE_GUIDE = """\
Regeln für Stadtzürcher Dialekt (Züritüütsch), geschrieben in üblicher Dialektschreibung:
- Verwende Zürcher Formen, keine Berner oder Basler Formen:
  hät / händ (nicht het, hei), nöd (nicht nid, nit, ned), gsi (nicht gsy), und (nicht u), ich (nicht ig),
  Artikel de / d / s und en / e / es (nicht dr), Mal, vil, Wält (nicht Mau, viu, Wäut).
- Häufige Formen: ich bi, du bisch, er/si isch, mer sind; ich han, du häsch, er hät, mer händ;
  ich chan, Sie chönd; ich wott, Sie wänd; mer gönd, gömmer; öppis, nüt, chli, hüt, geschter, morn,
  au, scho, gärn, jetzt, dänn, eifach, vilicht.
- Vergangenheit nur mit Perfekt: «ich bi gsi», «mer händ gmacht». Kein Präteritum.
- Grüezi / Uf Widerluege zu «Sie», Hoi / Sali / Tschüss zu «du». Merci (vilmal), Exgüsi, Gärn gscheh.
- Typische Wörter: Velo, Tram, Billett, Trottoir, poschte, luege, lose, säge, verzelle, schaffe,
  Znüni, Zmittag, Znacht, Gipfeli, Weggli, Rüebli, Poulet, Säckli, Händy.
- Höflichkeitsform: «Chönd Sie …?», «Wänd Sie …?», «Händ Sie …?», «Bruuched Sie …?».
- Schreib wirklich Dialekt, nicht Hochdeutsch mit ein paar Dialektwörtern."""

# Standard German -> Zurich German, the prompt measured in E3. In the app (with the style guide) it gives the
# learner's own sentence the way a local would say it, in a separate call that cannot mix it up with the reply.
TRANSLATE_TO_ZURICH = ("Du übersetzt Hochdeutsch ins Zürichdeutsche (Züritüütsch), so wie eine Person aus der Stadt "
                       "Zürich den Satz sagen und in Dialekt-Schreibweise aufschreiben würde. Antworte nur mit dem "
                       "übersetzten Satz.")

LEVELS = {
    "A2": "Die lernende Person ist Anfängerin (A2). Sprich sehr einfach: höchstens 1 kurzer Satz, "
          "häufige Wörter, keine Redewendungen.",
    "B1": "Die lernende Person hat mittleres Niveau (B1). Sprich natürlich, aber kurz: 1–2 Sätze, Alltagswortschatz.",
    "B2": "Die lernende Person ist fortgeschritten (B2). Sprich wie im echten Leben: 1–2 Sätze, "
          "gern mit typischen Zürcher Ausdrücken.",
}

HELP_LANGS = {"en": "Englisch", "de": "Deutsch (einfaches Hochdeutsch)", "ru": "Russisch"}

# Repeated in every turn message: in long conversations the replies grew to 3-5 sentences.
REPLY_LENGTH = {"A2": "1 kurzer Satz", "B1": "1–2 kurze Sätze", "B2": "1–2 Sätze"}

TURN_SCHEMA = """\
Antworte IMMER mit genau einem JSON-Objekt, ohne weiteren Text:
{
  "reply_gsw": "deine nächste Äusserung in der Rolle, auf Züritüütsch",
  "reply_de": "reply_gsw sinngemäss in natürlichem Hochdeutsch, gleiche Anrede (Sie/du), Gipfeli, Znüni, Velo usw. bleiben (nicht der Satz der lernenden Person)",
  "reply_help": "reply_gsw sinngemäss auf {help_lang}, ohne etwas hinzuzufügen",
  "words": [{"gsw": "Dialektwort aus reply_gsw", "de": "hochdeutsche Entsprechung", "hint": "kurze Erklärung auf {help_lang}"}],
  "suggestions_de": ["eine mögliche Antwort der lernenden Person auf reply_gsw", "eine andere mögliche Antwort"],
  "scene_done": true | false
}"""


def system_prompt(sc: Scenario, level: str, help_lang: str) -> str:
    lang = HELP_LANGS.get(help_lang, HELP_LANGS["en"])
    schema = TURN_SCHEMA.replace("{help_lang}", lang)
    return f"""\
Du bist Teil der App «Los emal!», mit der Zugezogene lernen, Zürichdeutsch zu VERSTEHEN.
In Zürich ist es normal, dass Zugezogene auf Hochdeutsch antworten, während Einheimische Dialekt sprechen.
Genau das üben wir: Du sprichst Züritüütsch, die lernende Person antwortet auf Hochdeutsch.

Situation: {sc.setting}
Deine Rolle: {sc.role}. Bleib in der Rolle und führe das Gespräch natürlich weiter.
Dein Gegenüber ist die lernende Person, nicht du: {sc.learner} Setz nichts voraus, was sie dir noch nicht gesagt hat.
Du kennst dich in deiner Rolle aus und hilfst konkret. Frag die lernende Person nie, was du selbst brauchst.
{LEVELS.get(level, LEVELS['B1'])}

{ZURICH_STYLE_GUIDE}

Aufgaben in jeder Runde:
1. reply_gsw: deine nächste Äusserung in der Rolle, auf Züritüütsch. Hat die lernende Person etwas gefragt,
   beantworte das zuerst konkret; danach höchstens eine kurze Frage, damit das Gespräch weitergeht.
2. words: 1–3 Wörter oder Wendungen aus reply_gsw, die für Zugezogene am schwersten zu verstehen sind
   (nur solche, die anders sind als auf Hochdeutsch); hint immer auf {lang}.
3. suggestions_de: zwei verschiedene kurze Sätze, mit denen die lernende Person auf deine Äusserung reply_gsw
   antworten könnte, aus ihrer Sicht («ich» ist die lernende Person) und auf ihrem Niveau. Wiederhole nicht,
   was sie schon gesagt hat. Hochdeutsch, wie die lernende Person spricht, ohne Dialektwörter wie isch, zrugg,
   morn (Schweizer Wörter wie Billett, Velo, Gipfeli sind gut).
4. scene_done: true, wenn das Ziel erreicht ist oder das Gespräch natürlich endet; dann verabschiede dich.
Ob die lernende Person dich verstanden hat und ob ihr Hochdeutsch stimmt, wird separat geprüft; du
korrigierst sie nicht. Hat sie etwas anderes verstanden, sag es in der Rolle einfacher noch einmal.

{schema}"""


# Separate call: inside the role-play turn the model passed almost every learner error as correct (E4),
# while this short, focused check finds most of them.
GERMAN_CHECK = """\
Prüfe, ob der folgende Satz einer lernenden Person grammatisch korrektes Hochdeutsch ist: Artikel und Genus, Kasus,
Verbform und Hilfsverb (haben/sein), Partizip, Wortstellung, Präposition, Verneinung. Schweizer Hochdeutsch und
Schweizer Wörter sind korrekt und werden nie geändert (ss statt ß, Velo, Gipfeli, Billett, parkieren, Trottoir).
Ignoriere Gross-/Kleinschreibung und Satzzeichen, der Text kommt aus einer Spracherkennung. Ändere nur echte Fehler,
nicht den Stil. Schreib tip auf {lang}.
Satz: «{text}»
Antworte nur mit JSON: {{"ok": true oder false, "corrected_de": "korrigierter Satz oder leer", "tip": "kurze Erklärung des Fehlers auf {lang} oder leer"}}"""


def german_check_message(text: str, help_lang: str) -> str:
    return GERMAN_CHECK.format(text=text, lang=HELP_LANGS.get(help_lang, HELP_LANGS["en"]))


# Separate call as well: inside the role-play turn the model detected 3 of 20 misunderstandings (E4),
# this focused check 19 of 20 without a false alarm.
COMPREHENSION_CHECK = """\
Eine Person aus Zürich hat in einem Gespräch zu einer lernenden Person gesagt: «{gsw}»
(auf Hochdeutsch: «{de}»). Die lernende Person hat geantwortet: «{answer}»
Passt die Antwort zu dem, was gesagt oder gefragt wurde? Eine kurze, unvollständige oder grammatisch
falsche Antwort passt, wenn sie sich auf das Gesagte bezieht. Sie passt nicht, wenn die lernende Person
offensichtlich etwas anderes verstanden hat.
Antworte nur mit JSON: {{"fits": true oder false, "note": "nur wenn false: was eigentlich gesagt oder gefragt wurde, kurz auf {lang}"}}"""


def comprehension_message(said_gsw: str, said_de: str, answer: str, help_lang: str) -> str:
    return COMPREHENSION_CHECK.format(gsw=said_gsw, de=said_de or said_gsw, answer=answer,
                                      lang=HELP_LANGS.get(help_lang, HELP_LANGS["en"]))


def opening_message(level: str = "B1") -> str:
    return (f"Die Szene beginnt jetzt. Sprich die lernende Person zuerst an "
            f"(reply_gsw: höchstens {REPLY_LENGTH.get(level, REPLY_LENGTH['B1'])}). "
            "Antworte nur mit JSON.")


def learner_message(text: str, level: str = "B1", role: str = "") -> str:
    you = f"Du bist weiterhin: {role}. " if role else ""
    return (f"{you}Die lernende Person antwortet dir (Hochdeutsch, Text aus Spracherkennung, kann Fehler haben):\n"
            f"«{text}»\nAntworte ihr in deiner Rolle (reply_gsw: höchstens "
            f"{REPLY_LENGTH.get(level, REPLY_LENGTH['B1'])}). Antworte nur mit JSON.")
