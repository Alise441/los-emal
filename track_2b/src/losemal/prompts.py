"""Prompts. ZURICH_STYLE_GUIDE is the one measured in E3 (src/eval/e3_generation.py) and stays as it was for the
"how a local would say it" call and the glossary; the role-play uses DIALECT_RULES and a sample conversation."""
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

# For the role-play: only the forms to use. Naming the wrong ones («nicht viu») can make a model more likely to
# write them, and lines of the conversations still had viu, dr, hei. The sample conversation of each scene shows
# the dialect in use.
DIALECT_RULES = """Regeln für Stadtzürcher Dialekt (Züritüütsch), geschrieben in üblicher Dialektschreibung:
- Zürcher Formen: ich bi, du bisch, er/si isch, mer sind, Sie sind; ich han, du häsch, er hät, mer händ, Sie händ;
  ich chan, Sie chönd; ich wott, Sie wänd; ich mues; mer gönd, gömmer; nöd, gsi, und, ich, zwei;
  Artikel de / d / s und en / e / es; öppis, nüt, chli, vil, Mal, hüt, geschter, morn, au, scho, gärn, jetzt,
  dänn, eifach, vilicht.
- Vergangenheit mit Perfekt: «ich bi gsi», «mer händ gmacht».
- Grüezi / Uf Widerluege zu «Sie», Hoi / Sali / Tschüss zu «du». Merci (vilmal), Exgüsi, Gärn gscheh.
- Typische Wörter: Velo, Tram, Billett, Trottoir, poschte, luege, lose, säge, verzelle, schaffe,
  Znüni, Zmittag, Znacht, Gipfeli, Weggli, Rüebli, Poulet, Säckli, Händy, Charte.
- Höflichkeitsform: «Chönd Sie …?», «Wänd Sie …?», «Händ Sie …?», «Bruuched Sie …?», «Zahled Sie …?».
- Jeder Satz ganz auf Züritüütsch."""

ADDRESS = {
    "Sie": "Du sprichst die lernende Person mit «Sie» an (Grüezi, «Händ Sie …?», Uf Widerluege).",
    "du": "Ihr duzt euch (Hoi, «Häsch du …?», Tschüss).",
}

# The prompt the fine-tuned translator was trained with (E5): one user message, nothing else.
FINETUNED_TRANSLATE = "Übersetze ins Zürichdeutsche (Züritüütsch):\n{de}"

# Standard German -> Zurich German, the prompt measured in E3. In the app (with the style guide) it gives the
# learner's own sentence the way a local would say it, in a separate call that cannot mix it up with the reply.
TRANSLATE_TO_ZURICH = ("Du übersetzt Hochdeutsch ins Zürichdeutsche (Züritüütsch), so wie eine Person aus der Stadt "
                       "Zürich den Satz sagen und in Dialekt-Schreibweise aufschreiben würde. Antworte nur mit dem "
                       "übersetzten Satz.")

LEVELS = {
    "A2": "Die lernende Person ist Anfängerin (A2). Sprich sehr einfach: höchstens 1 kurzer Satz mit "
          "häufigen Wörtern.",
    "B1": "Die lernende Person hat mittleres Niveau (B1). Sprich natürlich, aber kurz: 1–2 Sätze, Alltagswortschatz.",
    "B2": "Die lernende Person ist fortgeschritten (B2). Sprich wie im echten Leben: 1–2 Sätze, "
          "gern mit typischen Zürcher Ausdrücken.",
}

HELP_LANGS = {"en": "Englisch", "de": "Deutsch (einfaches Hochdeutsch)"}

# Repeated in every turn message: in long conversations the replies grew to 3-5 sentences.
REPLY_LENGTH = {"A2": "1 kurzer Satz", "B1": "1–2 kurze Sätze", "B2": "1–2 Sätze"}

# Translations, word explanations and reply ideas come from separate short calls (see below): inside this
# JSON the model sometimes described what it was doing instead of translating, and explained words in German.
# A short reply also arrives sooner.
TURN_SCHEMA = """\
Antworte IMMER mit genau einem JSON-Objekt, ohne weiteren Text:
{
  "reply_gsw": "deine nächste Äusserung in der Rolle, auf Züritüütsch",
  "scene_done": true | false
}"""

# DIALOGUE=de: Apertus writes Standard German, which it handles far better, and a translator turns it into Zurich
# German (E5: a fine-tuned 8B translates better than Apertus writes the dialect itself).
TURN_SCHEMA_DE = """\
Antworte IMMER mit genau einem JSON-Objekt, ohne weiteren Text:
{
  "reply_de": "deine nächste Äusserung in der Rolle, auf Schweizer Hochdeutsch",
  "scene_done": true | false
}"""
REPLY_KEY = {"gsw": "reply_gsw", "de": "reply_de"}


def example_dialogue(sc: Scenario, dialogue: str = "gsw") -> str:
    lines = sc.example_de if dialogue == "de" else sc.example
    return "\n".join(("Du: " if i % 2 == 0 else "Lernende Person: ") + line for i, line in enumerate(lines))


# Every instruction says what to do: the old «don't» rules did not stop the model from repeating the price, and
# the sample conversation does most of the work (see Scenario.example).
def system_prompt(sc: Scenario, level: str, dialogue: str = "gsw") -> str:
    de = dialogue == "de"
    example = (f"\nSo kann das Gespräch klingen (Vorbild für Ton{'' if de else ', Dialekt'} und Ablauf; du reagierst "
               f"immer auf das, was die lernende Person wirklich sagt):\n{example_dialogue(sc, dialogue)}\n"
               ) if sc.example else ""
    speech = ("Du spielst eine Person aus Zürich. Schreib, was sie sagt, auf Schweizer Hochdeutsch, so wie man spricht:\n"
              "kurze Sätze, Schweizer Wörter (Gipfeli, Billett, Velo, Znüni, Trottoir). Deine Sätze werden danach\n"
              "ins Züritüütsch übersetzt; die lernende Person hört sie auf Züritüütsch und antwortet auf Hochdeutsch."
              if de else "Genau das üben wir: Du sprichst Züritüütsch, die lernende Person antwortet auf Hochdeutsch.")
    return f"""\
Du bist Teil der App «Los emal!», mit der Zugezogene lernen, Zürichdeutsch zu VERSTEHEN.
In Zürich ist es normal, dass Zugezogene auf Hochdeutsch antworten, während Einheimische Dialekt sprechen.
{speech}

Situation: {sc.setting}
Deine Rolle: {sc.role}. Bleib in der Rolle und führe das Gespräch natürlich weiter.
Dein Gegenüber ist die lernende Person: {sc.learner} Was sie möchte, erfährst du von ihr.
Du kennst dich in deiner Rolle aus und hilfst konkret mit dem, was du weisst.
{ADDRESS.get(sc.address, ADDRESS["Sie"])}
Ablauf: {sc.steps} Jede deiner Äusserungen bringt das Gespräch einen Schritt weiter; was geklärt ist, bleibt geklärt.
{LEVELS.get(level, LEVELS['B1'])}
{example}
{'' if de else DIALECT_RULES}

Aufgaben in jeder Runde:
1. {REPLY_KEY[dialogue]}: deine nächste Äusserung in der Rolle, auf {'Schweizer Hochdeutsch' if de else 'Züritüütsch'}. Hat die lernende Person etwas gefragt,
   beantworte das zuerst konkret; danach höchstens eine kurze Frage, damit das Gespräch weitergeht.
2. scene_done: true, sobald der Ablauf zu Ende ist; dann ist deine Äusserung ein kurzer Abschiedsgruss.
Du reagierst inhaltlich auf ihre Antworten, wie im echten Leben; ob sie dich verstanden hat, prüft die App separat.
Hat sie etwas anderes verstanden, sag es in der Rolle einfacher noch einmal.

{TURN_SCHEMA_DE if de else TURN_SCHEMA}"""


# Evaluation only (E4): the app no longer corrects the learner's German. Inside the role-play turn the model
# passed almost every learner error as correct, while this short, focused check finds most of them.
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
# this focused check 18 of 20 without a false alarm.
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


def opening_message(level: str = "B1", dialogue: str = "gsw") -> str:
    return (f"Die Szene beginnt jetzt. Sprich die lernende Person zuerst an "
            f"({REPLY_KEY[dialogue]}: höchstens {REPLY_LENGTH.get(level, REPLY_LENGTH['B1'])}). "
            "Antworte nur mit JSON.")


def learner_message(text: str, level: str = "B1", role: str = "", address: str = "", dialogue: str = "gsw") -> str:
    # The role and the form of address are repeated each turn: further into the conversation the model fell
    # out of its role and from «Sie» into «du».
    you = f"Du bist weiterhin: {role}" + (f", per {address}" if address else "") + ". " if role else ""
    return (f"{you}Die lernende Person antwortet dir (Hochdeutsch, Text aus Spracherkennung, kann Fehler haben):\n"
            f"«{text}»\nAntworte ihr in deiner Rolle ({REPLY_KEY[dialogue]}: höchstens "
            f"{REPLY_LENGTH.get(level, REPLY_LENGTH['B1'])}). Antworte nur mit JSON.")


# The local's line in Standard German or the help language, one short call per language. The scene context
# disambiguates (at the SBB counter «eifach» is a one-way ticket, not "simply").
LANG_INTO = {"de": "ins Hochdeutsche", "en": "ins Englische"}
TRANSLATE_LINE = """\
Situation: {context}
Übersetze, was die Person aus Zürich auf Züritüütsch gesagt hat, {into}. Übersetze genau das Gesagte, ohne
etwas zu erklären oder hinzuzufügen, mit der gleichen Anrede (Sie/du).{keep} Antworte nur mit der Übersetzung.
«{text}»"""


def scene_context(sc: Scenario) -> str:
    return f"{sc.role}. {sc.setting}"


# For the help language we translate the Standard German version, which the model translates more reliably
# than the dialect (from the dialect, «eifach» (one-way) became "simply" even with the scene as context).
TRANSLATE_FROM_DE = """\
Situation: {context}
Hier ist auf Hochdeutsch, was die Person aus Zürich gesagt hat. Übersetze es {into}, genau und ohne etwas zu
erklären oder hinzuzufügen. Antworte nur mit der Übersetzung.
«{text}»"""


def translate_message(text: str, target: str, context: str = "", from_de: bool = False) -> str:
    context, into = context or "Alltag in Zürich", LANG_INTO.get(target, LANG_INTO["en"])
    if from_de:
        return TRANSLATE_FROM_DE.format(context=context, into=into, text=text)
    keep = " Schweizer Wörter wie Gipfeli, Znüni, Velo, Billett bleiben." if target == "de" else ""
    return TRANSLATE_LINE.format(context=context, into=into, keep=keep, text=text)


# The hardest dialect words of the line. The explanation is requested in the help language itself:
# asked in German for an explanation in another language, the model answered in German.
HINT_IN = {"en": "short explanation of the meaning in English", "de": "kurze Erklärung der Bedeutung auf Hochdeutsch"}
HARD_WORDS = """\
Situation: {context}
Die Person aus Zürich hat gesagt: «{text}»
Welche 1–3 einzelnen Wörter oder kurzen Wendungen (höchstens drei Wörter, keine ganzen Sätze) darin verstehen
Deutschlernende am schwersten, weil sie anders sind als auf Hochdeutsch? Schreib sie genau so, wie sie im Satz
stehen, und erkläre ihre Bedeutung, nicht die Aussprache.
Antworte nur mit JSON: {{"words": [{{"gsw": "Wort aus dem Satz", "de": "hochdeutsche Entsprechung", "hint": "{hint}"}}]}}"""


def words_message(text: str, help_lang: str, context: str = "") -> str:
    return HARD_WORDS.format(context=context or "Alltag in Zürich", text=text,
                             hint=HINT_IN.get(help_lang, HINT_IN["en"]))


# Two ideas for the learner's next reply, asked for separately once the local's line is there.
SUGGEST = """\
Situation: {context}
Die lernende Person: {learner}
Bisher hat sie gesagt: {said}
Die Person aus Zürich hat gerade gesagt: «{line}»
Schreib zwei verschiedene kurze Sätze, mit denen die lernende Person jetzt antworten könnte: aus ihrer Sicht
(«ich» ist die lernende Person), auf ihrem Niveau ({level}), so dass das Gespräch ihrem Ziel näherkommt.
Wiederhole nicht, was sie schon gesagt hat. Hochdeutsch ohne Dialektwörter wie isch, zrugg, morn
(Schweizer Wörter wie Billett, Velo, Gipfeli sind gut).
Antworte nur mit JSON: {{"ideas": ["erster Satz", "zweiter Satz"]}}"""


def suggest_message(sc: Scenario, level: str, line: str, said: list[str]) -> str:
    return SUGGEST.format(context=scene_context(sc), learner=sc.learner, line=line, level=level,
                          said=" / ".join(f"«{x}»" for x in said) or "noch nichts")
