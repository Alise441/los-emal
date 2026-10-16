"""Fast unit tests (no models, no network):  make test"""
import json

from eval.e3_generation import markers
from losemal.config import DATA_DIR
from losemal.llm import LLMReply, extract_json, flatten_conversation, fold_system
from losemal.prompts import scene_context, system_prompt, translate_message, words_message
from losemal.scenarios import BY_ID, SCENARIOS
from losemal.tutor import (Session, Turn, clean_local, clean_suggestions, clean_words, explain, parse_feedback,
                           suggest_replies, to_turn)


def test_extract_json_tolerates_fences_and_chatter():
    raw = 'Klar!\n```json\n{"reply_gsw": "Grüezi {mitenand}", "words": [1, 2,],}\n```\nMerci'
    assert extract_json(raw) == {"reply_gsw": "Grüezi {mitenand}", "words": [1, 2]}
    assert extract_json("kein JSON") is None


def test_fold_system_moves_instructions_into_first_user_turn():
    msgs = fold_system([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert msgs == [{"role": "user", "content": "[Anweisungen]\nS\n\n[Nachricht]\nU"}]


def test_flatten_conversation_keeps_history_in_one_user_turn():
    msgs = flatten_conversation([{"role": "system", "content": "S"}, {"role": "user", "content": "A"},
                                 {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}])
    assert msgs == [{"role": "user",
                     "content": "[Anweisungen]\nS\n\n[Bisheriges Gespräch]\nLernende Person: A\nDu: B\n\n[Nachricht]\nC"}]


def test_hard_words_must_occur_in_the_line():
    items = [{"gsw": "Säckli", "de": "Tüte"}, "x", {"gsw": "gmacht", "de": "gemacht"}, {"de": "ohne gsw"},
             {"gsw": "Bruuched Sie es Säckli", "de": "ganzer Satz"}]
    assert clean_words(items, "Bruuched Sie es Säckli?") == [{"gsw": "Säckli", "de": "Tüte", "hint": ""}]
    assert clean_words(None, "Grüezi") == []


def test_german_check_verdicts_are_validated():  # used by the evaluation (E4)
    fake = {"ok": False, "corrected_de": "Ich hätte gern ein Brot.", "tip": "t"}
    assert parse_feedback(fake, "Ich hätte gern ein Brot")["ok"] is True  # identical "correction"
    assert parse_feedback({"ok": None, "corrected_de": "", "tip": ""}, "Ich brauche ein Tüte") is None
    assert parse_feedback({"ok": False, "corrected_de": "", "tip": "falsch"}, "Ich brauche ein Tüte") is None


def test_suggestions_are_new_and_in_standard_german():
    said = ["Ich möchte ein Retourbillett nach Luzern für morgen kaufen."]
    sugg = ["Ich möchte ein Retourbillett nach Luzern für morgen.", "Zweite Klasse, bitte.", "Zweite Klasse bitte!",
            "Ja, bitte hin und zruck.", "Ein Billett mit Halbtax, bitte."]
    assert clean_suggestions(sugg, said) == ["Zweite Klasse, bitte.", "Ein Billett mit Halbtax, bitte."]


def test_local_version_is_a_real_translation():
    learner = "Kann ich mit Karte bezahlen?"
    assert clean_local("«Chan ich mit Charte zahle?»\nOder: ...", learner) == "Chan ich mit Charte zahle?"
    assert clean_local("Kann ich mit Karte zahlen?", learner) == ""  # copied, not translated
    assert clean_local("Kann ich mit Charte bezahlen?", learner) == ""  # one word changed: not worth showing
    learner = "Nehmen Sie das Tram 3 Richtung Klusplatz bis zur Haltestelle Kunsthaus."  # many names, four dialect words
    assert clean_local("Nehmed Sie s Tram 3 Richtig Klusplatz bis zur Haltestell Kunsthaus.", learner)


def test_to_turn_falls_back_to_raw_text():
    t = to_turn(None, "Grüezi wohl", "", 0.0)
    assert t.reply_gsw == "Grüezi wohl" and not t.parse_ok


def test_prompts_contain_scenario_and_help_language():
    p = system_prompt(BY_ID["migros"], "A2")
    assert "Cumulus" in p and "nöd" in p and "reply_de" not in p  # translations come from separate calls
    assert "ins Englische" in translate_message("Grüezi", "en") and "in English" in words_message("Grüezi", "en")
    assert "Kassierer. Migros-Filiale am Limmatplatz" in translate_message(
        "Bruuched Sie es Säckli?", "en", scene_context(BY_ID["migros"]))
    assert "Gipfeli" in translate_message("Grüezi", "de")  # Swiss words stay in the Standard German version


def test_scenes_have_a_sample_conversation_and_positive_rules():
    from losemal.prompts import DIALECT_RULES
    opening = json.loads((DATA_DIR / "openings.json").read_text(encoding="utf-8"))
    for s in SCENARIOS:
        assert s.address in ("Sie", "du") and s.steps and len(s.example) >= 7
        assert s.example[0] == opening[s.id]["gsw"]  # the sample starts where the scene starts
        assert f"Lernende Person: {s.example[1]}" in system_prompt(s, "B1")
    assert "nicht" not in DIALECT_RULES  # only the forms to use (see prompts.py)


def test_prompt_never_addresses_the_model_with_the_learners_goal():
    # The learner's goal card says «du»; inside the prompt «du» is the model, which then asked the learner
    # which documents it needed. The model gets the third-person description instead.
    for s in SCENARIOS:
        p = system_prompt(s, "B1")
        assert s.goal["de"] not in p and s.learner in p


class FakeClient:
    """Answers the comprehension check, the Zurich version and the role-play turn differently."""

    def chat(self, messages, **kw):
        assert not messages[-1]["content"].startswith("Prüfe"), "the app must not correct the learner's German"
        if messages[0]["content"].startswith("Du übersetzt"):
            return LLMReply("I bruuch e Täsche.", 0.1, "fake")
        if messages[-1]["content"].startswith("Eine Person aus Zürich hat in einem Gespräch"):
            return LLMReply('{"fits": false, "note": "She asked about a bag."}', 0.1, "fake")
        if "ins Hochdeutsche" in messages[-1]["content"]:
            return LLMReply("«Brauchen Sie eine Tüte?»\n\nHinweis: …", 0.1, "fake")
        if "ins Englische" in messages[-1]["content"]:
            assert "«Brauchen Sie eine Tüte?»" in messages[-1]["content"]  # translated from Standard German
            return LLMReply("Do you need a bag?", 0.1, "fake")
        if "Schreib zwei verschiedene kurze Sätze" in messages[-1]["content"]:
            return LLMReply('{"ideas": ["Ja, gern eine Tüte.", "Ja, gern eine Tüte!", "Nei, i bruuch keis Säckli."]}',
                            0.1, "fake")
        if "Welche 1–3" in messages[-1]["content"]:
            return LLMReply('{"words": [{"gsw": "Säckli", "de": "Tüte", "hint": "small bag"}, '
                            '{"gsw": "Täsche", "de": "Tasche"}]}', 0.1, "fake")
        return LLMReply('{"reply_gsw": "Gärn, da isch es Säckli."}', 0.2, "fake")

    def chat_json(self, messages, **kw):
        reply = self.chat(messages)
        return extract_json(reply.text), reply


def test_comprehension_and_zurich_version_come_from_separate_calls():
    s = Session("migros")
    s.turns = [Turn(reply_gsw="Bruuched Sie es Säckli?", reply_de="Brauchen Sie eine Tüte?")]
    t = s.respond(FakeClient(), "Ich brauche ein Tüte.")
    assert t.reply_gsw == "Gärn, da isch es Säckli."
    assert t.local_gsw == "I bruuch e Täsche."
    assert t.understood is False and t.understood_note == "She asked about a bag."


def test_explain_fills_translations_and_words_from_separate_calls():
    turn = Turn(reply_gsw="Bruuched Sie es Säckli?")
    explain(FakeClient(), turn, "en")
    assert turn.reply_de == "Brauchen Sie eine Tüte?" and turn.reply_help == "Do you need a bag?"
    assert turn.words == [{"gsw": "Säckli", "de": "Tüte", "hint": "small bag"}]  # «Täsche» is not in the line
    explain(FakeClient(), turn, "de")
    assert turn.reply_help == ""  # German help needs no second translation


def test_reply_ideas_are_separate_and_in_standard_german():
    s = Session("migros")
    s.turns = [Turn(reply_gsw="Bruuched Sie es Säckli?")]
    assert suggest_replies(FakeClient(), s) == ["Ja, gern eine Tüte."]  # no repeat, no dialect


def test_prepared_opening_starts_without_the_model():
    opening = json.loads((DATA_DIR / "openings.json").read_text(encoding="utf-8"))
    assert {s.id for s in SCENARIOS} == set(opening)
    for lang in ("en", "de"):
        s = Session("migros", help_lang=lang)
        turn = s.open_with(opening["migros"])
        assert turn.reply_gsw == opening["migros"]["gsw"] and turn.reply_de and turn.suggestions_de
        assert bool(turn.reply_help) == (lang == "en")
        assert s.history[-1] == {"role": "assistant", "content": turn.reply_de}  # DIALOGUE=de: its own Standard German
    s = Session("migros", dialogue="gsw")
    assert s.open_with(opening["migros"]) and s.history[-1]["content"] == opening["migros"]["gsw"]


class FakeTranslator:
    """A fine-tuned translator: answers its training prompt only."""
    fine_tuned = True

    def __init__(self):
        self.seen = []

    def chat(self, messages, **kw):
        self.seen.append(messages)
        assert len(messages) == 1 and messages[0]["content"].startswith("Übersetze ins Zürichdeutsche (Züritüütsch):\n")
        return LLMReply("Zwei Gipfeli und es Ruchbrot, bitte.", 0.1, "fake")


class FakeDialogue:
    """Apertus writing the local's lines in Standard German (DIALOGUE=de)."""

    def chat_json(self, messages, **kw):
        assert "reply_de" in messages[0]["content"] and "Regeln für Stadtzürcher Dialekt" not in messages[0]["content"]
        return {"reply_de": "Das macht 7.40, bar oder Karte?", "scene_done": False}, LLMReply("{}", 0.2, "fake")


def test_standard_german_dialogue_is_translated_by_the_fine_tuned_model():
    from losemal.tutor import local_version
    tr = FakeTranslator()
    s = Session("baeckerei", dialogue="de")
    s.open_with(json.loads((DATA_DIR / "openings.json").read_text(encoding="utf-8"))["baeckerei"])
    assert s.history[-1]["content"] == "Grüezi! Was darf es sein?"  # the model sees its own Standard German
    t = s.reply(FakeDialogue(), "Zwei Gipfeli und ein Ruchbrot, bitte.", tr)
    assert t.reply_de == "Das macht 7.40, bar oder Karte?" and t.reply_gsw == "Zwei Gipfeli und es Ruchbrot, bitte."
    assert s.history[-1] == {"role": "assistant", "content": "Das macht 7.40, bar oder Karte?"}
    assert local_version(tr, "Ich hätte gern zwei Gipfeli und ein Brot.") == "Zwei Gipfeli und es Ruchbrot, bitte."
    assert "Lernende Person: Zwei Gipfeli und ein Ruchbrot, bitte." in system_prompt(BY_ID["baeckerei"], "B1", "de")


def test_known_slips_of_the_translators_are_fixed():
    from losemal.tutor import _fix_slips
    assert _fix_slips("Zwei Gipfeli, bitte.", "Zwoi Gipfeli, bitte. Merci viu mol.") == "Zwei Gipfeli, bitte. Merci vil mol."
    assert _fix_slips("Haben Sie Orangensaft?", "Haubi Orangensaft?") == "Händ Sie Orangensaft?"
    assert _fix_slips("Hat es geklappt?", "Häter gschafft?") == "Hät s gschafft?"
    assert (_fix_slips("Halten Sie die Karte ans Gerät. Möchten Sie bar zahlen?", "Händ Sie d Charte am Gäret. Möchted bar zahle?")
            == "Hebed Sie d Charte as Grät. Möchted Sie bar zahle?")
    assert _fix_slips("Zahlen Sie bar?", "Zahled Sie bar?") == "Zahled Sie bar?"  # no second «Sie»


class EchoTranslator:
    """A fine-tuned translator that marks what it was given, and says goodbye the way the real one did."""
    fine_tuned = True

    def __init__(self):
        self.seen = []

    def chat(self, messages, **kw):
        text = messages[0]["content"].split("\n", 1)[1]
        self.seen.append(text)
        return LLMReply("ZH: " + text.replace("Auf Wiedersehen", "Uf Wiederhersäge").replace("auf Wiedersehen", "uf Wiederhersäge"), 0.1, "fake")


def test_greetings_and_farewells_come_from_the_checked_list():
    from losemal.tutor import to_zurich
    tr = EchoTranslator()
    assert to_zurich(tr, "Gern. Sonst noch etwas?") == "Gärn. Suscht no öppis?" and tr.seen == []
    # the whole line goes to the translator (context), the formula's sentence is then replaced
    assert (to_zurich(tr, "Vielen Dank für Ihre Hilfe! Das ist nett. Auf Wiedersehen.")
            == "ZH: Vielen Dank für Ihre Hilfe! Das ist nett. Uf Widerluege.")
    assert tr.seen[-1] == "Vielen Dank für Ihre Hilfe! Das ist nett. Auf Wiedersehen."
    assert to_zurich(tr, "Danke und auf Wiedersehen!") == "ZH: Danke und uf Widerluege!"  # inside a sentence
    assert to_zurich(tr, "Gern. Das kostet 2.50 Franken.") == "Gärn. Das kostet 2.50 Franken."  # 2.50 stays whole


def test_session_history_is_bounded():
    s = Session("baeckerei")
    s.history = [{"role": "user", "content": str(i)} for i in range(40)]
    msgs = s._messages("neu")
    assert msgs[0]["role"] == "system" and msgs[-1]["content"] == "neu" and len(msgs) <= 16


def test_marker_metric_separates_dialects():
    zh = markers(["Er hät nöd gsi und de Maa isch cho."])
    be = markers(["Är het nid gsy u dr Maa isch cho."])
    assert zh["zurich_share"] > 0.9 and be["zurich_share"] < 0.1


def test_learner_testset_is_consistent():
    rows = [json.loads(line) for line in open(DATA_DIR / "learner_de_testset.jsonl", encoding="utf-8")]
    assert len(rows) >= 60
    for r in rows:
        assert r["scenario"] in BY_ID
        if r["kind"] == "error":
            assert r["gold_ok"] is False and r["expected_in_correction"]
        else:
            assert r["gold_ok"] is True
        assert r["gold_understood"] is (r["kind"] != "misunderstanding")


def test_scenarios_have_all_ui_languages():
    for s in SCENARIOS:
        assert set(s.title) == {"en", "de"} == set(s.goal)


def test_client_switches_to_the_fallback_model_when_throttled(monkeypatch):
    import httpx
    from openai import RateLimitError

    from losemal import llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    c = llm.ApertusClient(model="big", api_key="x", fallback="small")
    asked = []

    def create(model, **kw):
        asked.append(model)
        if model == "big":
            raise RateLimitError("busy", response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
                                 body=None)
        msg = type("M", (), {"content": "Grüezi"})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    monkeypatch.setattr(c.client.chat.completions, "create", create)
    reply = c.chat([{"role": "user", "content": "Hallo"}])
    assert reply.text == "Grüezi" and reply.model == "small" and asked == ["big", "small"]


def test_a_male_reference_voice_replaces_only_the_male_roles(monkeypatch, tmp_path):
    import dataclasses

    from losemal import tts

    own = tmp_path / "speaker.wav"
    own.write_bytes(b"")
    (tmp_path / "speaker.txt").write_text("Grüezi mitenand", encoding="utf-8")
    monkeypatch.setattr(tts, "settings", dataclasses.replace(tts.settings, tts_ref_wav_m=str(own)))
    assert tts.voice_files("m") == (str(own), "Grüezi mitenand")
    assert tts.voice_files("f")[0].endswith("vreni.wav")
