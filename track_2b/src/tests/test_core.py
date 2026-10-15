"""Fast unit tests (no models, no network):  make test"""
import json

from eval.e3_generation import markers
from losemal.config import DATA_DIR
from losemal.llm import LLMReply, extract_json, flatten_conversation, fold_system
from losemal.prompts import system_prompt
from losemal.scenarios import BY_ID, SCENARIOS
from losemal.tutor import Session, Turn, clean_local, clean_suggestions, to_turn


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


def test_to_turn_validates_and_drops_fake_corrections():
    data = {"reply_gsw": "Grüezi!", "words": [{"gsw": "Säckli", "de": "Tüte"}, "x"],
            "feedback": {"ok": False, "corrected_de": "Ich hätte gern ein Brot.", "tip": "t"}}
    t = to_turn(data, "", "Ich hätte gern ein Brot", 1.0)
    assert t.words == [{"gsw": "Säckli", "de": "Tüte", "hint": ""}]
    assert t.feedback["ok"] is True  # identical "correction" is not a correction


def test_to_turn_shows_no_feedback_without_a_verdict():
    data = {"reply_gsw": "Grüezi!", "feedback": {"ok": None, "corrected_de": "", "tip": ""}}
    assert to_turn(data, "", "Ich brauche ein Tüte", 1.0).feedback is None
    data["feedback"] = {"ok": False, "corrected_de": "", "tip": "falsch"}  # "wrong" with nothing to show
    assert to_turn(data, "", "Ich brauche ein Tüte", 1.0).feedback is None


def test_suggestions_are_new_and_in_standard_german():
    said = ["Ich möchte ein Retourbillett nach Luzern für morgen kaufen."]
    sugg = ["Ich möchte ein Retourbillett nach Luzern für morgen.", "Zweite Klasse, bitte.", "Zweite Klasse bitte!",
            "Ja, bitte hin und zruck.", "Ein Billett mit Halbtax, bitte."]
    assert clean_suggestions(sugg, said) == ["Zweite Klasse, bitte.", "Ein Billett mit Halbtax, bitte."]


def test_local_version_is_a_real_translation():
    learner = "Kann ich mit Karte bezahlen?"
    assert clean_local("«Chan ich mit Charte zahle?»\nOder: ...", learner) == "Chan ich mit Charte zahle?"
    assert clean_local("Kann ich mit Karte zahlen?", learner) == ""  # copied, not translated


def test_to_turn_falls_back_to_raw_text():
    t = to_turn(None, "Grüezi wohl", "", 0.0)
    assert t.reply_gsw == "Grüezi wohl" and not t.parse_ok


def test_prompt_contains_scenario_and_help_language():
    p = system_prompt(BY_ID["sbb"], "A2", "ru")
    assert "Halbtax" in p and "Russisch" in p and "nöd" in p


def test_prompt_never_addresses_the_model_with_the_learners_goal():
    # The learner's goal card says «du»; inside the prompt «du» is the model, which then asked the learner
    # which documents it needed. The model gets the third-person description instead.
    for s in SCENARIOS:
        p = system_prompt(s, "B1", "en")
        assert s.goal["de"] not in p and s.learner in p


class FakeClient:
    """Answers the Standard German check, the Zurich version and the role-play turn differently."""

    def chat(self, messages, **kw):
        if messages[-1]["content"].startswith("Prüfe"):
            return LLMReply('{"ok": false, "corrected_de": "Ich brauche eine Tüte.", "tip": "die Tüte"}', 0.1, "fake")
        if messages[0]["content"].startswith("Du übersetzt"):
            return LLMReply("I bruuch e Täsche.", 0.1, "fake")
        if messages[-1]["content"].startswith("Eine Person aus Zürich"):
            return LLMReply('{"fits": false, "note": "She asked about a bag."}', 0.1, "fake")
        return LLMReply('{"reply_gsw": "Gärn, da isch es Säckli.", "feedback": {"ok": true}}', 0.2, "fake")

    def chat_json(self, messages, **kw):
        reply = self.chat(messages)
        return extract_json(reply.text), reply


def test_feedback_comes_from_the_separate_german_check():
    s = Session("migros")
    s.turns = [Turn(reply_gsw="Bruuched Sie es Säckli?", reply_de="Brauchen Sie eine Tüte?")]
    t = s.respond(FakeClient(), "Ich brauche ein Tüte.")
    assert t.reply_gsw == "Gärn, da isch es Säckli."
    assert t.feedback == {"ok": False, "corrected_de": "Ich brauche eine Tüte.", "tip": "die Tüte"}
    assert t.local_gsw == "I bruuch e Täsche."
    assert t.understood is False and t.understood_note == "She asked about a bag."


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
        assert {"en", "de", "ru"} <= set(s.title) and {"en", "de", "ru"} <= set(s.goal)


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
