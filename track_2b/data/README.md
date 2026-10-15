# Data

## `learner_de_testset.jsonl` — synthetic learner answers (63 items)

Created by the project team for evaluation E4 (see `technical_report.md`). Each item is a short
exchange in one of the app's Zurich scenarios: a line spoken in Zurich German (`context_gsw`, with a
Standard German translation `context_de`) and a learner's answer in Standard German (`learner_de`).

| field | meaning |
|---|---|
| `kind` | `correct` · `error` (exactly one typical learner error) · `misunderstanding` (correct German, but does not answer what was said) · `swiss_variant` (correct Swiss Standard German such as *parkieren*, *grillieren*, *das Tram* — must not be "corrected") |
| `gold_ok` | is the learner's Standard German correct? |
| `gold_understood` | does the answer fit what was said? |
| `error_type` | for `error` items: article/gender, case, auxiliary, participle, word order, preposition, negation |
| `expected_in_correction` | any of these strings must appear in a correct correction (case-insensitive) |

All sentences were written by the authors; no personal data. Licence: CDLA-Permissive-2.0.

## `voices/`

Synthetic reference voices for speech synthesis, generated with Qwen3-TTS VoiceDesign from text
descriptions (`src/losemal/make_voice.py`). No real person's voice was recorded or cloned.

SwissDial (used in evaluations E1–E3) is **not** included: it is available from ETH Zurich on
request for research use.
