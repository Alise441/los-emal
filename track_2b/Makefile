.PHONY: run stop run-local voice test eval-judge eval-generation eval-feedback eval-asr venv-asr

PY ?= python3
# SwissDial (research use only) lives outside the repository; default: data/swissdial next to the repo
SWISSDIAL ?= $(abspath $(CURDIR)/../../data/swissdial)
# E1 needs its own environment: Apertus audio input requires the Swiss AI transformers fork
ASR_PY ?= .venv-asr/bin/python
APERTUS_TRANSFORMERS = transformers[torch,vision] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4

# Judges: start the app in Docker -> http://localhost:7860
# Needs LLM_NAME / LLM_BASE_URL / LLM_API_KEY in the environment or in .env
run:
	docker compose up --build

stop:
	docker compose down

# Native run (fastest on Apple Silicon: ASR_BACKEND=mlx TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base)
run-local:
	PYTHONPATH=src $(PY) -m losemal.app

# Re-create the synthetic reference voices in data/voices/ (Qwen3-TTS VoiceDesign)
voice:
	PYTHONPATH=src $(PY) -m losemal.make_voice

test:
	PYTHONPATH=src $(PY) -m pytest -q src/tests

# Evaluations from the technical report (SwissDial must be requested from ETH; see report)
eval-judge:
	PYTHONPATH=src $(PY) -m eval.e2_judge --swissdial $(SWISSDIAL)

eval-generation:
	PYTHONPATH=src $(PY) -m eval.e3_generation --swissdial $(SWISSDIAL)

eval-feedback:
	PYTHONPATH=src $(PY) -m eval.e4_feedback

eval-asr:
	PYTHONPATH=src $(ASR_PY) -m eval.e1_asr --swissdial $(SWISSDIAL)

# Environment for E1 (Apple Silicon: mlx-whisper). The fork's [audio] extra is left out on purpose:
# it pulls kenlm, which has no macOS arm64 wheel; Apertus decodes audio with librosa.
venv-asr:
	uv venv --allow-existing --python 3.11 .venv-asr
	uv pip install --python .venv-asr "$(APERTUS_TRANSFORMERS)" accelerate librosa soundfile mlx-whisper jiwer sacrebleu
