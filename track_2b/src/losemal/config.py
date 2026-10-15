"""All runtime settings come from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]  # .../track_2b
DATA_DIR = PROJECT_DIR / "data"


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    # LLM: any OpenAI-compatible endpoint serving Apertus (names follow the hackathon template)
    llm_name: str = _env("LLM_NAME", "swiss-ai/apertus-v1.5-70b")
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.publicai.co/v1")
    llm_api_key: str = _env("LLM_API_KEY")
    llm_timeout: float = float(_env("LLM_TIMEOUT", "90"))
    llm_log: str = _env("LLM_LOG")  # JSONL file for every request/response (debugging; contains learner input)
    llm_fallback_name: str = _env("LLM_FALLBACK_NAME")  # model to switch to when LLM_NAME is throttled (429)
    llm_concurrency: int = int(_env("LLM_CONCURRENCY", "2"))  # requests at once per client (Featherless allows 2)

    # Speech recognition: faster-whisper (CPU/CUDA, default) | mlx (Apple Silicon) | none
    asr_backend: str = _env("ASR_BACKEND", "faster-whisper")
    asr_model: str = _env("ASR_MODEL", "large-v3-turbo")
    asr_language: str = _env("ASR_LANGUAGE", "de")

    # Speech synthesis: qwen (Qwen3-TTS voice clone) | none
    tts_backend: str = _env("TTS_BACKEND", "qwen")
    tts_model: str = _env("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    tts_ref_wav: str = _env("TTS_REF_WAV", str(DATA_DIR / "voices" / "vreni.wav"))
    tts_ref_text: str = _env("TTS_REF_TEXT")  # defaults to the .txt next to the wav
    tts_ref_wav_m: str = _env("TTS_REF_WAV_M")  # replaces only the male voice, e.g. a consented recording
    tts_ref_text_m: str = _env("TTS_REF_TEXT_M")
    tts_device: str = _env("TTS_DEVICE", "auto")  # auto | cuda | mps | cpu

    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "7860"))


settings = Settings()
