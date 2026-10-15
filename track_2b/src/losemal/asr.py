"""Speech recognition with open-weights Whisper.

Backends:
- faster-whisper (default; CPU int8 or CUDA float16) — used in Docker;
- mlx (Apple Silicon, native) — fastest for local development on a Mac;
- none — text input only.

Whisper was chosen over Apertus' own audio input after our evaluation on SwissDial
(see technical report, E1): WER 30.9 % vs 88.8 % on 100 SwissDial Zurich German clips.
"""
from __future__ import annotations

import logging
import threading

from .config import settings

log = logging.getLogger(__name__)


class ASR:
    def __init__(self, backend: str | None = None, model: str | None = None, language: str | None = None):
        self.backend = (backend or settings.asr_backend).lower()
        self.model_name = model or settings.asr_model
        self.language = language or settings.asr_language or None
        self._model = None
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.backend != "none" and self.error is None

    def load(self) -> None:
        if self.backend == "none" or self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                if self.backend == "mlx":
                    import mlx_whisper  # noqa: F401

                    repo = self.model_name
                    if "/" not in repo:
                        repo = f"mlx-community/whisper-{repo}"
                    self._model = ("mlx", repo)
                else:
                    from faster_whisper import WhisperModel

                    device = "cuda" if _cuda() else "cpu"
                    compute = "float16" if device == "cuda" else "int8"
                    self._model = ("fw", WhisperModel(self.model_name, device=device, compute_type=compute))
                log.info("ASR ready: %s %s", self.backend, self.model_name)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                log.exception("ASR could not load")

    def transcribe(self, wav_path: str) -> str:
        self.load()
        if not self.enabled or self._model is None:
            raise RuntimeError(self.error or "speech recognition is disabled")
        kind, model = self._model
        if kind == "mlx":
            import mlx_whisper

            r = mlx_whisper.transcribe(wav_path, path_or_hf_repo=model, language=self.language,
                                       condition_on_previous_text=False, verbose=None)
            return (r.get("text") or "").strip()
        segments, _ = model.transcribe(wav_path, language=self.language, beam_size=5, vad_filter=True,
                                       condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


def _cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False
