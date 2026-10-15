"""Speech synthesis with Qwen3-TTS (Apache-2.0) cloning a reference voice.

The reference voices in data/voices/ are synthetic: they were generated with
Qwen3-TTS VoiceDesign (see make_voice.py), so no real person's voice is cloned.
Any other consented recording can be used via TTS_REF_WAV / TTS_REF_TEXT.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import settings

log = logging.getLogger(__name__)


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def voice_files(voice: str) -> tuple[str, str]:
    """Reference wav + transcript for voice 'f' or 'm'. TTS_REF_WAV replaces both voices, TTS_REF_WAV_M only the
    male one (female roles then keep the synthetic voice)."""
    if voice == "m" and settings.tts_ref_wav_m:
        wav, text = Path(settings.tts_ref_wav_m), settings.tts_ref_text_m
    else:
        default_wav = Path(settings.tts_ref_wav)
        cand = (default_wav.with_name(f"vreni{'' if voice == 'f' else '_m'}.wav") if default_wav.name == "vreni.wav"
                else default_wav)
        wav, text = (cand if cand.exists() else default_wav), settings.tts_ref_text
    if not text:
        txt = wav.with_suffix(".txt")
        text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    return str(wav), text


def limit_decoder_chunks(frames: int = 50) -> None:
    """Decode speech in ~4 s chunks instead of up to 25 s. On CPU the decoder needs about 2.4 GB extra for a
    7 s reply in one piece, growing with the length, which breaks an 8 GB Docker memory limit."""
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder

    Qwen3TTSTokenizerV2Decoder.chunked_decode.__defaults__ = (frames, 25)  # (chunk_size, left_context_size)


def shrink_text_embedding(model) -> None:
    """Keep the text embedding (151,936 x 2048, a third of all TTS weights) in bfloat16 as stored in the
    checkpoint: lossless, 0.6 GB less memory in float32 runs, and a table lookup needs no fast bf16 maths."""
    import torch

    emb = model.model.talker.get_text_embeddings()
    emb.to(torch.bfloat16)
    emb.register_forward_hook(lambda _module, _args, out: out.float())


def trim_heap() -> None:
    """Hand freed memory back to the OS (Linux/glibc): loading and synthesis leave gigabytes of free heap."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class TTS:
    def __init__(self, backend: str | None = None, model: str | None = None):
        self.backend = (backend or settings.tts_backend).lower()
        self.model_name = model or settings.tts_model
        self._model = None
        self._prompts: dict[str, object] = {}
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[int, np.ndarray]] = OrderedDict()
        self.error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.backend != "none" and self.error is None

    def load(self) -> None:
        if self.backend == "none" or self._model is not None:
            return
        if self.backend == "tone":  # development only: a beep instead of speech, no model needed
            self._model = "tone"
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from qwen_tts import Qwen3TTSModel

                device = pick_device(settings.tts_device)
                options = [(device, torch.bfloat16), (device, torch.float32)] if device != "cpu" else []
                options.append(("cpu", torch.float32))
                last = None
                for dev, dtype in options:
                    try:
                        model = Qwen3TTSModel.from_pretrained(
                            self.model_name, device_map=dev, dtype=dtype,
                            attn_implementation="flash_attention_2" if dev == "cuda" else "sdpa")
                        self._model = model
                        if dev == "cpu":
                            limit_decoder_chunks()
                            shrink_text_embedding(model)
                        self._prompt_for("f")  # smoke test: reference voice + device work
                        trim_heap()
                        log.info("TTS ready: %s on %s/%s", self.model_name, dev, dtype)
                        return
                    except Exception as e:  # noqa: BLE001
                        last = e
                        self._model, self._prompts = None, {}
                        log.warning("TTS failed on %s/%s: %s", dev, dtype, e)
                raise RuntimeError(last)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                log.exception("TTS could not load")

    def _prompt_for(self, voice: str):
        if voice not in self._prompts:
            wav, text = voice_files(voice)
            if not Path(wav).exists():
                raise FileNotFoundError(f"reference voice {wav} missing — run `make voice`")
            self._prompts[voice] = self._model.create_voice_clone_prompt(
                ref_audio=wav, ref_text=text or None, x_vector_only_mode=not text)
        return self._prompts[voice]

    def synth(self, text: str, voice: str = "f", slow: bool = False) -> tuple[int, np.ndarray]:
        key = hashlib.sha1(f"{voice}|{slow}|{text}".encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if slow:
            sr, wav = self.synth(text, voice, slow=False)
            import librosa

            out = (sr, librosa.effects.time_stretch(wav.astype(np.float32), rate=0.78))
        else:
            self.load()
            if not self.enabled or self._model is None:
                raise RuntimeError(self.error or "speech synthesis is disabled")
            if self._model == "tone":
                sr = 16000
                t = np.arange(int(sr * min(0.06 * len(text), 4.0))) / sr
                return sr, (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            with self._lock:
                wavs, sr = self._model.generate_voice_clone(
                    text=text, language="German", voice_clone_prompt=self._prompt_for(voice))
                trim_heap()
            out = (int(sr), _mono(wavs[0]))
        self._cache[key] = out
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return out


def _mono(wav) -> np.ndarray:
    try:
        import torch

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
    except ImportError:
        pass
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=0) if wav.shape[0] < wav.shape[1] else wav.mean(axis=1)
    return wav
