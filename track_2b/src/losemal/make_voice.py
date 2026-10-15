"""Create the synthetic reference voices in data/voices/ with Qwen3-TTS VoiceDesign.

No real person's voice is used: each voice is designed from a text description,
so the resulting wav can be shipped under the project licence.

    python -m losemal.make_voice            # 3 candidates per voice, first one becomes the default
    python -m losemal.make_voice --pick f=2 m=1   # choose other candidates after listening
"""
from __future__ import annotations

import argparse
import shutil

import soundfile as sf

from .config import DATA_DIR
from .tts import pick_device

VOICES = {
    "f": {
        "file": "vreni",
        "instruct": ("A friendly, warm woman in her thirties from Zurich, Switzerland. She speaks Swiss German "
                     "with a natural Zurich accent, clear articulation and a relaxed, everyday pace."),
        "text": "Grüezi mitenand! Ich bi d Vreni und wohne scho mis ganze Läbe z Züri. "
                "Hüt verzell ich Ihne chli öppis über üsi Stadt.",
    },
    "m": {
        "file": "vreni_m",
        "instruct": ("A calm, friendly man in his forties from Zurich, Switzerland. He speaks Swiss German "
                     "with a natural Zurich accent, a medium-low voice and an unhurried pace."),
        "text": "Sali zäme! Ich bi de Beat. Am Wuchenänd gang ich gärn a de See go bade "
                "oder mit em Velo uf de Üetliberg.",
    },
}

OUT = DATA_DIR / "voices"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="candidates per voice")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--pick", nargs="*", default=[], help="e.g. f=2 m=1 to select candidates without regenerating")
    args = ap.parse_args()
    cand_dir = OUT / "candidates"

    if args.pick:
        for item in args.pick:
            voice, idx = item.split("=")
            src = cand_dir / f"{VOICES[voice]['file']}_{idx}.wav"
            shutil.copy(src, OUT / f"{VOICES[voice]['file']}.wav")
            (OUT / f"{VOICES[voice]['file']}.txt").write_text(VOICES[voice]["text"], encoding="utf-8")
            print(f"{voice}: using {src.name}")
        return

    import torch
    from qwen_tts import Qwen3TTSModel

    device = pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = Qwen3TTSModel.from_pretrained(args.model, device_map=device, dtype=dtype,
                                          attn_implementation="flash_attention_2" if device == "cuda" else "sdpa")
    cand_dir.mkdir(parents=True, exist_ok=True)
    for voice, spec in VOICES.items():
        for i in range(1, args.n + 1):
            torch.manual_seed(1000 * i + ord(voice))
            wavs, sr = model.generate_voice_design(text=spec["text"], instruct=spec["instruct"], language="German")
            path = cand_dir / f"{spec['file']}_{i}.wav"
            sf.write(path, wavs[0], sr)
            print("wrote", path)
        shutil.copy(cand_dir / f"{spec['file']}_1.wav", OUT / f"{spec['file']}.wav")
        (OUT / f"{spec['file']}.txt").write_text(spec["text"], encoding="utf-8")
    (OUT / "README.md").write_text(
        "Synthetic reference voices generated with Qwen3-TTS-12Hz-1.7B-VoiceDesign (Apache-2.0) from the text "
        "descriptions in src/losemal/make_voice.py. No real person's voice was recorded or cloned.\n",
        encoding="utf-8")
    print("Listen to data/voices/candidates/*.wav and pick with: python -m losemal.make_voice --pick f=N m=N")


if __name__ == "__main__":
    main()
