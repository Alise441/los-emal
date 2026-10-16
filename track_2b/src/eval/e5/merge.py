"""Merge the LoRA adapter into the text-only Apertus v1.5 8B (bf16) for conversion to MLX / GGUF."""
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL, LORA, OUT = "/home/alina/models/apertus-v15-8b-text", "/home/alina/ft/lora-zh", "/home/alina/models/apertus-v15-8b-gsw-zh"
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16), LORA).merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL).save_pretrained(OUT)
for f in ("chat_template.jinja", "generation_config.json", "LICENSE.txt"):
    shutil.copy(f"{MODEL}/{f}", f"{OUT}/{f}")
print("merged ->", OUT)
