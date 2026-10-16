"""Translate the test sentences into Zurich German with the base and the fine-tuned Apertus v1.5 8B (greedy)."""
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, LORA = "/home/alina/models/apertus-v15-8b-text", "/home/alina/ft/lora-zh"
FT_PROMPT = "Übersetze ins Zürichdeutsche (Züritüütsch):\n{de}"
P = json.load(open("/home/alina/ft/prompts.json", encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
test = [json.loads(l) for l in open("/home/alina/ft/test.jsonl", encoding="utf-8")]
app = [json.loads(l) for l in open("/home/alina/ft/app.jsonl", encoding="utf-8")]


def first_line(text):
    line = next((ln for ln in (text or "").splitlines() if ln.strip()), "")
    return line.strip().strip('"„“«»').strip()


def generate(model, convs, bs=32):
    out = []
    for i in range(0, len(convs), bs):
        texts = [tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs[i:i + bs]]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [first_line(tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)) for g in gen]
        print(f"  {min(i + bs, len(convs))}/{len(convs)}", flush=True)
    return out


def e3_conv(system):
    return lambda r: [{"role": "system", "content": system}, {"role": "user", "content": r["de"]}]


ft_conv = lambda r: [{"role": "user", "content": FT_PROMPT.format(de=r["de"])}]
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
results = {}
for name, conv in (("base8b_baseline_prompt", e3_conv(P["base"])), ("base8b_with_style_guide", e3_conv(P["guided"]))):
    print("==", name)
    results[name] = {"test": generate(model, [conv(r) for r in test]), "app": generate(model, [conv(r) for r in app])}
model = PeftModel.from_pretrained(model, LORA).eval()
print("== finetuned8b")
results["finetuned8b"] = {"test": generate(model, [ft_conv(r) for r in test]), "app": generate(model, [ft_conv(r) for r in app])}
json.dump(results, open("/home/alina/ft/gen_results.json", "w", encoding="utf-8"), ensure_ascii=False)
print("saved gen_results.json")
