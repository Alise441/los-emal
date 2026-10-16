"""Turn the official multimodal Apertus v1.5 8B into a text-only ApertusForCausalLM checkpoint and check that it
computes the same logits as the original on text."""
import json, sys
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, ApertusConfig, ApertusForCausalLM

SRC, DST = "/home/alina/models/apertus-v15-8b", "/home/alina/models/apertus-v15-8b-text"
cfg = json.load(open(f"{SRC}/config.json"))
tc = cfg["text_config"]
text_vocab = tc["output_vocab_size"]  # 131072 text tokens; ids above are image/audio tokens

conf = ApertusConfig(
    vocab_size=text_vocab, hidden_size=tc["hidden_size"], intermediate_size=tc["intermediate_size"],
    num_hidden_layers=tc["num_hidden_layers"], num_attention_heads=tc["num_attention_heads"],
    num_key_value_heads=tc["num_key_value_heads"], hidden_act=tc["hidden_act"],
    max_position_embeddings=tc["max_position_embeddings"], rms_norm_eps=tc["rms_norm_eps"],
    rope_parameters=tc["rope_parameters"], attention_bias=tc["attention_bias"], attention_dropout=0.0,
    bos_token_id=tc["bos_token_id"], eos_token_id=tc["eos_token_id"], pad_token_id=tc["pad_token_id"],
    tie_word_embeddings=False, qk_norm=tc.get("qk_norm", True))
print("config ok:", {k: getattr(conf, k, None) for k in ("vocab_size", "hidden_act", "qk_norm")})

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
state = {}
for name, shard in idx.items():
    if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
        continue
    with safe_open(f"{SRC}/{shard}", framework="pt") as f:
        t = f.get_tensor(name)
    new = name.replace("model.language_model.", "model.")
    if new == "model.embed_tokens.weight":
        t = t[:text_vocab].clone()
    state[new] = t
model = ApertusForCausalLM(conf).to(torch.bfloat16)
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing:", missing[:10], len(missing), "| unexpected:", unexpected[:10], len(unexpected))
assert not unexpected and all("rotary" in m for m in missing), "weights do not line up"
model = model.cuda().eval()

tok = AutoTokenizer.from_pretrained(SRC)
msgs = [{"role": "user", "content": "Übersetze ins Zürichdeutsche (Züritüütsch):\nDas Kind geht heute nicht zur Schule."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda()
print("prompt tokens:", ids.shape[1], "max id:", int(ids.max()))
with torch.no_grad():
    mine = model(ids).logits[0, -1].float()
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
print("generation:", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))

if "--compare" in sys.argv:  # the official multimodal model on the same text
    from transformers import AutoModelForImageTextToText
    ref = AutoModelForImageTextToText.from_pretrained(SRC, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        theirs = ref(input_ids=ids).logits[0, -1].float()[:text_vocab]
    print("max |logit diff|:", float((mine - theirs).abs().max()), "| same argmax:", int(mine.argmax()) == int(theirs.argmax()),
          "| top5 overlap:", len(set(mine.topk(5).indices.tolist()) & set(theirs.topk(5).indices.tolist())))
    del ref

model.save_pretrained(DST, safe_serialization=True)
tok.save_pretrained(DST)
import shutil
for f in ("chat_template.jinja", "generation_config.json", "LICENSE.txt"):
    try:
        shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
    except FileNotFoundError:
        pass
print("saved", DST)
