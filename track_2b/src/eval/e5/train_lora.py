"""LoRA fine-tuning of the text-only Apertus v1.5 8B on SwissDial pairs: Standard German -> Zurich German.
The loss is on the assistant's answer only; 100 training pairs are held out to watch the dev loss."""
import json, math, random, time
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL, OUT = "/home/alina/models/apertus-v15-8b-text", "/home/alina/ft/lora-zh"
PROMPT = "Übersetze ins Zürichdeutsche (Züritüütsch):\n{de}"
EPOCHS, BATCH, LR = 2, 16, 1e-4
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open("/home/alina/ft/train.jsonl", encoding="utf-8")]
random.Random(0).shuffle(rows)
dev, train = rows[:100], rows[100:]


def encode(r):
    msgs = [{"role": "user", "content": PROMPT.format(de=r["de"])}]
    p_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    f_text = tok.apply_chat_template(msgs + [{"role": "assistant", "content": r["zh"]}], tokenize=False)
    assert f_text.startswith(p_text), "the prompt must be a prefix of the full conversation"
    p = tok(p_text, add_special_tokens=False)["input_ids"]
    f = tok(f_text, add_special_tokens=False)["input_ids"]
    return f, [-100] * len(p) + f[len(p):]


ex = encode(train[0])
print("example (end of rendering):", repr(tok.apply_chat_template(
    [{"role": "user", "content": PROMPT.format(de=train[0]["de"])}, {"role": "assistant", "content": train[0]["zh"]}],
    tokenize=False)[-260:]))
print("tokens:", len(ex[0]), "trained on:", sum(l != -100 for l in ex[1]))
train_enc, dev_enc = [encode(r) for r in train], [encode(r) for r in dev]


def batches(data, shuffle):
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), BATCH):
        chunk = [data[j] for j in order[i:i + BATCH]]
        n = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), n), tok.pad_token_id)
        lab = torch.full((len(chunk), n), -100)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, :len(x)], lab[k, :len(y)], att[k, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids.cuda(), lab.cuda(), att.cuda()


model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
steps = EPOCHS * math.ceil(len(train_enc) / BATCH)
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)


def dev_loss():
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for ids, lab, att in batches(dev_enc, False):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            tot, n = tot + loss.item(), n + 1
    model.train()
    return tot / n


print(f"dev loss before: {dev_loss():.3f}")
step, t0 = 0, time.time()
model.train()
for epoch in range(EPOCHS):
    for ids, lab, att in batches(train_enc, True):
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        step += 1
        if step % 25 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f} s)", flush=True)
    print(f"epoch {epoch + 1}: dev loss {dev_loss():.3f} ({time.time() - t0:.0f} s)", flush=True)
model.save_pretrained(OUT)
print("saved", OUT)
