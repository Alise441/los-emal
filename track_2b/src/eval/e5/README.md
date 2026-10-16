# E5 — fine-tuning Apertus 8B as a Standard German → Zurich German translator (research)

The scripts that produced the research model of E5,
[alise441/apertus-v1.5-8b-de-to-zurich](https://huggingface.co/alise441/apertus-v1.5-8b-de-to-zurich)
(technical report, section 5). They are here for transparency. The training data (SwissDial,
CC BY-NC 4.0) and the weights are not in this repository, and the model is not part of the submission.

They ran on one NVIDIA RTX PRO 6000 (Nebius) with torch, transformers ≥ 4.56 and peft; the paths at the
top of each script are those of that machine.

1. `extract_text.py` — turns the multimodal `swiss-ai/Apertus-v1.5-8B` checkpoint into a text-only
   `ApertusForCausalLM` (all text tensors; the embedding cut to the 131,072 text tokens) and checks that
   it generates.
2. `train_lora.py` — LoRA (rank 16, α 32, dropout 0.05 on q/k/v/o/up/down projections), 2 epochs,
   learning rate 1e-4, batch 16, loss on the answer only. Reads `train.jsonl`, one `{"de": …, "zh": …}`
   pair per line, and keeps 100 pairs as dev set.
3. `merge.py` — merges the adapter into the weights (for the MLX 4-bit conversion with `mlx_lm.convert`).
4. `gen_eval.py` — greedy translations of the held-out sentences and of 90 short sentences from the app
   by the base model (with the E3 prompts from `prompts.py`) and by the fine-tuned model; they were scored
   with the E3 metrics of `src/eval/`.

Data: the 4,065 SwissDial Zurich sentences with their Standard German versions; 450 are held out (among
them the 100 sentences of E3), the other 3,615 are `train.jsonl`.
