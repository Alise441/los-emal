# Los emal! — understand Zurich German with Apertus

*«Los emal!» is Zurich German for "Listen!"*

![A bakery scene in Los emal!: the baker's line in Zurich German with translations and word help, and a correction of the learner's Standard German](track_2b/docs/screenshot.png)

Newcomers in Zurich learn Standard German in class, and then the baker, the neighbour and the
colleague at the coffee machine answer in *Züritüütsch*. Answering in Standard German is socially
perfectly normal; **understanding** the dialect is the real barrier.

**Los emal!** is a speaking-and-listening trainer for exactly this situation. In ten everyday
Zurich scenes (bakery, Migros checkout, Kreisbüro, GP's practice, SBB counter, laundry-room talk
with the neighbour, flat viewing …) a local speaks Zurich German, the learner listens — text hidden
by default — and answers in Standard German by voice. Apertus 1.5 plays the local, explains the
hardest dialect words, checks whether the learner understood, and gives feedback on the learner's
Standard German.

Hack Apertus 2026 · Track 2B (own project). Code and technical report: [`track_2b/`](track_2b/).

## Quick start

```bash
cd track_2b
cp .env.example .env        # put your Apertus endpoint and key in LLM_* variables
make run                    # Docker → http://localhost:7860
```

The first build downloads about 6 GB (Python packages and the speech models, which are baked into
the image) and takes a few minutes. The container needs about 7.5 GB of memory: give Docker at least
8 GB (Docker Desktop → Settings → Resources). On CPU the speech models are slow, so a turn takes
20–50 s; natively on Apple Silicon it takes 10–20 s (technical report, sections 5 and 8).

## Why it is built this way

Our evaluation (technical report, section 5) shaped the design:

- **Apertus speaks Zurich German reasonably, but cannot judge it.** Asked whether a sentence is
  Zurich German, Standard German or Bernese German, 70B and 8B were right in only 60 % and 58 % of
  150 cases, below the 67 % of always answering "no" (E2). So the app never grades the learner's
  dialect.
- **Feedback on the learner's answer comes from separate, focused checks.** Inside the role-play
  prompt Apertus passed almost every learner error as correct (0 of 19) and noticed 3 of 20
  misunderstandings. Short check calls that run in parallel find about two thirds of the errors
  (70B 13 of 20, 8B 12 of 19) without a single false alarm on correct sentences, Swiss Standard German
  included, and 18 of 20 misunderstandings (8B: 17 of 19), again without a false alarm (E4).
- **Whisper understands spoken Zurich German far better than Apertus' experimental audio input**
  (WER 30.9 % vs 88.8 % on 100 SwissDial clips; Apertus declined the task in 10 of them),
  so Whisper handles speech input.
- **A short Zurich style guide in the prompt** raises the share of Zurich forms (*hät, de, gsi*
  instead of *het, dr, gsy*) in Apertus' output from 61 % to 70 % (70B), but does not improve chrF
  against SwissDial (58.6 with and without it, E3); Bernese and Basel forms still slip in.

## Sovereign by design

Apertus is called through any OpenAI-compatible endpoint: a Swiss-hosted API (Public AI,
Infomaniak) or a self-hosted vLLM server. Speech recognition (Whisper) and speech synthesis
(Qwen3-TTS) are open-weights models baked into the container image, so the whole system can run
on-premise or air-gapped. The tutor's voice is synthetic: no real person's voice is cloned.

## Licence

Code: Apache-2.0. Documentation and test data: CC-BY-4.0 / CDLA-Permissive-2.0
(see the Hack Apertus terms).
