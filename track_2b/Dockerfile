FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PYTHONPATH=/app/src \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MALLOC_ARENA_MAX=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg sox libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU build of PyTorch keeps the image small and runs anywhere (also linux/arm64 on Apple Silicon).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchaudio \
 && pip install -r requirements.txt

# Open-weights speech models are baked into the image: no download (and no memory peak from it) at
# first start, and at runtime the container needs nothing but the Apertus endpoint.
ARG ASR_MODEL=large-v3-turbo
ARG TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
RUN python -c "from faster_whisper.utils import download_model; download_model('${ASR_MODEL}')" \
 && python -c "from huggingface_hub import snapshot_download; snapshot_download('${TTS_MODEL}')"

COPY src/ src/
COPY data/ data/

EXPOSE 7860
CMD ["python", "-m", "losemal.app"]
