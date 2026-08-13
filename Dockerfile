# VectorPrism — Linux runtime (avoids Windows native DLL conflicts)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

# Default: show CLI help. Override with docker compose / docker run commands.
CMD ["python", "vectorprism.py", "--help"]
