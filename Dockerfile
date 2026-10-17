# MedGuide AI — production image (Python 3.11 to match the dev venv).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
COPY templates/ ./templates/

EXPOSE 8000

# Conversation memory lives here; mount a volume at /data to persist it.
ENV MEDICAL_MEMORY_DB=/data/medical_memory.db

# API keys come from the environment (compose reads .env) — never baked in.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
