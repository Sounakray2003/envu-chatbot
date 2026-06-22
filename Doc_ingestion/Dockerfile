FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    libuuid1 \
    libffi8 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt \
    && find /usr/local -type d -name '__pycache__' -prune -exec rm -rf '{}' + \
    && find /usr/local -type f -name '*.pyc' -delete

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy only runtime files needed by the ingestion API.
COPY retrieval_new.py ./
COPY admin_api.py ./
COPY services ./services

# Temporary upload and registry workspaces used by the ingestion API.
RUN mkdir -p uploaded_files uploads file_registry

EXPOSE 8094

# Default to the old handoff-compatible admin API.
CMD ["uvicorn", "admin_api:app", "--host", "0.0.0.0", "--port", "8094"]
