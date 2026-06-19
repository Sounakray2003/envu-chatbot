FROM python:3.11-slim

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements-retrieval.txt
ARG SYSTEM_DEPS_PROFILE=retrieval

# Install system dependencies. Retrieval needs only certificates; ingestion
# opts into the full document-processing stack through docker-compose args.
RUN if [ "$SYSTEM_DEPS_PROFILE" = "full" ]; then \
        apt-get update && apt-get install -y \
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
            ca-certificates; \
    else \
        apt-get update && apt-get install -y ca-certificates; \
    fi && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements*.txt .
RUN if [ "$SYSTEM_DEPS_PROFILE" = "full" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi
RUN pip install --no-cache-dir --default-timeout=300 -r ${REQUIREMENTS_FILE}

# Download NLTK data only when the selected requirements include NLTK.
RUN python -c "import importlib.util; spec = importlib.util.find_spec('nltk'); spec and __import__('nltk').download('punkt', quiet=True)"

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8093

# Default to the retrieval API; docker-compose overrides this for ingestion.
CMD ["uvicorn", "retrieval_new:app", "--host", "0.0.0.0", "--port", "8093"]
