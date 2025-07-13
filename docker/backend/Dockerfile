# Builder stage for compilation
FROM python:3.10 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libespeak-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir
RUN pip install numpy==2.2.6 --no-cache-dir  # Pre-install for aeneas
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
RUN pip install -r requirements.txt --no-deps --no-cache-dir

# Runtime stage, slim image
FROM python:3.10-slim
WORKDIR /app
# Install runtime deps for aeneas (ffmpeg, libespeak1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libespeak1 \
    && rm -rf /var/lib/apt/lists/*
# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy app code and configs
COPY src/backend /app/src/backend
COPY .env /app/.env
COPY cert.pem /app/cert.pem
COPY key.pem /app/key.pem
# Non-root user for security
RUN useradd -m appuser
USER appuser
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--ssl-certfile", "/app/cert.pem", "--ssl-keyfile", "/app/key.pem"]