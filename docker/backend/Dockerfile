FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/backend /app/src/backend
COPY .env /app/.env
COPY cert.pem /app/cert.pem
COPY key.pem /app/key.pem
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--ssl-certfile", "/app/cert.pem", "--ssl-keyfile", "/app/key.pem"]