FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/backend /app/src/backend
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]