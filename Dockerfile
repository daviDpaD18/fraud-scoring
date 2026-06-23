FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS prod
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8080
CMD ["uvicorn", "src.fraud_scoring.api:app", "--host", "0.0.0.0", "--port",  "8080"]