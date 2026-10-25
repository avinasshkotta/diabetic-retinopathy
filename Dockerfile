# ---- stage 1: builder ----------------------------------------------------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- stage 2: runtime ----------------------------------------------------
FROM python:3.12-slim
# opencv needs libgl; non-root user for ECS
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser

COPY --from=builder /install /usr/local
WORKDIR /app
COPY src/ src/
COPY pyproject.toml .

# weights: baked in by CI (docker build --build-arg) or mounted at runtime.
# MODEL_URI (e.g. models:/dr-effnetb0@production) overrides via MLflow registry.
ARG WEIGHTS_DIR=models
COPY ${WEIGHTS_DIR}/ models/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PORT=8080

USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["sh", "-c", "uvicorn dr_pipeline.serving:app --host 0.0.0.0 --port ${PORT}"]
