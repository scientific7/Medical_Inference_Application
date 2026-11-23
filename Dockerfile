FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /opt/inference
RUN groupadd --gid 10001 inference && useradd --uid 10001 --gid 10001 --create-home inference \
    && install -o inference -g inference -d /var/lib/inference

FROM base AS backend-dependencies
ARG FRAMEWORK_EXTRAS=""
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --prefix=/install ".[aws,postgres${FRAMEWORK_EXTRAS}]"

FROM base AS backend
COPY --from=backend-dependencies /install/ /usr/local/
COPY src/ ./src/
USER inference
ENV INFERENCE_DATABASE_PATH=/var/lib/inference/history.sqlite3 \
    INFERENCE_S3_CACHE_ROOT=/var/lib/inference/artifact-cache
EXPOSE 8000
CMD ["uvicorn", "medical_imaging_inference.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM base AS frontend-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --prefix=/install ".[frontend]"

FROM base AS frontend
COPY --from=frontend-dependencies /install/ /usr/local/
COPY src/ ./src/
COPY apps/frontend/ ./apps/frontend/
USER inference
EXPOSE 8501
CMD ["streamlit", "run", "apps/frontend/app.py", "--server.address=0.0.0.0", "--server.headless=true", "--server.maxUploadSize=20", "--browser.gatherUsageStats=false"]
