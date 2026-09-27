# DevDesk API image: FastAPI server over the router agent, with the
# prebuilt Chroma index baked in (build it first: python -m devdesk.rag.ingest).
# Gemini embeddings only — the torch-based offline fallback is left out.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    PORT=8080

WORKDIR /app

# Dependencies first, so code edits don't bust this layer.
COPY pyproject.toml README.md ./
RUN mkdir -p src/devdesk && touch src/devdesk/__init__.py \
    && uv pip install --system . \
    && rm -rf src

COPY src ./src
RUN uv pip install --system --no-deps .

# Index is data, not code: tenant source repos aren't in the image, so
# queries hit this snapshot and git_status_lookup reports "no local repo".
COPY data/.chroma ./data/.chroma

RUN useradd --create-home --uid 10001 devdesk \
    && mkdir -p logs && chown -R devdesk:devdesk /app
USER devdesk

# config.REPO_ROOT resolves from the installed package's location, so point
# the data dir at /app explicitly.
ENV DEVDESK_HOME=/app
# stdout only: Cloud Run's filesystem is RAM, and stdout reaches Cloud Logging.
ENV DEVDESK_LOG_FILE=0

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn devdesk.server:app --host 0.0.0.0 --port ${PORT}"]
