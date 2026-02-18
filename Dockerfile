# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY . /app/

RUN uv pip install dspy openinference-instrumentation-dspy openinference-instrumentation-litellm pyvis brotli rich

CMD ["uv", "run", "app.py"]