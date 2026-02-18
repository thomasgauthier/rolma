# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY . /app/

# Install DSPy from GitHub - PyPI release has broken RLM module (as of 2026-02-17)
RUN uv pip install "git+https://github.com/stanfordnlp/dspy" openinference-instrumentation-dspy openinference-instrumentation-litellm pyvis brotli rich

RUN curl -fsSL https://deno.land/install.sh | sh && \
    ln -s /root/.deno/bin/deno /usr/local/bin/deno

CMD ["uv", "run", "app.py"]