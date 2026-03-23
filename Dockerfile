FROM python:3.11.15-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/
WORKDIR /app
ENV API_HOST=localhost
ENV API_PORT=8000
ENV API_TIMEOUT=30
ENV PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY app/ ./
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]