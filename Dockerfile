
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies first for better build caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source and required project files
COPY . .

# Use the environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Backend API
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
