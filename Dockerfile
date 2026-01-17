FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if necessary (e.g. for build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition
COPY pyproject.toml .
# Copy entire source code
COPY src/ src/
COPY cli/ cli/
COPY configs/ configs/

# Install dependencies and the project itself
# --system installs into the system python, avoiding venv creation inside docker
RUN uv pip install --system .

# Set PYTHONPATH to include the current directory so 'cli' module is found
ENV PYTHONPATH=/app

# Default entrypoint
ENTRYPOINT ["python", "-m", "cli.main"]
CMD ["--help"]
