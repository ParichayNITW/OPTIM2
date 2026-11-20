FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 + bcrypt
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install backend requirements first (better layer caching)
COPY backend_requirements.txt /app/backend_requirements.txt
RUN pip install --no-cache-dir -r backend_requirements.txt

# Copy the backend package
COPY backend/ /app/backend/

# The solver lives at /solver (mounted read-only via docker-compose)
ENV SOLVER_PATH=/solver

EXPOSE 8000

# Default: run the API server
# Override CMD in docker-compose for the worker service
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
