# Pipeline Optima™ — Docker Image
#
# Build:
#   docker build -t pipeline-optima .
#
# Run (single user, local):
#   docker run -p 8501:8501 pipeline-optima
#   Then open: http://localhost:8501
#
# Run (company server, background):
#   docker run -d --restart=unless-stopped -p 8501:8501 --name pipeline-optima pipeline-optima
#   Users access: http://YOUR-SERVER-IP:8501
#
# No Streamlit Cloud account needed. Runs 100% on your own infrastructure.

FROM python:3.11-slim

# Install OS-level dependencies for matplotlib, kaleido, numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and data
COPY pipeline_optimization_app.py .
COPY pipeline_model.py .
COPY hydraulic_check.py .
COPY baseline_engine.py .
COPY dra_analysis.py .
COPY dra_utils.py .
COPY linefill_utils.py .
COPY schedule_utils.py .
COPY generate_thesis.py .
COPY logo.png .
COPY secrets.toml .
COPY *.csv ./
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "pipeline_optimization_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false"]
