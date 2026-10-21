FROM python:3.12.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/models
WORKDIR /app
COPY pyproject.toml README.md LICENSE setup.py requirements.lock.txt ./
COPY src ./src
COPY api.py app.py ./
RUN pip install --no-cache-dir -c requirements.lock.txt ".[serve,explain]" && useradd --create-home appuser
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
