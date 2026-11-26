FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.lock.txt requirements-api.lock.txt
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.lock.txt
COPY . .
RUN pip install --no-cache-dir --no-deps --no-build-isolation . \
    && useradd --uid 1000 --create-home appuser
USER appuser
ENV SAFEDRIVE_DEVICE=cpu SAFEDRIVE_MANIFEST=/app/models.json PYTHONUNBUFFERED=1
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3)"
CMD ["uvicorn", "safedrive.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--limit-concurrency", "16"]
