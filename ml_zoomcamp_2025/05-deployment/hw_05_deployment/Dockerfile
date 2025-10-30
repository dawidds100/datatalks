FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

# Kopiuj pliki projektowe
COPY pyproject.toml uv.lock ./
COPY q6_app.py pipeline_v1.bin ./

# Zainstaluj uv, jeśli jeszcze nie ma
RUN pip install --no-cache-dir scikit-learn==1.6.1 fastapi uvicorn requests uv

# Zainstaluj zależności zgodnie z uv.lock
RUN uv sync

# Uruchom serwer FastAPI
CMD ["uvicorn", "q6_app:app", "--host", "0.0.0.0", "--port", "8000"]