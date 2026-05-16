FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake embeddings into the image so startup loads from cache instead of
# re-fetching and re-encoding on every container restart.
RUN python scripts/build_embeddings.py

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
