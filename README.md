# Mise — Recipe Recommender

**Live demo:** [mise-recipe-recommender.fly.dev](https://mise-recipe-recommender.fly.dev)

A hybrid recipe search system combining BM25 keyword matching and sentence-transformer semantic embeddings. Built with FastAPI, deployed on Fly.io.

## Features

- **Semantic search** — describe what you want in natural language; the transformer finds recipes by *meaning*, not just keywords
- **BM25 keyword ranking** — the same algorithm used by Elasticsearch, with cooking-specific stop word removal
- **Stemmed ingredient matching** — "tomatoes" matches canned tomatoes, roma tomatoes, tomato paste, and cherry tomatoes using Porter stemming with a core-word heuristic
- **Cuisine & category filters** — narrow by region and meal type
- **5-axis combined scoring** — all signals weighted and merged into a single relevance score, tunable via API query params
- **Meal detail view** — full recipe with semantically similar and keyword-similar neighbors

---

## ML Stack

| Signal | Weight | Method |
|---|---|---|
| Semantic (MiniLM) | 0.45 | Cosine similarity of 384-d sentence embeddings |
| Ingredient overlap | 0.20 | Stemmed core-word matching with modifier verification |
| BM25 | 0.15 | Okapi BM25 keyword ranking with cooking stop words |
| Category | 0.10 | Exact match |
| Area / Cuisine | 0.10 | Exact match |

### Sentence Transformer (all-MiniLM-L6-v2, 384-d)

Each recipe is converted into a natural-language description combining the name, cuisine, ingredients, tags, and instructions, then encoded with **all-MiniLM-L6-v2**. Free-text queries are encoded with the same model and compared via cosine similarity against all 598 recipe embeddings.

Embeddings are pre-computed at Docker build time and baked into the image so container startup loads from a pickle file (~5s) rather than re-encoding on every restart (~3 min).

### BM25 (Okapi)

A custom BM25 implementation (k1=1.5, b=0.75) with 50+ cooking-specific stop words. Improves on TF-IDF with term frequency saturation and document length normalization.

### Stemmed Ingredient Matching

Uses NLTK's Porter Stemmer with a core-word heuristic: the last word in an ingredient phrase is the core. "Soy sauce" matches "dark soy sauce" (modifier overlap) but not "fish sauce" (no modifier match). Single-word ingredients skip the modifier check.

---

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Semantic search | all-MiniLM-L6-v2 (PyTorch CPU) |
| Keyword ranking | BM25 (custom implementation) |
| Stemming | Porter Stemmer (NLTK) |
| Data source | TheMealDB API (598 meals) |
| Deployment | Fly.io |

---

## Project Structure

```
mise/
├── app.py                      # FastAPI app — scoring, routes, startup
├── scripts/
│   └── build_embeddings.py     # Run at build time to generate data/cache.pkl
├── data/
│   └── cache.pkl               # Pre-computed embeddings (generated, not committed)
├── templates/
│   └── index.html              # Frontend UI
├── static/
├── eval/
│   ├── harness.py              # Eval harness (Type 1/2/3 queries)
│   └── queries.json            # Semantic and hand-labeled eval queries
├── requirements.txt
├── Dockerfile
├── fly.toml
└── .dockerignore
```

---

## Deployment (Fly.io)

### First deploy

```bash
# Install flyctl and log in
fly auth login

# Create the app (only needed once)
fly launch --no-deploy

# Deploy — builds image, runs build_embeddings.py, starts machine
fly deploy
```

The first `fly deploy` takes ~8–10 minutes: pip install (~4 min) + embedding build (~3 min) + image push (~1 min). Subsequent deploys reuse the pip layer and take ~4–5 min.

### Machine configuration

`fly.toml` sets `auto_stop_machines = false` and `min_machines_running = 1` so the machine stays alive and there are no cold starts. The 256MB machine is tight with MiniLM — monitor memory after first deploy with:

```bash
fly metrics
```

### Local dev

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Option A: let the app fetch and build on startup (slow, ~3 min)
uvicorn app:app --reload

# Option B: pre-build the cache locally first (fast restarts after)
python scripts/build_embeddings.py
uvicorn app:app --reload
```

---

## Eval Results (v5, all-MiniLM-L6-v2)

| Metric | Score |
|---|---|
| Type 1 Recall@10 (ingredient recall, n=50) | 0.9600 |
| Type 1 MRR | 0.8502 |
| Type 2 Precision@5 (semantic + category GT, n=20) | 0.9294 |
| Type 2 Recall@10 | 0.4907 |

Run the eval harness:

```bash
python eval/harness.py
```
## Production Lessons

- **Initial sizing was wrong.** Deployed on 256MB based on local memory profiling, 
  which didn't account for PyTorch's inference-time allocations. First semantic queries 
  in production triggered OOM kills (anon-rss ~386MB at crash time).
- **Diagnosis:** Fly.io machine logs showed clear `Out of memory: Killed process` 
  entries correlated with `/api/recommend` requests carrying text queries. 
  Non-semantic endpoints (`/api/random`, ingredient-only search) worked fine.
- **Fix:** Scaled to 1024MB. Future optimization path: ONNX Runtime would reduce 
  inference memory footprint significantly and allow returning to a smaller tier.

---

## License

MIT
