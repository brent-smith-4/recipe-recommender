# Mise v2 — ML-Powered Recipe Recommender

A recipe recommender powered by **MobileNetV2 image embeddings** and **TF-IDF text similarity**, plus ingredient matching and cuisine/category filtering.

Built with **FastAPI** + **TensorFlow** + **scikit-learn** + **TheMealDB**.

---

## Features

- **Ingredient matching** — enter what's in your kitchen, ranked by overlap
- **Cuisine & category filters** — narrow by region and meal type
- **Visual similarity (MobileNetV2)** — pick a reference dish, find meals that *look* similar using learned CNN features (1280-d embeddings)
- **Text similarity (TF-IDF)** — describe what you're craving in natural language, or find recipes with similar instructions/ingredients via TF-IDF with bigrams
- **Combined 5-axis scoring** — all signals weighted and merged into a single relevance score
- **Meal detail modal** — full recipe with visually similar *and* textually similar neighbors

## How the ML Works

### MobileNetV2 Image Embeddings

Instead of hand-crafted color histograms, we now run every meal thumbnail through **MobileNetV2** (pretrained on ImageNet) with the classification head removed. This produces a **1280-dimensional embedding** per image that captures high-level visual features — textures, shapes, colors, and food-like patterns the network learned from millions of images.

Similarity is computed via cosine distance between embedding vectors. This means "creamy white pasta" will match other creamy dishes, and "charred grilled meat" will find similar grilled foods — far beyond what raw color matching can achieve.

### TF-IDF Text Similarity

Each meal is converted into a text document combining:
- Ingredients and their measures
- Category and cuisine/area (double-weighted for emphasis)
- Tags
- Full cooking instructions

These documents are vectorized with **TF-IDF** using:
- Up to 5,000 features
- Unigram + bigram tokens (e.g., "olive oil", "stir fry")
- Sublinear TF scaling (log-dampened term frequency)
- English stop word removal

You can either pick a reference meal (and find meals with similar text profiles) or type a free-text query like "spicy chicken stir fry with noodles" and the system will TF-IDF-match it against all recipes.

### Scoring Weights

| Signal | Default Weight | Method |
|---|---|---|
| Ingredients | 0.30 | Fraction of user's ingredients found in recipe |
| Category | 0.15 | Exact match |
| Cuisine/Area | 0.15 | Exact match |
| Image (CNN) | 0.20 | Cosine similarity of MobileNetV2 embeddings |
| Text (TF-IDF) | 0.20 | Cosine similarity of TF-IDF vectors |

All weights are adjustable via API query params.

---

## Quick Start

```bash
cd recipe-recommender

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000**.

> **Startup time:** ~2-4 minutes on first run (fetches meals, downloads ImageNet weights, processes ~300 images through MobileNetV2). Subsequent starts with cached weights are faster.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Web UI |
| `GET /api/meta` | Categories, areas, ingredients, index stats |
| `GET /api/recommend` | Combined recommendation (see params below) |
| `GET /api/meal/{id}` | Full detail + visual & text neighbors |
| `GET /api/random?n=6` | Random meals |

### Recommend Params

```
/api/recommend?
  ingredients=chicken,garlic,rice
  &category=seafood
  &area=japanese
  &similar_to=52772
  &text_query=spicy noodle soup
  &limit=12
  &w_ingredient=0.3
  &w_category=0.15
  &w_area=0.15
  &w_image=0.2
  &w_text=0.2
```

---

## Deploying

### Render

1. Push to GitHub
2. New **Web Service** on [render.com](https://render.com)
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Set instance to at least **1 GB RAM** (TensorFlow needs it)

### Railway

1. Push to GitHub
2. Railway auto-detects `Procfile`
3. Ensure at least 1 GB memory

### Docker

```bash
docker build -t mise-recommender .
docker run -p 8000:8000 mise-recommender
```

> **Memory note:** TensorFlow + MobileNetV2 + 300 image embeddings requires ~800MB–1GB RAM. Use `tensorflow-cpu` (already in requirements) to avoid GPU dependencies.

---

## Project Structure

```
recipe-recommender/
├── app.py              # FastAPI + MobileNetV2 + TF-IDF engine
├── templates/
│   └── index.html      # Frontend UI
├── static/
├── requirements.txt    # Includes tensorflow-cpu, scikit-learn
├── Procfile
├── Dockerfile
└── README.md
```

---

## Future Enhancements

- **Fine-tune embeddings** on food-specific datasets (Food-101, Recipe1M)
- **Word2Vec / BERT embeddings** for richer text understanding
- **Collaborative filtering** with user accounts and favorites
- **Learned ranking model** trained on click/engagement data
- **Nutritional data** integration and dietary filtering

---

## License

MIT
