"""
Recipe Recommender v3 -- FastAPI backend
ML Stack:
  - MobileNetV2 learned embeddings for image similarity (1280-d)
  - Sentence Transformers (all-MiniLM-L6-v2) for semantic text similarity (384-d)
  - TF-IDF as lightweight fallback / secondary signal
  - Ingredient overlap, category, and area matching
"""

import asyncio
import io
import logging
import os
import random
from collections import defaultdict
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ---------------------------------------------------------------------------
# Lazy-load heavy models
# ---------------------------------------------------------------------------
_mobilenet_model = None
_sentence_model = None


def _get_mobilenet():
    """Load MobileNetV2 once, stripping the classification head."""
    global _mobilenet_model
    if _mobilenet_model is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        base = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
        )
        base.trainable = False
        _mobilenet_model = base
        LOG.info("MobileNetV2 loaded (1280-d embedding layer)")
    return _mobilenet_model


def _get_sentence_model():
    """
    Load the sentence-transformer model once.
    all-MiniLM-L6-v2: 384-d embeddings, fast, good quality.
    Understands meaning, not just word overlap.
    """
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        LOG.info("Sentence Transformer loaded (all-MiniLM-L6-v2, 384-d)")
    return _sentence_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MEALDB_BASE = "https://www.themealdb.com/api/json/v1/1"
LOG = logging.getLogger("recommender")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Recipe Recommender v3", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# In-memory store (populated on startup)
# ---------------------------------------------------------------------------
MEALS: list[dict] = []
MEAL_INDEX: dict[str, dict] = {}
INGREDIENT_INDEX: dict[str, set] = defaultdict(set)
CATEGORY_INDEX: dict[str, set] = defaultdict(set)
AREA_INDEX: dict[str, set] = defaultdict(set)

# ML features
IMAGE_FEATURES: dict[str, np.ndarray] = {}        # idMeal -> 1280-d MobileNetV2
SEMANTIC_EMBEDDINGS: dict[str, np.ndarray] = {}    # idMeal -> 384-d sentence embedding
SEMANTIC_MATRIX: np.ndarray = None                  # (n_meals, 384) dense matrix
SEMANTIC_MEAL_IDS: list[str] = []                   # row index -> idMeal

# TF-IDF fallback
TFIDF_MATRIX = None
TFIDF_VECTORIZER: TfidfVectorizer = None
TFIDF_MEAL_IDS: list[str] = []


# ---------------------------------------------------------------------------
# Helpers: data fetching
# ---------------------------------------------------------------------------
def _parse_ingredients(meal: dict) -> list[str]:
    ingredients = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip().lower()
        if ing:
            ingredients.append(ing)
    return ingredients


def _build_text_document(meal: dict) -> str:
    """
    Build a natural-language description of the meal for the sentence
    transformer. Unlike TF-IDF, we want readable sentences because
    the transformer understands meaning and context.
    """
    name = (meal.get("strMeal") or "").strip()
    cat = (meal.get("strCategory") or "").strip()
    area = (meal.get("strArea") or "").strip()
    tags = (meal.get("strTags") or "").strip().replace(",", ", ")

    # Build a natural ingredient list
    ings = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        meas = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            ings.append(f"{meas} {ing}".strip())
    ing_text = ", ".join(ings) if ings else ""

    instructions = (meal.get("strInstructions") or "").strip()
    # Truncate instructions to first ~500 chars for embedding efficiency
    if len(instructions) > 500:
        instructions = instructions[:500] + "..."

    parts = []
    if name:
        parts.append(f"{name}.")
    if cat or area:
        origin = f"This is a {cat} dish" if cat else "This dish"
        if area:
            origin += f" from {area} cuisine"
        parts.append(origin + ".")
    if tags:
        parts.append(f"Tags: {tags}.")
    if ing_text:
        parts.append(f"Ingredients: {ing_text}.")
    if instructions:
        parts.append(instructions)

    return " ".join(parts)


def _build_tfidf_document(meal: dict) -> str:
    """Simpler bag-of-words document for TF-IDF fallback."""
    parts = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        meas = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            parts.append(f"{meas} {ing}".strip())
    cat = (meal.get("strCategory") or "").strip()
    area = (meal.get("strArea") or "").strip()
    if cat:
        parts.extend([cat, cat])
    if area:
        parts.extend([area, area])
    tags = (meal.get("strTags") or "").strip()
    if tags:
        parts.append(tags.replace(",", " "))
    instructions = (meal.get("strInstructions") or "").strip()
    if instructions:
        parts.append(instructions)
    return " ".join(parts)


async def _fetch_all_meals(client: httpx.AsyncClient) -> list[dict]:
    meals = []
    seen_ids = set()
    letters = "abcdefghijklmnopqrstuvwxyz"

    async def fetch_letter(letter: str):
        try:
            r = await client.get(
                f"{MEALDB_BASE}/search.php", params={"f": letter}, timeout=15
            )
            data = r.json()
            return data.get("meals") or []
        except Exception as e:
            LOG.warning(f"Failed to fetch letter {letter}: {e}")
            return []

    for i in range(0, len(letters), 6):
        batch = letters[i : i + 6]
        results = await asyncio.gather(*(fetch_letter(l) for l in batch))
        for meal_list in results:
            for m in meal_list:
                if m["idMeal"] not in seen_ids:
                    seen_ids.add(m["idMeal"])
                    meals.append(m)
        await asyncio.sleep(0.3)

    LOG.info(f"Fetched {len(meals)} unique meals from TheMealDB")
    return meals


# ---------------------------------------------------------------------------
# Image embeddings (MobileNetV2)
# ---------------------------------------------------------------------------
async def _download_image(client: httpx.AsyncClient, url: str) -> Optional[Image.Image]:
    try:
        r = await client.get(url + "/preview", timeout=10)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except Exception:
        pass
    return None


async def _build_image_index(meals: list[dict]):
    """Download thumbnails and extract MobileNetV2 embeddings."""
    LOG.info("Downloading thumbnails and extracting CNN embeddings...")

    images: dict[str, Image.Image] = {}
    async with httpx.AsyncClient() as client:
        for i in range(0, len(meals), 10):
            batch = meals[i : i + 10]
            tasks = []
            for m in batch:
                url = m.get("strMealThumb")
                if url:
                    tasks.append((m["idMeal"], _download_image(client, url)))
            results = await asyncio.gather(*(t[1] for t in tasks))
            for (mid, _), img in zip(tasks, results):
                if img is not None:
                    images[mid] = img
            await asyncio.sleep(0.15)

    LOG.info(f"Downloaded {len(images)} images, running through MobileNetV2...")

    model = _get_mobilenet()
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    batch_size = 16
    meal_ids = list(images.keys())
    for i in range(0, len(meal_ids), batch_size):
        batch_ids = meal_ids[i : i + batch_size]
        batch_arrays = []
        for mid in batch_ids:
            img = images[mid].resize((224, 224)).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            batch_arrays.append(arr)

        batch_np = np.stack(batch_arrays)
        batch_np = preprocess_input(batch_np)
        embeddings = model.predict(batch_np, verbose=0)

        for j, mid in enumerate(batch_ids):
            vec = embeddings[j].flatten()
            norm = np.linalg.norm(vec)
            IMAGE_FEATURES[mid] = vec / norm if norm > 0 else vec

    LOG.info(f"Built MobileNetV2 embeddings for {len(IMAGE_FEATURES)} meals (1280-d each)")


# ---------------------------------------------------------------------------
# Sentence Transformer embeddings (semantic text understanding)
# ---------------------------------------------------------------------------
def _build_semantic_index(meals: list[dict]):
    """
    Encode every meal's text description into a 384-d dense vector using
    a pretrained sentence transformer (all-MiniLM-L6-v2).

    Unlike TF-IDF which matches keywords:
      - "saute garlic in butter" vs "fry minced garlic with butter" = LOW match (different words)

    Sentence transformers understand meaning:
      - "saute garlic in butter" vs "fry minced garlic with butter" = HIGH match (same meaning)

    The model was trained on over 1 billion sentence pairs to learn that
    semantically similar text should have similar vector representations.
    """
    global SEMANTIC_MATRIX, SEMANTIC_MEAL_IDS

    model = _get_sentence_model()

    documents = []
    meal_ids = []
    for m in meals:
        doc = _build_text_document(m)
        if doc.strip():
            documents.append(doc)
            meal_ids.append(m["idMeal"])

    LOG.info(f"Encoding {len(documents)} meal descriptions with sentence transformer...")

    # Encode all documents in one batch (the library handles internal batching)
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize so dot product = cosine sim
    )

    # Store as dict and as matrix
    for i, mid in enumerate(meal_ids):
        SEMANTIC_EMBEDDINGS[mid] = embeddings[i]

    SEMANTIC_MATRIX = embeddings  # shape: (n_meals, 384)
    SEMANTIC_MEAL_IDS = meal_ids

    LOG.info(
        f"Built semantic index: {SEMANTIC_MATRIX.shape[0]} meals x "
        f"{SEMANTIC_MATRIX.shape[1]}-d embeddings"
    )


def _semantic_similarity(meal_id: str) -> dict[str, float]:
    """Compute semantic similarity between one meal and all others."""
    if SEMANTIC_MATRIX is None or meal_id not in SEMANTIC_EMBEDDINGS:
        return {}
    ref = SEMANTIC_EMBEDDINGS[meal_id].reshape(1, -1)
    sims = sklearn_cosine(ref, SEMANTIC_MATRIX).flatten()
    return {
        SEMANTIC_MEAL_IDS[i]: float(s)
        for i, s in enumerate(sims)
        if SEMANTIC_MEAL_IDS[i] != meal_id
    }


def _semantic_query_similarity(query_text: str) -> dict[str, float]:
    """
    Encode a free-text query with the same sentence transformer and
    compute cosine similarity against all meal embeddings.

    This is the key advantage over TF-IDF: the query "I want something
    warm and comforting with noodles" will match recipes about soups,
    ramen, and stews even if they don't contain the word "comforting".
    """
    if SEMANTIC_MATRIX is None:
        return {}
    model = _get_sentence_model()
    query_vec = model.encode(
        [query_text],
        normalize_embeddings=True,
    )
    sims = sklearn_cosine(query_vec, SEMANTIC_MATRIX).flatten()
    return {SEMANTIC_MEAL_IDS[i]: float(s) for i, s in enumerate(sims)}


# ---------------------------------------------------------------------------
# TF-IDF fallback index
# ---------------------------------------------------------------------------
def _build_tfidf_index(meals: list[dict]):
    """Build TF-IDF as a secondary/fallback signal."""
    global TFIDF_MATRIX, TFIDF_VECTORIZER, TFIDF_MEAL_IDS

    documents = []
    meal_ids = []
    for m in meals:
        doc = _build_tfidf_document(m)
        if doc.strip():
            documents.append(doc)
            meal_ids.append(m["idMeal"])

    TFIDF_VECTORIZER = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(documents)
    TFIDF_MEAL_IDS = meal_ids

    LOG.info(
        f"Built TF-IDF fallback: {TFIDF_MATRIX.shape[0]} docs x "
        f"{TFIDF_MATRIX.shape[1]} features"
    )


def _tfidf_similarity(meal_id: str) -> dict[str, float]:
    if TFIDF_MATRIX is None or meal_id not in TFIDF_MEAL_IDS:
        return {}
    idx = TFIDF_MEAL_IDS.index(meal_id)
    row = TFIDF_MATRIX[idx]
    sims = sklearn_cosine(row, TFIDF_MATRIX).flatten()
    return {
        TFIDF_MEAL_IDS[i]: float(s)
        for i, s in enumerate(sims)
        if TFIDF_MEAL_IDS[i] != meal_id
    }


def _tfidf_query_similarity(query_text: str) -> dict[str, float]:
    if TFIDF_MATRIX is None or TFIDF_VECTORIZER is None:
        return {}
    query_vec = TFIDF_VECTORIZER.transform([query_text])
    sims = sklearn_cosine(query_vec, TFIDF_MATRIX).flatten()
    return {TFIDF_MEAL_IDS[i]: float(s) for i, s in enumerate(sims)}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    async with httpx.AsyncClient() as client:
        meals = await _fetch_all_meals(client)

    for m in meals:
        mid = m["idMeal"]
        MEALS.append(m)
        MEAL_INDEX[mid] = m

        for ing in _parse_ingredients(m):
            INGREDIENT_INDEX[ing].add(mid)

        cat = (m.get("strCategory") or "").strip()
        if cat:
            CATEGORY_INDEX[cat.lower()].add(mid)

        area = (m.get("strArea") or "").strip()
        if area:
            AREA_INDEX[area.lower()].add(mid)

    # 1. Sentence transformer embeddings (primary text engine)
    _build_semantic_index(meals)

    # 2. TF-IDF fallback (lightweight, keyword-based)
    _build_tfidf_index(meals)

    # 3. Image embeddings (CNN)
    await _build_image_index(meals)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _meal_summary(meal: dict) -> dict:
    return {
        "id": meal["idMeal"],
        "name": meal["strMeal"],
        "category": meal.get("strCategory", ""),
        "area": meal.get("strArea", ""),
        "thumbnail": meal.get("strMealThumb", ""),
        "tags": meal.get("strTags", ""),
        "ingredients": _parse_ingredients(meal),
        "youtube": meal.get("strYoutube", ""),
        "source": meal.get("strSource", ""),
        "instructions": meal.get("strInstructions", ""),
    }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/meta")
async def meta():
    categories = sorted(CATEGORY_INDEX.keys())
    areas = sorted(AREA_INDEX.keys())
    ingredients = sorted(INGREDIENT_INDEX.keys())
    return {
        "categories": categories,
        "areas": areas,
        "ingredients": ingredients,
        "meal_count": len(MEALS),
        "image_index_size": len(IMAGE_FEATURES),
        "semantic_index_size": len(SEMANTIC_EMBEDDINGS),
        "tfidf_vocab_size": TFIDF_MATRIX.shape[1] if TFIDF_MATRIX is not None else 0,
    }


@app.get("/api/recommend")
async def recommend(
    ingredients: Optional[str] = Query(None, description="Comma-separated ingredients"),
    category: Optional[str] = Query(None),
    area: Optional[str] = Query(None),
    similar_to: Optional[str] = Query(None, description="Meal ID for visual similarity"),
    text_query: Optional[str] = Query(None, description="Free-text semantic search"),
    limit: int = Query(12, ge=1, le=50),
    w_ingredient: float = Query(0.25),
    w_category: float = Query(0.10),
    w_area: float = Query(0.10),
    w_image: float = Query(0.20),
    w_semantic: float = Query(0.25, description="Weight for sentence transformer similarity"),
    w_tfidf: float = Query(0.10, description="Weight for TF-IDF keyword similarity"),
):
    """
    Combined recommendation with 6 scoring axes:
      1. Ingredient overlap (set math)
      2. Category match (exact)
      3. Area/cuisine match (exact)
      4. MobileNetV2 image embedding similarity (CNN)
      5. Sentence transformer semantic similarity (transformer neural net)
      6. TF-IDF keyword similarity (statistical baseline)
    """
    user_ings = set()
    if ingredients:
        user_ings = {i.strip().lower() for i in ingredients.split(",") if i.strip()}

    target_cat = (category or "").strip().lower()
    target_area = (area or "").strip().lower()

    # Image similarity reference
    ref_img_feat = None
    if similar_to and similar_to in IMAGE_FEATURES:
        ref_img_feat = IMAGE_FEATURES[similar_to]

    # Semantic scores (sentence transformer -- primary)
    semantic_scores: dict[str, float] = {}
    if text_query and text_query.strip():
        semantic_scores = _semantic_query_similarity(text_query.strip())
    elif similar_to:
        semantic_scores = _semantic_similarity(similar_to)

    # TF-IDF scores (keyword fallback -- secondary)
    tfidf_scores: dict[str, float] = {}
    if text_query and text_query.strip():
        tfidf_scores = _tfidf_query_similarity(text_query.strip())
    elif similar_to:
        tfidf_scores = _tfidf_similarity(similar_to)

    scored: list[tuple[float, dict]] = []

    for meal in MEALS:
        mid = meal["idMeal"]
        score = 0.0

        # 1) Ingredient overlap
        if user_ings:
            meal_ings = set(_parse_ingredients(meal))
            if meal_ings:
                overlap = len(user_ings & meal_ings)
                ing_score = overlap / len(user_ings)
            else:
                ing_score = 0.0
            score += w_ingredient * ing_score

        # 2) Category match
        if target_cat:
            meal_cat = (meal.get("strCategory") or "").strip().lower()
            score += w_category * (1.0 if meal_cat == target_cat else 0.0)

        # 3) Area match
        if target_area:
            meal_area = (meal.get("strArea") or "").strip().lower()
            score += w_area * (1.0 if meal_area == target_area else 0.0)

        # 4) Image similarity (MobileNetV2 CNN)
        if ref_img_feat is not None and mid in IMAGE_FEATURES:
            sim = _cosine_sim(ref_img_feat, IMAGE_FEATURES[mid])
            score += w_image * sim

        # 5) Semantic text similarity (sentence transformer)
        if semantic_scores:
            sem_sim = semantic_scores.get(mid, 0.0)
            score += w_semantic * sem_sim

        # 6) TF-IDF keyword similarity (fallback)
        if tfidf_scores:
            tfidf_sim = tfidf_scores.get(mid, 0.0)
            score += w_tfidf * tfidf_sim

        if score > 0:
            scored.append((score, meal))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for s, m in scored[:limit]:
        summary = _meal_summary(m)
        summary["score"] = round(s, 4)
        results.append(summary)

    return {
        "count": len(results),
        "query": {
            "ingredients": list(user_ings) if user_ings else None,
            "category": target_cat or None,
            "area": target_area or None,
            "similar_to": similar_to,
            "text_query": text_query,
        },
        "results": results,
    }


@app.get("/api/meal/{meal_id}")
async def meal_detail(meal_id: str):
    meal = MEAL_INDEX.get(meal_id)
    if not meal:
        return {"error": "Meal not found"}
    summary = _meal_summary(meal)

    # Measures
    measures = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        meas = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            measures.append({"ingredient": ing, "measure": meas})
    summary["measures"] = measures

    # Visual neighbors (MobileNetV2 CNN)
    if meal_id in IMAGE_FEATURES:
        ref = IMAGE_FEATURES[meal_id]
        sims = []
        for other_id, feat in IMAGE_FEATURES.items():
            if other_id == meal_id:
                continue
            sims.append((other_id, _cosine_sim(ref, feat)))
        sims.sort(key=lambda x: x[1], reverse=True)
        summary["visual_neighbors"] = [
            {
                "id": oid,
                "name": MEAL_INDEX[oid]["strMeal"],
                "thumbnail": MEAL_INDEX[oid].get("strMealThumb", ""),
                "similarity": round(s, 4),
            }
            for oid, s in sims[:6]
        ]

    # Semantic neighbors (sentence transformer)
    sem_sims = _semantic_similarity(meal_id)
    if sem_sims:
        top_sem = sorted(sem_sims.items(), key=lambda x: x[1], reverse=True)[:6]
        summary["semantic_neighbors"] = [
            {
                "id": oid,
                "name": MEAL_INDEX[oid]["strMeal"],
                "thumbnail": MEAL_INDEX[oid].get("strMealThumb", ""),
                "similarity": round(s, 4),
            }
            for oid, s in top_sem
            if oid in MEAL_INDEX
        ]

    # TF-IDF neighbors (keyword-based)
    tfidf_sims = _tfidf_similarity(meal_id)
    if tfidf_sims:
        top_tfidf = sorted(tfidf_sims.items(), key=lambda x: x[1], reverse=True)[:6]
        summary["tfidf_neighbors"] = [
            {
                "id": oid,
                "name": MEAL_INDEX[oid]["strMeal"],
                "thumbnail": MEAL_INDEX[oid].get("strMealThumb", ""),
                "similarity": round(s, 4),
            }
            for oid, s in top_tfidf
            if oid in MEAL_INDEX
        ]

    return summary


@app.get("/api/random")
async def random_meals(n: int = Query(6, ge=1, le=20)):
    picks = random.sample(MEALS, min(n, len(MEALS)))
    return [_meal_summary(m) for m in picks]


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r") as f:
        return f.read()
