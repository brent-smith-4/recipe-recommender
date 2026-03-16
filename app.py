"""
Recipe Recommender v4 -- FastAPI backend
ML Stack:
  - MobileNetV2 learned embeddings for image similarity (1280-d CNN)
  - Sentence Transformer (all-mpnet-base-v2) for semantic text similarity (768-d)
  - BM25 (Okapi) for keyword ranking (replaces TF-IDF as primary keyword engine)
  - TF-IDF with tuned params + cooking stop words (secondary keyword signal)
  - Ingredient overlap, category, and area matching
"""

import asyncio
import io
import logging
import math
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
# Ingredient stemming
# ---------------------------------------------------------------------------
# We use Porter stemming + substring matching so that searching for
# "tomatoes" matches "canned tomatoes", "roma tomatoes", "tomato paste",
# "cherry tomatoes", etc. The stemmer reduces all of these to the root
# "tomato" for comparison.

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
except ImportError:
    _stemmer = None


def _stem_word(word: str) -> str:
    """Stem a single word to its root form."""
    if _stemmer:
        return _stemmer.stem(word.lower().strip())
    # Fallback: simple suffix stripping if nltk unavailable
    w = word.lower().strip()
    for suffix in ("es", "s", "ed", "ing"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[: -len(suffix)]
    return w


def _stem_phrase(phrase: str) -> set[str]:
    """Stem all words in a phrase and return as a set."""
    return {_stem_word(w) for w in phrase.split() if len(w) > 1}

# ---------------------------------------------------------------------------
# Cooking-specific stop words
# These appear in nearly every recipe and dilute signal for both
# TF-IDF and BM25. We remove them so distinctive terms stand out.
# ---------------------------------------------------------------------------
COOKING_STOP_WORDS = {
    "add", "added", "cook", "cooked", "cooking", "heat", "heated",
    "stir", "stirring", "mix", "mixed", "mixing", "place", "placed",
    "minutes", "minute", "hour", "hours", "serve", "serving",
    "remove", "removed", "cover", "covered", "set", "aside",
    "large", "small", "medium", "bowl", "pan", "pot", "dish",
    "oven", "degrees", "preheat", "preheated", "bake", "baking",
    "bring", "boil", "boiling", "simmer", "simmering",
    "cut", "chop", "chopped", "slice", "sliced", "dice", "diced",
    "drain", "rinse", "wash", "dry", "pat",
    "tablespoon", "tablespoons", "teaspoon", "teaspoons", "tbsp", "tsp",
    "cup", "cups", "ml", "oz", "lb", "lbs", "kg", "gram", "grams",
    "fresh", "freshly", "finely", "roughly", "well", "evenly",
    "top", "side", "ready", "needed", "using", "use",
    "turn", "reduce", "season", "taste", "water",
}

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

    UPGRADE: all-mpnet-base-v2 (768-d) replaces all-MiniLM-L6-v2 (384-d).

    MPNet is the highest-quality general-purpose sentence transformer:
      - 768-dimensional embeddings (vs 384 for MiniLM)
      - Trained on over 1 billion sentence pairs
      - Scores #1 on Semantic Textual Similarity benchmarks
      - ~2x slower than MiniLM but with 598 meals this is negligible

    The richer embedding space means it captures finer distinctions:
    MiniLM might put "grilled chicken" and "fried chicken" at similar
    distances, while MPNet better separates cooking methods.
    """
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-mpnet-base-v2")
        LOG.info("Sentence Transformer loaded (all-mpnet-base-v2, 768-d)")
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
app = FastAPI(title="Recipe Recommender v4", version="4.0.0")
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
SEMANTIC_EMBEDDINGS: dict[str, np.ndarray] = {}    # idMeal -> 768-d MPNet embedding
SEMANTIC_MATRIX: np.ndarray = None                  # (n_meals, 768) dense matrix
SEMANTIC_MEAL_IDS: list[str] = []                   # row index -> idMeal

# BM25 index
BM25_INDEX = None                                    # BM25Okapi instance
BM25_MEAL_IDS: list[str] = []

# TF-IDF (tuned, secondary)
TFIDF_MATRIX = None
TFIDF_VECTORIZER: TfidfVectorizer = None
TFIDF_MEAL_IDS: list[str] = []


# ---------------------------------------------------------------------------
# BM25 implementation (Okapi BM25)
# ---------------------------------------------------------------------------
class BM25Okapi:
    """
    Okapi BM25 -- the ranking function used by real search engines.

    BM25 improves on TF-IDF in two key ways:

    1. Term frequency saturation: In TF-IDF (even with sublinear_tf), more
       occurrences of a word keep increasing the score. BM25 has a saturation
       curve controlled by k1 -- after a few occurrences, additional ones
       barely matter. This prevents recipes that repeat "chicken" 20 times
       from dominating over recipes that mention it 3 times.

    2. Document length normalization: Controlled by parameter b. A short recipe
       description mentioning "garlic" once is more focused on garlic than a
       long recipe that mentions it once among 500 other words. BM25 accounts
       for this; raw TF-IDF doesn't do it as elegantly.

    Parameters:
      k1 = 1.5 (term frequency saturation; higher = less saturation)
      b  = 0.75 (length normalization; 0 = no normalization, 1 = full)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: dict[str, int] = {}     # term -> number of docs containing it
        self.idf: dict[str, float] = {}         # term -> IDF score
        self.doc_len: list[int] = []            # length of each document
        self.term_freqs: list[dict[str, int]] = []  # per-doc term frequency

    def fit(self, tokenized_corpus: list[list[str]]):
        """Build the BM25 index from a list of tokenized documents."""
        self.corpus_size = len(tokenized_corpus)
        self.doc_len = []
        self.term_freqs = []

        # Count document frequencies
        df: dict[str, int] = defaultdict(int)
        for doc in tokenized_corpus:
            self.doc_len.append(len(doc))
            tf: dict[str, int] = defaultdict(int)
            seen = set()
            for token in doc:
                tf[token] += 1
                if token not in seen:
                    df[token] += 1
                    seen.add(token)
            self.term_freqs.append(dict(tf))

        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 1.0
        self.doc_freqs = dict(df)

        # Compute IDF using the standard BM25 formula
        # IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

        return self

    def score(self, query_tokens: list[str]) -> np.ndarray:
        """Score all documents against a tokenized query."""
        scores = np.zeros(self.corpus_size)

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for i in range(self.corpus_size):
                tf = self.term_freqs[i].get(token, 0)
                if tf == 0:
                    continue
                dl = self.doc_len[i]
                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (numerator / denominator)

        return scores

    def score_normalized(self, query_tokens: list[str]) -> np.ndarray:
        """Score and normalize to 0-1 range for combining with other signals."""
        scores = self.score(query_tokens)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        return scores


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


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Simple tokenizer for BM25: lowercase, split on non-alpha,
    remove cooking stop words and short tokens.
    """
    import re
    tokens = re.findall(r"[a-z]+", text.lower())
    return [
        t for t in tokens
        if len(t) > 2 and t not in COOKING_STOP_WORDS
    ]


def _build_text_document(meal: dict) -> str:
    """
    Build a natural-language description for the sentence transformer.
    Readable sentences because the transformer understands context.
    """
    name = (meal.get("strMeal") or "").strip()
    cat = (meal.get("strCategory") or "").strip()
    area = (meal.get("strArea") or "").strip()
    tags = (meal.get("strTags") or "").strip().replace(",", ", ")

    ings = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        meas = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            ings.append(f"{meas} {ing}".strip())
    ing_text = ", ".join(ings) if ings else ""

    instructions = (meal.get("strInstructions") or "").strip()
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


def _build_keyword_document(meal: dict) -> str:
    """Full text for BM25 and TF-IDF keyword matching."""
    parts = []
    name = (meal.get("strMeal") or "").strip()
    if name:
        # Repeat name for boosting (important for name-based searches)
        parts.extend([name, name, name])
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
# Sentence Transformer embeddings (all-mpnet-base-v2)
# ---------------------------------------------------------------------------
def _build_semantic_index(meals: list[dict]):
    """
    Encode every meal's text into a 768-d dense vector using MPNet.
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

    LOG.info(f"Encoding {len(documents)} meal descriptions with MPNet...")

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for i, mid in enumerate(meal_ids):
        SEMANTIC_EMBEDDINGS[mid] = embeddings[i]

    SEMANTIC_MATRIX = embeddings
    SEMANTIC_MEAL_IDS = meal_ids

    LOG.info(
        f"Built semantic index: {SEMANTIC_MATRIX.shape[0]} meals x "
        f"{SEMANTIC_MATRIX.shape[1]}-d embeddings"
    )


def _semantic_similarity(meal_id: str) -> dict[str, float]:
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
    if SEMANTIC_MATRIX is None:
        return {}
    model = _get_sentence_model()
    query_vec = model.encode([query_text], normalize_embeddings=True)
    sims = sklearn_cosine(query_vec, SEMANTIC_MATRIX).flatten()
    return {SEMANTIC_MEAL_IDS[i]: float(s) for i, s in enumerate(sims)}


# ---------------------------------------------------------------------------
# BM25 index (Okapi BM25 -- what search engines use)
# ---------------------------------------------------------------------------
def _build_bm25_index(meals: list[dict]):
    """
    Build BM25 index from tokenized recipe documents.
    BM25 is the industry standard for keyword search ranking,
    used by Elasticsearch, Solr, and most production search engines.
    """
    global BM25_INDEX, BM25_MEAL_IDS

    tokenized_docs = []
    meal_ids = []
    for m in meals:
        doc = _build_keyword_document(m)
        tokens = _tokenize_for_bm25(doc)
        if tokens:
            tokenized_docs.append(tokens)
            meal_ids.append(m["idMeal"])

    BM25_INDEX = BM25Okapi(k1=1.5, b=0.75).fit(tokenized_docs)
    BM25_MEAL_IDS = meal_ids

    LOG.info(
        f"Built BM25 index: {len(meal_ids)} docs, "
        f"{len(BM25_INDEX.doc_freqs)} unique terms"
    )


def _bm25_query_scores(query_text: str) -> dict[str, float]:
    """Score all meals against a query using BM25."""
    if BM25_INDEX is None:
        return {}
    tokens = _tokenize_for_bm25(query_text)
    if not tokens:
        return {}
    scores = BM25_INDEX.score_normalized(tokens)
    return {BM25_MEAL_IDS[i]: float(s) for i, s in enumerate(scores)}


def _bm25_meal_scores(meal_id: str) -> dict[str, float]:
    """Find similar meals using BM25 on the meal's own text as query."""
    meal = MEAL_INDEX.get(meal_id)
    if not meal or BM25_INDEX is None:
        return {}
    # Use meal name + ingredients as the query
    name = (meal.get("strMeal") or "").strip()
    ings = " ".join(_parse_ingredients(meal))
    query = f"{name} {ings}"
    tokens = _tokenize_for_bm25(query)
    if not tokens:
        return {}
    scores = BM25_INDEX.score_normalized(tokens)
    return {
        BM25_MEAL_IDS[i]: float(s)
        for i, s in enumerate(scores)
        if BM25_MEAL_IDS[i] != meal_id
    }


# ---------------------------------------------------------------------------
# TF-IDF (tuned parameters, secondary signal)
# ---------------------------------------------------------------------------
def _build_tfidf_index(meals: list[dict]):
    """
    TF-IDF with tuned parameters for this specific dataset:
      - max_features=3000 (reduced from 5000; 598 recipes don't need 5k)
      - min_df=1 (was 2; lets rare distinctive terms like "szechuan" contribute)
      - ngram_range=(1,2) kept for bigrams like "olive oil"
      - Custom stop words: English defaults + cooking-specific terms
    """
    global TFIDF_MATRIX, TFIDF_VECTORIZER, TFIDF_MEAL_IDS

    # Merge sklearn's English stop words with our cooking stop words
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined_stops = list(ENGLISH_STOP_WORDS | COOKING_STOP_WORDS)

    documents = []
    meal_ids = []
    for m in meals:
        doc = _build_keyword_document(m)
        if doc.strip():
            documents.append(doc)
            meal_ids.append(m["idMeal"])

    TFIDF_VECTORIZER = TfidfVectorizer(
        max_features=3000,           # tuned down for small corpus
        stop_words=combined_stops,   # English + cooking stop words
        ngram_range=(1, 2),          # unigrams + bigrams
        min_df=1,                    # allow rare but distinctive terms
        sublinear_tf=True,           # log-scaled term frequency
    )
    TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(documents)
    TFIDF_MEAL_IDS = meal_ids

    LOG.info(
        f"Built TF-IDF index: {TFIDF_MATRIX.shape[0]} docs x "
        f"{TFIDF_MATRIX.shape[1]} features (tuned)"
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

    # 1. Sentence transformer (MPNet, primary semantic engine)
    _build_semantic_index(meals)

    # 2. BM25 (primary keyword engine)
    _build_bm25_index(meals)

    # 3. TF-IDF (tuned, secondary keyword signal)
    _build_tfidf_index(meals)

    # 4. Image embeddings (CNN)
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
        "semantic_dim": SEMANTIC_MATRIX.shape[1] if SEMANTIC_MATRIX is not None else 0,
        "bm25_vocab_size": len(BM25_INDEX.doc_freqs) if BM25_INDEX else 0,
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
    w_ingredient: float = Query(0.20),
    w_category: float = Query(0.10),
    w_area: float = Query(0.10),
    w_image: float = Query(0.15),
    w_semantic: float = Query(0.35, description="Sentence transformer (meaning)"),
    w_bm25: float = Query(0.10, description="BM25 keyword ranking"),
    w_tfidf: float = Query(0.00, description="TF-IDF keyword fallback"),
):
    """
    Combined recommendation with 7 scoring axes:
      1. Ingredient overlap (set math)
      2. Category match (exact)
      3. Area/cuisine match (exact)
      4. MobileNetV2 image embedding similarity (CNN)
      5. MPNet sentence transformer (semantic meaning)
      6. BM25 keyword ranking (search engine standard)
      7. TF-IDF keyword similarity (statistical fallback)
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

    # Semantic scores (MPNet -- primary meaning engine)
    semantic_scores: dict[str, float] = {}
    if text_query and text_query.strip():
        semantic_scores = _semantic_query_similarity(text_query.strip())
    elif similar_to:
        semantic_scores = _semantic_similarity(similar_to)

    # BM25 scores (primary keyword engine)
    bm25_scores: dict[str, float] = {}
    if text_query and text_query.strip():
        bm25_scores = _bm25_query_scores(text_query.strip())
    elif similar_to:
        bm25_scores = _bm25_meal_scores(similar_to)

    # TF-IDF scores (secondary keyword fallback)
    tfidf_scores: dict[str, float] = {}
    if text_query and text_query.strip():
        tfidf_scores = _tfidf_query_similarity(text_query.strip())
    elif similar_to:
        tfidf_scores = _tfidf_similarity(similar_to)

    scored: list[tuple[float, dict]] = []

    for meal in MEALS:
        mid = meal["idMeal"]
        score = 0.0

        # 1) Ingredient overlap (stemmed + substring matching)
        #    Strategy: The LAST word is the core ingredient.
        #    If the user's ingredient has modifiers (multi-word like "soy sauce"),
        #    at least one modifier must also appear in the meal ingredient.
        #    This prevents "soy sauce" from matching "fish sauce" or "plum sauce."
        #    Single-word ingredients ("garlic") skip modifier check.
        if user_ings:
            meal_ings = _parse_ingredients(meal)
            if meal_ings:
                matched = 0
                for user_ing in user_ings:
                    user_words = [w for w in user_ing.split() if len(w) > 1]
                    if not user_words:
                        continue
                    user_core = _stem_word(user_words[-1])
                    user_modifiers = {_stem_word(w) for w in user_words[:-1]}

                    for meal_ing in meal_ings:
                        meal_words = [w for w in meal_ing.split() if len(w) > 1]
                        if not meal_words:
                            continue
                        meal_core = _stem_word(meal_words[-1])

                        # Core must match first
                        if user_core != meal_core:
                            continue

                        # If user ingredient is single word (e.g. "garlic"),
                        # core match is enough
                        if not user_modifiers:
                            matched += 1
                            break

                        # Multi-word: at least one modifier must also match
                        # "soy sauce" vs "dark soy sauce" -> "soy" matches
                        # "soy sauce" vs "fish sauce" -> no modifier overlap
                        meal_all_stems = {_stem_word(w) for w in meal_words[:-1]}
                        if user_modifiers & meal_all_stems:
                            matched += 1
                            break

                        # Fallback: raw substring check with core already matched
                        if user_ing in meal_ing or meal_ing in user_ing:
                            matched += 1
                            break

                ing_score = matched / len(user_ings)
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

        # 5) Semantic similarity (MPNet transformer)
        if semantic_scores:
            sem_sim = semantic_scores.get(mid, 0.0)
            score += w_semantic * sem_sim

        # 6) BM25 keyword score
        if bm25_scores:
            bm25_sim = bm25_scores.get(mid, 0.0)
            score += w_bm25 * bm25_sim

        # 7) TF-IDF keyword score
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

    # Semantic neighbors (MPNet transformer)
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

    # BM25 neighbors (keyword ranking)
    bm25_sims = _bm25_meal_scores(meal_id)
    if bm25_sims:
        top_bm25 = sorted(bm25_sims.items(), key=lambda x: x[1], reverse=True)[:6]
        summary["bm25_neighbors"] = [
            {
                "id": oid,
                "name": MEAL_INDEX[oid]["strMeal"],
                "thumbnail": MEAL_INDEX[oid].get("strMealThumb", ""),
                "similarity": round(s, 4),
            }
            for oid, s in top_bm25
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
