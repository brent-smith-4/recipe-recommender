"""
Pre-compute meal embeddings and BM25 index at Docker build time.

Usage:
    python scripts/build_embeddings.py

Output: data/cache.pkl

This script is run as a RUN step in the Dockerfile so the embeddings are baked
into the image. Container startup then loads from cache (~5s) instead of
fetching + re-encoding on every restart (~3 min).
"""

import asyncio
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import app as rec


async def build() -> None:
    print("Fetching meals from TheMealDB...")
    async with httpx.AsyncClient() as client:
        meals = await rec._fetch_all_meals(client)
    print(f"Fetched {len(meals)} meals.")

    rec._populate_indices(meals)

    print("Building MiniLM semantic index...")
    rec._build_semantic_index(meals)

    print("Building BM25 index...")
    rec._build_bm25_index(meals)

    cache = {
        "meals": meals,
        "semantic_embeddings": dict(rec.SEMANTIC_EMBEDDINGS),
        "semantic_matrix": rec.SEMANTIC_MATRIX,
        "semantic_meal_ids": rec.SEMANTIC_MEAL_IDS,
        "bm25_index": rec.BM25_INDEX,
        "bm25_meal_ids": rec.BM25_MEAL_IDS,
    }

    out = Path("data/cache.pkl")
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(cache, fh)

    size_mb = out.stat().st_size / 1_000_000
    print(f"Saved {out} ({size_mb:.1f} MB, {len(meals)} meals)")


if __name__ == "__main__":
    asyncio.run(build())
