"""
Evaluation harness for Recipe Recommender v5.

Query types
-----------
Type 1 — Ingredient recall (auto-generated, ~50 queries)
    Samples recipes, uses their top-3 ingredients as the query string.
    Relevant set: the source recipe only.
    Metrics: Recall@10, MRR.

Type 2 — Semantic with programmatic ground truth (~20 queries)
    Hand-written natural-language queries (e.g. "spicy Thai noodles").
    Relevant set derived at runtime from strArea/strCategory/ingredient filters.
    Metrics: Precision@5, Recall@10.

Type 3 — Exact name match (auto-generated, default n=50, seed=42)
    Samples recipes and uses the exact recipe name as the query.
    Relevant set: the source recipe only.
    Metrics: Precision@1.

Type 4 — Human-style name queries (hand-labeled, queries.json: type4_human_style)
    Hand-picked recipes with queries written as a human would type them.
    Each entry has a single relevant_id. Stubs (empty relevant_ids) are skipped.
    Metrics: MRR.

Type 5 — Constraint & negation (hand-labeled, queries.json: type5_constraint)
    Vague, conjunctive, or negation queries (e.g. "creamy without dairy").
    Stubs (empty relevant_ids) are skipped with a count shown in the report.
    Metrics: Precision@5, Precision@10.

Usage
-----
    python eval/harness.py
    python eval/harness.py --w-semantic 0.50 --w-bm25 0.20 --w-ingredient 0.15 --w-category 0.10 --w-area 0.05
    python eval/harness.py --type1-n 30 --seed 7

Programmatic use (grid search)
-------------------------------
    from eval.harness import init_recommender, run_eval
    init_recommender()
    for ws, wb in [(0.4, 0.2), (0.5, 0.1)]:
        results = run_eval(weights={"semantic": ws, "bm25": wb, "ingredient": 0.20, "category": 0.10, "area": 0.10})
        print(ws, wb, results["type1"]["Recall@10"])
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as rec

QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"

DEFAULT_WEIGHTS = {
    "ingredient": 0.20,
    "category":   0.10,
    "area":        0.10,
    "semantic":    0.45,
    "bm25":        0.15,
}

_initialized = False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_recommender() -> None:
    """Fetch meal data and build all indices. Idempotent."""
    global _initialized
    if _initialized:
        return
    print("Initializing recommender (fetches ~598 meals, builds MPNet + BM25 indices)...")
    asyncio.run(rec.startup())
    print(f"Ready — {len(rec.MEALS)} meals loaded.\n")
    _initialized = True


# ---------------------------------------------------------------------------
# Scoring (direct port of recommend() without HTTP layer)
# ---------------------------------------------------------------------------

def _score_meals(
    query_text: str | None = None,
    ingredients: list[str] | None = None,
    category: str | None = None,
    area: str | None = None,
    weights: dict | None = None,
) -> list[str]:
    """Return all meal IDs ranked by fusion score, highest first."""
    w = weights or DEFAULT_WEIGHTS
    user_ings = {i.strip().lower() for i in (ingredients or []) if i.strip()}
    target_cat = (category or "").strip().lower()
    target_area = (area or "").strip().lower()

    sem_scores: dict[str, float] = (
        rec._semantic_query_similarity(query_text.strip()) if query_text else {}
    )

    bm25_scores: dict[str, float] = {}
    if query_text:
        bm25_scores = rec._bm25_query_scores(query_text.strip())
    elif user_ings:
        # Ingredient-only query: use joined ingredient string for BM25 signal
        bm25_scores = rec._bm25_query_scores(" ".join(sorted(user_ings)))

    scored: list[tuple[str, float]] = []
    for meal in rec.MEALS:
        mid = meal["idMeal"]
        s = 0.0

        if user_ings:
            meal_ings = rec._parse_ingredients(meal)
            if meal_ings:
                matched = 0
                for u_ing in user_ings:
                    u_words = [ww for ww in u_ing.split() if len(ww) > 1]
                    if not u_words:
                        continue
                    u_core = rec._stem_word(u_words[-1])
                    u_mods = {rec._stem_word(ww) for ww in u_words[:-1]}
                    for m_ing in meal_ings:
                        m_words = [ww for ww in m_ing.split() if len(ww) > 1]
                        if not m_words or rec._stem_word(m_words[-1]) != u_core:
                            continue
                        if not u_mods:
                            matched += 1
                            break
                        m_mods = {rec._stem_word(ww) for ww in m_words[:-1]}
                        if u_mods & m_mods or u_ing in m_ing or m_ing in u_ing:
                            matched += 1
                            break
                s += w["ingredient"] * (matched / len(user_ings))

        if target_cat and (meal.get("strCategory") or "").strip().lower() == target_cat:
            s += w["category"]
        if target_area and (meal.get("strArea") or "").strip().lower() == target_area:
            s += w["area"]

        s += w["semantic"] * sem_scores.get(mid, 0.0)
        s += w["bm25"] * bm25_scores.get(mid, 0.0)

        scored.append((mid, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in scored]


# ---------------------------------------------------------------------------
# Ground-truth helpers (Type 2)
# ---------------------------------------------------------------------------

def _relevant_for_filters(filters: dict) -> set[str]:
    """All meal IDs satisfying every filter (AND logic)."""
    result: set[str] = set()
    for meal in rec.MEALS:
        if "area" in filters:
            if (meal.get("strArea") or "").strip().lower() != filters["area"].lower():
                continue
        if "category" in filters:
            if (meal.get("strCategory") or "").strip().lower() != filters["category"].lower():
                continue
        if "require_ingredient" in filters:
            kw = filters["require_ingredient"].lower()
            if not any(kw in ing for ing in rec._parse_ingredients(meal)):
                continue
        result.add(meal["idMeal"])
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for rank, rid in enumerate(ranked, 1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Per-type runners
# ---------------------------------------------------------------------------

def run_type1(weights: dict, n: int = 50, seed: int = 42) -> dict:
    """
    Sample n recipes with >= 3 ingredients. Use their top-3 ingredients as both
    the text query (for semantic + BM25) and the ingredient filter (for overlap
    scoring). Relevant set is always the single source recipe.
    """
    rng = random.Random(seed)
    pool = [m for m in rec.MEALS if len(rec._parse_ingredients(m)) >= 3]
    sampled = rng.sample(pool, min(n, len(pool)))

    r10s: list[float] = []
    mrrs: list[float] = []
    for meal in sampled:
        ings = rec._parse_ingredients(meal)[:3]
        ranked = _score_meals(
            query_text=" ".join(ings),
            ingredients=ings,
            weights=weights,
        )
        rel = {meal["idMeal"]}
        r10s.append(_recall_at_k(ranked, rel, 10))
        mrrs.append(_mrr(ranked, rel))

    n_q = len(sampled)
    return {
        "n": n_q,
        "Recall@10": sum(r10s) / n_q if n_q else 0.0,
        "MRR":       sum(mrrs) / n_q if n_q else 0.0,
        "misses":    sum(1 for r in r10s if r == 0.0),
    }


def run_type2(queries: list[dict], weights: dict) -> dict:
    """
    For each query, compute the relevant set from its filters at runtime, then
    score all meals with the text query. Queries whose filters match nothing are
    skipped (reported as skipped_empty_gt).
    """
    p5s: list[float] = []
    r10s: list[float] = []
    skipped = 0

    for q in queries:
        relevant = _relevant_for_filters(q.get("filters", {}))
        if not relevant:
            skipped += 1
            continue
        ranked = _score_meals(query_text=q["query"], weights=weights)
        p5s.append(_precision_at_k(ranked, relevant, 5))
        r10s.append(_recall_at_k(ranked, relevant, 10))

    n_eval = len(p5s)
    return {
        "n": len(queries),
        "n_evaluated":     n_eval,
        "skipped_empty_gt": skipped,
        "Precision@5": sum(p5s)  / n_eval if n_eval else 0.0,
        "Recall@10":   sum(r10s) / n_eval if n_eval else 0.0,
    }


def run_type3(weights: dict, n: int = 50, seed: int = 42) -> dict:
    """
    Sample n recipes and query by exact recipe name. Relevant set is the
    source recipe only. Measures whether the system surfaces it at rank 1.
    """
    rng = random.Random(seed)
    sampled = rng.sample(rec.MEALS, min(n, len(rec.MEALS)))

    p1s: list[float] = []
    for meal in sampled:
        ranked = _score_meals(query_text=meal["strMeal"], weights=weights)
        p1s.append(_precision_at_k(ranked, {meal["idMeal"]}, 1))

    n_q = len(sampled)
    return {
        "n":           n_q,
        "seed":        seed,
        "Precision@1": sum(p1s) / n_q if n_q else 0.0,
        "misses":      sum(1 for p in p1s if p == 0.0),
    }


def run_type4(queries: list[dict], weights: dict) -> dict:
    """
    Hand-picked recipes with human-style name queries. Each entry should have
    a single relevant_id. Stubs (empty relevant_ids) are skipped.
    """
    mrrs: list[float] = []
    stubs = 0

    for q in queries:
        relevant = set(q.get("relevant_ids", []))
        if not relevant:
            stubs += 1
            continue
        ranked = _score_meals(query_text=q["query"], weights=weights)
        mrrs.append(_mrr(ranked, relevant))

    n_eval = len(mrrs)
    return {
        "n":             len(queries),
        "n_labeled":     n_eval,
        "stubs_skipped": stubs,
        "MRR":           sum(mrrs) / n_eval if n_eval else 0.0,
    }


def run_type5(queries: list[dict], weights: dict) -> dict:
    """
    Vague, conjunctive, or negation queries with hand-labeled relevant_ids.
    Stubs (empty relevant_ids) are skipped and counted separately.
    """
    p5s:  list[float] = []
    p10s: list[float] = []
    stubs = 0

    for q in queries:
        relevant = set(q.get("relevant_ids", []))
        if not relevant:
            stubs += 1
            continue
        ranked = _score_meals(query_text=q["query"], weights=weights)
        p5s.append(_precision_at_k(ranked, relevant, 5))
        p10s.append(_precision_at_k(ranked, relevant, 10))

    n_eval = len(p5s)
    return {
        "n":             len(queries),
        "n_labeled":     n_eval,
        "stubs_skipped": stubs,
        "Precision@5":  sum(p5s)  / n_eval if n_eval else 0.0,
        "Precision@10": sum(p10s) / n_eval if n_eval else 0.0,
    }


# ---------------------------------------------------------------------------
# Top-level eval entry point (importable for grid search)
# ---------------------------------------------------------------------------

def run_eval(
    weights: dict | None = None,
    type1_n: int = 50,
    type3_n: int = 50,
    seed: int = 42,
) -> dict:
    """
    Run all five query types and return a results dict.
    init_recommender() must be called before this.

    Returns:
        {
            "weights": {...},
            "type1":   {"n", "Recall@10", "MRR", "misses"},
            "type2":   {"n", "n_evaluated", "skipped_empty_gt", "Precision@5", "Recall@10"},
            "type3":   {"n", "seed", "Precision@1", "misses"},
            "type4":   {"n", "n_labeled", "stubs_skipped", "MRR"},
            "type5":   {"n", "n_labeled", "stubs_skipped", "Precision@5", "Precision@10"},
        }
    """
    w = weights or DEFAULT_WEIGHTS
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return {
        "weights": w,
        "type1": run_type1(w, n=type1_n, seed=seed),
        "type2": run_type2(queries.get("type2_semantic", []), w),
        "type3": run_type3(w, n=type3_n, seed=seed),
        "type4": run_type4(queries.get("type4_human_style", []), w),
        "type5": run_type5(queries.get("type5_constraint", []), w),
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    w  = results["weights"]
    t1 = results["type1"]
    t2 = results["type2"]
    t3 = results["type3"]
    t4 = results["type4"]
    t5 = results["type5"]
    sep = "=" * 64

    print(f"\n{sep}")
    print("  RECIPE RECOMMENDER v5 — EVAL REPORT")
    print(sep)
    print(
        f"  Weights  semantic={w['semantic']:.2f}  bm25={w['bm25']:.2f}"
        f"  ingredient={w['ingredient']:.2f}  category={w['category']:.2f}"
        f"  area={w['area']:.2f}"
    )
    print(sep)

    print(f"\n  TYPE 1 — Ingredient Recall  (n={t1['n']})")
    print(f"    Recall@10 : {t1['Recall@10']:.4f}")
    print(f"    MRR       : {t1['MRR']:.4f}")
    print(f"    Misses    : {t1['misses']} / {t1['n']}  (source not in top 10)")

    print(f"\n  TYPE 2 — Semantic + Category GT  "
          f"(n={t2['n']}, evaluated={t2['n_evaluated']}"
          + (f", skipped={t2['skipped_empty_gt']} empty GT" if t2["skipped_empty_gt"] else "")
          + ")")
    print(f"    Precision@5 : {t2['Precision@5']:.4f}")
    print(f"    Recall@10   : {t2['Recall@10']:.4f}")

    print(f"\n  TYPE 3 — Exact Name Match  "
          f"(auto-generated, n={t3['n']}, seed={t3['seed']})")
    print(f"    Precision@1 : {t3['Precision@1']:.4f}")
    print(f"    Misses      : {t3['misses']} / {t3['n']}  (source not at rank 1)")

    print(f"\n  TYPE 4 — Human-Style Name Queries  "
          f"(n={t4['n']}, labeled={t4['n_labeled']}"
          + (f", stubs={t4['stubs_skipped']}" if t4["stubs_skipped"] else "")
          + ")")
    if t4["n_labeled"] > 0:
        print(f"    MRR : {t4['MRR']:.4f}")
    else:
        print("    Fill in type4_human_style entries in eval/queries.json to activate.")

    print(f"\n  TYPE 5 — Constraint & Negation  "
          f"(n={t5['n']}, labeled={t5['n_labeled']}"
          + (f", stubs={t5['stubs_skipped']}" if t5["stubs_skipped"] else "")
          + ")")
    if t5["n_labeled"] > 0:
        print(f"    Precision@5  : {t5['Precision@5']:.4f}")
        print(f"    Precision@10 : {t5['Precision@10']:.4f}")
    else:
        print("    Fill in type5_constraint entries in eval/queries.json to activate.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recipe Recommender v5 eval harness")
    p.add_argument("--w-semantic",    type=float, default=DEFAULT_WEIGHTS["semantic"],    metavar="W")
    p.add_argument("--w-bm25",        type=float, default=DEFAULT_WEIGHTS["bm25"],        metavar="W")
    p.add_argument("--w-ingredient",  type=float, default=DEFAULT_WEIGHTS["ingredient"],  metavar="W")
    p.add_argument("--w-category",    type=float, default=DEFAULT_WEIGHTS["category"],    metavar="W")
    p.add_argument("--w-area",        type=float, default=DEFAULT_WEIGHTS["area"],        metavar="W")
    p.add_argument("--type1-n",       type=int,   default=50,   metavar="N",
                   help="Number of Type 1 queries to sample (default: 50)")
    p.add_argument("--seed",          type=int,   default=42,
                   help="RNG seed for Type 1 sampling (default: 42)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    weights = {
        "semantic":   args.w_semantic,
        "bm25":       args.w_bm25,
        "ingredient": args.w_ingredient,
        "category":   args.w_category,
        "area":       args.w_area,
    }
    init_recommender()
    results = run_eval(weights=weights, type1_n=args.type1_n, seed=args.seed)
    print_report(results)


if __name__ == "__main__":
    main()
