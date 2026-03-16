# Mise v4 | ML-Powered Recipe Recommender

A multi-model recipe recommender combining **MPNet** (sentence transformer), **MobileNetV2** (CNN), **BM25**, and **stemmed ingredient matching** to find recipes by meaning, appearance, keywords, and what's in your kitchen.

Built with **FastAPI** + **PyTorch** + **TensorFlow** + **scikit-learn** + **NLTK** + **TheMealDB**.

---

## Features

- **Semantic search (MPNet)** | describe what you're craving in natural language and the transformer finds recipes by *meaning*, not just keywords
- **Visual similarity (MobileNetV2)** | pick a reference dish and find meals that *look* similar using 1280-d CNN embeddings
- **Keyword ranking (BM25)** | the same algorithm search engines use, with cooking-specific stop word removal
- **Stemmed ingredient matching** | "tomatoes" matches canned tomatoes, roma tomatoes, tomato paste, and cherry tomatoes using Porter stemming with a core-word heuristic
- **Cuisine & category filters** | narrow by region and meal type
- **7-axis combined scoring** | all signals weighted and merged into a single relevance score, fully tunable via API
- **Meal detail modal** | full recipe with visually similar, semantically similar, and keyword-similar neighbors side by side

---

## How the ML Works

### MPNet Sentence Transformer (768-d)

The primary text engine. Each recipe is converted into a natural-language description combining the name, cuisine, ingredients, tags, and instructions. These documents are encoded with **all-mpnet-base-v2**, the highest-quality general-purpose sentence transformer, producing 768-dimensional embeddings.

Unlike keyword matching, the transformer understands that "pasta bolognese" and "spaghetti bolognese" are the same concept, and that "fry garlic in oil" and "saute minced garlic with oil" mean the same thing. It was trained on over 1 billion sentence pairs to learn semantic similarity.

Free-text queries are encoded with the same model and compared via cosine similarity against all recipe embeddings.

### MobileNetV2 Image Embeddings (1280-d)

Every meal thumbnail is downloaded and passed through **MobileNetV2** (pretrained on ImageNet) with the classification head removed. The global average pooling layer outputs a 1280-dimensional feature vector capturing textures, shapes, colors, and visual patterns.

Cosine similarity between vectors finds visually similar dishes | a creamy soup will match other creamy soups, a charred grilled dish will find similar grilled foods.

### BM25 (Okapi)

A custom implementation of the BM25 ranking algorithm, the industry standard used by Elasticsearch and Solr. BM25 improves on TF-IDF with:

- **Term frequency saturation** (parameter k1=1.5) | mentioning "chicken" 20 times doesn't make a recipe 20x more relevant
- **Document length normalization** (parameter b=0.75) | a short focused recipe mentioning "garlic" once scores higher than a long recipe mentioning it once among 500 words

The index uses custom cooking-specific stop words (50+ terms like "add", "cook", "stir", "minutes") to prevent ubiquitous recipe verbs from diluting the signal.

### Stemmed Ingredient Matching

Ingredient matching uses NLTK's Porter Stemmer with a core-word heuristic:

- **Core word identification** | the last word in an ingredient phrase is the core ("baby plum **tomatoes**", "canned **tomatoes**", "plum **sauce**")
- **Core match required** | "baby plum tomatoes" matches "canned tomatoes" (core: tomato = tomato) but NOT "plum sauce" (core: tomato != sauce)
- **Modifier verification** | for multi-word ingredients, at least one modifier must also match. "Soy sauce" matches "dark soy sauce" (modifier "soy" found) but NOT "fish sauce" (modifier "soy" not found)

This prevents false positives like "plum" in "baby plum tomatoes" matching "plum sauce" while still broadly matching across all tomato variants.

### Scoring Weights

| Signal | Weight | Method |
|---|---|---|
| Semantic (MPNet) | 0.35 | Cosine similarity of 768-d sentence embeddings |
| Ingredients | 0.20 | Stemmed core-word matching with modifier verification |
| Image (CNN) | 0.15 | Cosine similarity of 1280-d MobileNetV2 embeddings |
| BM25 | 0.10 | Okapi BM25 keyword ranking with cooking stop words |
| Category | 0.10 | Exact match |
| Area/Cuisine | 0.10 | Exact match |
| TF-IDF | 0.00 | Disabled by default (available via API param) |

Weights are tunable via API query parameters. The semantic transformer carries the highest weight because it captures meaning rather than just keywords, while BM25 ensures exact name matches aren't buried under semantically similar results.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| API | FastAPI | Async REST API with auto-generated Swagger docs |
| Semantic search | all-mpnet-base-v2 (PyTorch) | 768-d sentence embeddings for meaning-based matching |
| Image similarity | MobileNetV2 (TensorFlow) | 1280-d CNN embeddings for visual matching |
| Keyword ranking | BM25 (custom implementation) | Search-engine-grade keyword scoring |
| Keyword fallback | TF-IDF (scikit-learn) | Statistical keyword baseline |
| Stemming | Porter Stemmer (NLTK) | Ingredient fuzzy matching |
| Data source | TheMealDB API | 598 meals, 877 ingredients |

---

## Project Structure

```
recipe-recommender/
├── app.py              # FastAPI + MPNet + MobileNetV2 + BM25 engine
├── templates/
│   └── index.html      # Frontend UI (frog-green theme)
├── static/
├── requirements.txt    # PyTorch, TensorFlow, sentence-transformers, NLTK
├── Procfile            # Render/Railway deployment
├── Dockerfile          # Container deployment
└── README.md
```

---

## Known Limitations & Future Work

### Current Limitations
- Single-word dish names not in the database (e.g., "minestrone") return poor results because the transformer lacks culinary domain knowledge to expand them into descriptive queries
- Abstract food concepts like "light" or "comforting" don't translate well because the model learned language from text, not from the experience of eating
- Visual similarity matches on photographic style rather than food content | two dishes from the same cuisine may look similar despite being different foods

### Planned Enhancements
- **LLM query expansion** | use an LLM to expand dish names into rich descriptions before semantic search ("minestrone" becomes "hearty Italian vegetable soup with tomato broth, beans, and pasta")
- **Structured embeddings** | separate embeddings for ingredients, cooking method, and cuisine context instead of one text blob
- **Cuisine proximity scoring** | encode regional relationships so Syrian and Egyptian dishes score as culturally related
- **Fine-tuned image model** | retrain on Food-101 for food-specific visual understanding
- **Collaborative filtering** | user accounts with favorites to enable "people who liked X also liked Y"
- **Learned ranking weights** | train on user click data to optimize the 7-axis scoring automatically

---

## License

MIT
