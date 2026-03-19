# 07 — Embeddings & Vector Search: From Zero to Production

---

## 1. The Problem Computers Have With Words

Computers are great with numbers but bad with language. Everything in a computer is ultimately numbers. So how do we represent the *meaning* of words as numbers?

### Old Approach: One-Hot Encoding (Why It Fails)

```
Vocabulary: [cat, dog, car, tree] — 4 words

One-hot representation:
  cat  → [1, 0, 0, 0]
  dog  → [0, 1, 0, 0]
  car  → [0, 0, 1, 0]
  tree → [0, 0, 0, 1]

Problems:
  1. SPARSE: Mostly zeros. Huge vocabulary (50K words) = 50K-dim vector!
  2. NO MEANING: cat and dog look completely different
                 Distance(cat, dog) = sqrt(2) ← same as Distance(cat, car)
                 But cat and dog are similar (both are pets)!
  3. NO RELATIONSHIPS: "king" and "queen" look unrelated
                       "Paris" and "France" look unrelated
```

### New Approach: Dense Embeddings (Why They Work)

```
Embedding representation:
  cat  → [0.21, -0.08, 0.64, 0.15, 0.33, ...]   (768 numbers)
  dog  → [0.19, -0.07, 0.62, 0.14, 0.31, ...]   (768 numbers)
  car  → [-0.82, 0.45, -0.23, 0.91, 0.12, ...]  (768 numbers)
  tree → [-0.45, 0.20, -0.11, 0.55, 0.05, ...]  (768 numbers)

Benefits:
  1. DENSE: Every dimension has a value (not mostly zeros)
  2. MEANING: cat and dog are SIMILAR (vectors are close)
             Distance(cat, dog) = 0.12 ← much smaller!
             Distance(cat, car) = 0.89 ← much larger!
  3. RELATIONSHIPS: mathematical relationships preserved
```

---

## 2. The GPS Coordinate Analogy

Think of embedding space like a map, but with 768 dimensions instead of 2.

```
2D Map (simplified):
  Latitude, Longitude
  New York: (40.7, -74.0)
  London:   (51.5,  -0.1)
  Tokyo:    (35.7, 139.7)

  Distance(New York, London) = 5,570 km  ← closer
  Distance(New York, Tokyo) = 10,836 km  ← farther

768D "Meaning Map" (embedding space):
  "cat":    [0.21, -0.08, 0.64, ...]
  "kitten": [0.20, -0.07, 0.65, ...]   ← similar to cat
  "car":    [-0.82, 0.45, -0.23, ...]  ← very different

  Similarity(cat, kitten) ≈ 0.97  ← almost identical meaning
  Similarity(cat, car)    ≈ 0.21  ← very different meaning

Just as GPS coordinates tell you WHERE something is physically,
embedding coordinates tell you WHERE something is in MEANING SPACE.
```

---

## 3. How Embedding Models Are Trained

Embedding models are trained with **contrastive learning** — the model learns which texts should be similar and which should be different.

```
TRAINING APPROACH: Contrastive Learning

Positive pairs (similar, should have close vectors):
  "I love my dog" ↔ "My puppy is adorable"
  "Python is great for AI" ↔ "Machine learning with Python"
  "Return within 30 days" ↔ "Can I send this back in a month?"

Negative pairs (different, should have distant vectors):
  "I love my dog" ↔ "The stock market fell today"
  "Python is great for AI" ↔ "Make me a sandwich"

Training objective:
  Minimize distance between positive pairs
  Maximize distance between negative pairs

Result: Model learns to encode MEANING, not just surface text
```

---

## 4. Cosine Similarity — The Standard Metric

Cosine similarity measures the angle between two vectors. It's the most common way to compare embeddings.

```
Why cosine (not Euclidean distance)?
  Euclidean distance depends on vector LENGTH as well as direction.
  Cosine similarity only looks at the ANGLE (direction).

  If we scale "cat" vector by 2: [0.42, -0.16, 1.28, ...]
  Cosine similarity with original "cat" = 1.0 (same direction!)
  Euclidean distance = large (different magnitude)

  For normalized embeddings (length=1): cosine sim = dot product

Formula:
  cos(θ) = (A · B) / (|A| × |B|)
         = sum(aᵢ × bᵢ) / sqrt(sum(aᵢ²)) × sqrt(sum(bᵢ²))

Range:
   1.0 = identical direction (very similar meaning)
   0.0 = perpendicular (unrelated)
  -1.0 = opposite direction (opposite meaning)
```

```python
import numpy as np
from openai import OpenAI

client = OpenAI()


def get_embedding(text: str) -> list[float]:
    """
    Convert text to an embedding vector using OpenAI.

    The model text-embedding-3-small produces vectors with 1536 dimensions.
    Each dimension is a floating point number capturing some aspect of meaning.
    """
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],   # Clean newlines
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns a value between -1 and 1:
      ~1.0  = very similar meanings
      ~0.0  = unrelated
      ~-1.0 = opposite meanings (rare in practice)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    # Dot product = sum of element-wise products
    dot_product = np.dot(a, b)

    # Magnitudes = length of each vector
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Prevent division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return float(dot_product / (magnitude_a * magnitude_b))


# Test semantic similarity
pairs = [
    ("I love my dog", "My puppy is adorable"),         # Very similar
    ("Python is for AI", "JavaScript for web dev"),    # Different domains
    ("The cat sat", "A feline rested"),                # Paraphrase
    ("Buy tickets now", "Purchase passes today"),      # Paraphrase
    ("Hot coffee", "Cold ice cream"),                  # Semantic opposites
]

print(f"{'Text A':<30} {'Text B':<30} {'Similarity':>10}")
print("-" * 75)
for text_a, text_b in pairs:
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)
    sim = cosine_similarity(emb_a, emb_b)
    print(f"{text_a:<30} {text_b:<30} {sim:>10.3f}")

# Expected output:
# I love my dog          My puppy is adorable             0.876
# Python is for AI       JavaScript for web dev           0.712
# The cat sat            A feline rested                  0.891
# Buy tickets now        Purchase passes today            0.924
# Hot coffee             Cold ice cream                   0.412
```

---

## 5. The King - Man + Woman = Queen Example

One of the most famous properties of word embeddings is that they encode relationships that support vector arithmetic.

```python
# This is a classic demonstration of word vector relationships
# Works best with Word2Vec or GloVe; approximate with sentence embeddings

def demonstrate_word_arithmetic():
    """
    Show that embeddings capture relationships, not just similarity.

    The classic example:
      king - man + woman ≈ queen

    This works because the vector from "man" to "king" encodes
    the concept of "royalty". Adding that concept to "woman" gives "queen".

    Other examples:
      Paris - France + Germany ≈ Berlin
      Tokyo - Japan + China   ≈ Beijing
      doctor - man + woman    ≈ nurse (reveals gender bias!)
    """
    # Get embeddings
    words = ["king", "man", "woman", "queen", "prince", "princess"]
    embeddings = {word: np.array(get_embedding(word)) for word in words}

    # Compute: king - man + woman
    result_vector = embeddings["king"] - embeddings["man"] + embeddings["woman"]

    # Find which word is closest to the result
    similarities = {}
    for word, vec in embeddings.items():
        if word not in ("king", "man", "woman"):  # Exclude input words
            sim = cosine_similarity(result_vector.tolist(), vec.tolist())
            similarities[word] = sim

    print("king - man + woman ≈ ?")
    for word, sim in sorted(similarities.items(), key=lambda x: -x[1]):
        print(f"  {word}: {sim:.3f}")
    # queen should have highest similarity!
```

---

## 6. OpenAI Embedding Models

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── text-embedding-3-small (recommended default) ──
def embed_small(texts: list[str]) -> list[list[float]]:
    """
    Best balance of cost and quality for most use cases.

    Specs:
      Dimensions: 1536
      Max input: 8191 tokens (~6000 words)
      Price: $0.02 per million tokens
      Speed: Fast

    When to use:
      - RAG systems
      - Semantic search
      - Clustering
      - Most general use cases
    """
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]


# ── text-embedding-3-large ──
def embed_large(texts: list[str]) -> list[list[float]]:
    """
    Higher quality for tasks that need the best accuracy.

    Specs:
      Dimensions: 3072 (2x as many dimensions!)
      Max input: 8191 tokens
      Price: $0.13 per million tokens (6.5x more expensive)
      Speed: Slower

    When to use:
      - When small is not accurate enough
      - High-stakes retrieval (legal, medical)
      - Multilingual tasks (better cross-language)
    """
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-large"
    )
    return [item.embedding for item in response.data]


# ── Dimension reduction (unique to 3rd-gen models) ──
def embed_reduced_dimensions(texts: list[str], dimensions: int = 256) -> list[list[float]]:
    """
    OpenAI's 3rd-gen models support dimension reduction.
    Use fewer dimensions for lower cost/faster search, trading accuracy.

    dimensions can be anything from 1 to full size (1536 or 3072).
    Rule of thumb: 256 dims = ~80% quality at 1/6 the storage cost.
    """
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
        dimensions=dimensions
    )
    return [item.embedding for item in response.data]
```

---

## 7. Sentence Transformers (Local, Free)

```python
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util
import torch


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a local embedding model from HuggingFace.
    Downloads automatically on first use.

    Popular models:
    ┌─────────────────────────────────┬──────┬────────┬──────────────────────┐
    │ Model                           │ Dims │ Size   │ Notes                │
    ├─────────────────────────────────┼──────┼────────┼──────────────────────┤
    │ all-MiniLM-L6-v2                │  384 │  80MB  │ Fast, good quality   │
    │ all-mpnet-base-v2               │  768 │ 420MB  │ Better quality       │
    │ multi-qa-MiniLM-L6-cos-v1       │  384 │  80MB  │ Good for Q&A         │
    │ paraphrase-multilingual-MiniLM  │  384 │ 470MB  │ 50+ languages        │
    └─────────────────────────────────┴──────┴────────┴──────────────────────┘
    """
    return SentenceTransformer(model_name)


# Load once, reuse many times
model = load_sentence_transformer("all-MiniLM-L6-v2")


def embed_locally(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings using a local model — completely free, private.

    Advantages:
      - Free (no API cost)
      - Private (data never leaves your machine)
      - Works offline
      - No rate limits

    Disadvantages:
      - Need to download model files (~80-420MB)
      - Slower on CPU (fast on GPU)
      - Slightly lower quality than OpenAI
    """
    # encode() returns a numpy array of shape (N, dimensions)
    embeddings = model.encode(
        texts,
        batch_size=32,       # Process 32 texts at once
        show_progress_bar=False,
        convert_to_tensor=False   # Return numpy array
    )
    return embeddings.tolist()


# Compute similarity
def semantic_similarity_local(text1: str, text2: str) -> float:
    """Compare two texts using local model."""
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    # util.cos_sim handles normalization automatically
    return float(util.cos_sim(emb1, emb2))


print(semantic_similarity_local("I love cats", "I adore felines"))
# → ~0.87
```

---

## 8. Nomic Embed via Ollama (Completely Free Local)

```bash
# Install Ollama first, then:
ollama pull nomic-embed-text
```

```python
import ollama
import numpy as np


def embed_with_ollama(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings using Nomic Embed via Ollama.

    Completely free, runs locally, no API key needed.
    Requires Ollama installed and nomic-embed-text pulled.

    Quality comparable to OpenAI text-embedding-3-small.
    768 dimensions.
    """
    embeddings = []
    for text in texts:
        response = ollama.embeddings(
            model='nomic-embed-text',
            prompt=text
        )
        embeddings.append(response['embedding'])
    return embeddings


# Test
texts = ["Hello world", "Goodbye world"]
embeddings = embed_with_ollama(texts)
print(f"Embedding dimensions: {len(embeddings[0])}")  # 768
```

---

## 9. Cohere Embed v3 (with Asymmetric Search)

```python
import cohere
import os

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])


def embed_cohere_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed documents for STORAGE in a vector database.

    CRITICAL: Use input_type="search_document" for documents.
    This tells Cohere to create embeddings optimised for being retrieved.
    """
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"   # ← IMPORTANT for documents
    )
    return response.embeddings


def embed_cohere_query(query: str) -> list[float]:
    """
    Embed a SEARCH QUERY for retrieving documents.

    CRITICAL: Use input_type="search_query" for queries.
    Query embeddings are optimised for finding matching documents.

    Why different types?
    Documents and queries have different linguistic styles:
      Document: "The return policy allows 30-day returns..."
      Query:    "can I return something?"
    Separate embeddings help bridge this vocabulary gap.
    """
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"   # ← IMPORTANT for queries
    )
    return response.embeddings[0]
```

---

## 10. Complete Semantic Search System

```python
import numpy as np
from openai import OpenAI
from typing import List, Dict

client = OpenAI()


class SemanticSearchEngine:
    """
    A complete semantic search system.

    Unlike keyword search (which requires exact word matches),
    semantic search finds documents by MEANING.

    Example:
      Query: "automobile"
      Keyword search finds: documents with exact word "automobile"
      Semantic search finds: "car", "vehicle", "sedan", "SUV" ← also matches!
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadata: list[dict] = []

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of texts."""
        response = client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]

    def add_documents(
        self,
        documents: list[str],
        metadata: list[dict] = None
    ) -> None:
        """
        Index documents for search.

        Args:
            documents: List of text documents to index
            metadata: Optional list of dicts with extra info per document
        """
        print(f"Embedding {len(documents)} documents...")

        # Embed in batches of 100 (API limit)
        batch_size = 100
        new_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            new_embeddings.extend(self._embed(batch))

        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        self.metadata.extend(metadata or [{}] * len(documents))

        print(f"Indexed {len(self.documents)} total documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> list[dict]:
        """
        Find the most semantically relevant documents.

        Args:
            query: Natural language search query
            top_k: Max results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of {document, similarity, metadata} dicts
        """
        if not self.documents:
            return []

        # Embed the query
        query_embedding = self._embed([query])[0]
        query_vec = np.array(query_embedding)

        # Compute cosine similarity with all documents
        results = []
        for i, (doc, doc_emb) in enumerate(zip(self.documents, self.embeddings)):
            doc_vec = np.array(doc_emb)

            # Cosine similarity
            dot = np.dot(query_vec, doc_vec)
            norms = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            similarity = float(dot / norms) if norms > 0 else 0

            if similarity >= min_similarity:
                results.append({
                    "document":   doc,
                    "similarity": round(similarity, 4),
                    "rank":       0,
                    "metadata":   self.metadata[i]
                })

        # Sort by similarity (highest first), take top-k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        for i, r in enumerate(results[:top_k]):
            r["rank"] = i + 1

        return results[:top_k]


# Usage example
engine = SemanticSearchEngine()

engine.add_documents([
    "Return policy: items must be returned within 30 days.",
    "Free shipping on orders over $50.",
    "Customer support: Monday-Friday, 9am-6pm EST.",
    "Electronics warranty: 2 years from purchase date.",
    "Gift wrapping available for $5 per item.",
    "Loyalty points: earn 1 point per dollar spent.",
    "International shipping to 40+ countries available.",
], metadata=[
    {"category": "returns"},
    {"category": "shipping"},
    {"category": "support"},
    {"category": "warranty"},
    {"category": "gifts"},
    {"category": "loyalty"},
    {"category": "shipping"},
])

results = engine.search("Can I send something back?")
for r in results:
    print(f"Rank {r['rank']}: ({r['similarity']:.3f}) {r['document']}")
```

---

## 11. Clustering Embeddings with K-Means

```python
from sklearn.cluster import KMeans
import numpy as np


def cluster_documents(
    documents: list[str],
    n_clusters: int = 5
) -> dict[int, list[str]]:
    """
    Automatically group similar documents into clusters.
    No labels needed — unsupervised clustering.

    Useful for:
      - Discovering topic groups in a dataset
      - Organising unstructured content
      - Understanding what themes exist

    Args:
        documents: List of text documents to cluster
        n_clusters: Number of groups to create

    Returns:
        Dict of {cluster_id: [documents in cluster]}
    """
    # Get embeddings for all documents
    embeddings = [get_embedding(doc) for doc in documents]
    embedding_matrix = np.array(embeddings)

    # Run K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embedding_matrix)

    # Group documents by cluster
    clusters: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
    for doc, label in zip(documents, labels):
        clusters[label].append(doc)

    return clusters


# Example: cluster customer feedback
feedback = [
    "Great product, fast delivery!",
    "Item broke after 2 days. Very disappointed.",
    "Shipping was delayed by a week.",
    "Amazing quality, highly recommend.",
    "Wrong item sent. Still waiting for refund.",
    "Packaging was damaged but product is fine.",
    "Best purchase this year!",
    "Took 3 weeks to arrive.",
]

clusters = cluster_documents(feedback, n_clusters=3)
for cluster_id, docs in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
```

---

## 12. Batching for Efficiency

```python
import time
from openai import OpenAI

client = OpenAI()


def embed_batch_efficient(
    texts: list[str],
    batch_size: int = 100
) -> list[list[float]]:
    """
    Embed many texts efficiently using batching.

    Without batching: 1000 texts = 1000 API calls (slow, expensive)
    With batching:    1000 texts = 10 API calls  (10x faster!)

    OpenAI allows up to 2048 inputs per call,
    but 100 is a safe batch size that stays under token limits.

    Args:
        texts: Texts to embed
        batch_size: How many to embed per API call (max 2048)

    Returns:
        List of embeddings in same order as input
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )

        # Embeddings come back in the same order as input
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        print(f"Batch {batch_num+1}/{total_batches} complete ({len(all_embeddings)}/{len(texts)})")

    return all_embeddings


# ── Embedding Cache ──

import json
import hashlib
from pathlib import Path


class EmbeddingCache:
    """
    Cache embeddings to avoid re-computing the same text twice.
    Saves money and speeds up repeated operations.
    """

    def __init__(self, cache_file: str = ".embedding_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load()

    def _load(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def _key(self, text: str, model: str) -> str:
        """Generate a unique cache key for text+model combo."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        key = self._key(text, model)
        return self.cache.get(key)

    def set(self, text: str, model: str, embedding: list[float]):
        key = self._key(text, model)
        self.cache[key] = embedding
        self._save()

    def embed_with_cache(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        """Get embedding from cache or compute and cache it."""
        cached = self.get(text, model)
        if cached is not None:
            return cached

        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        self.set(text, model, embedding)
        return embedding
```

---

## 13. Choosing the Right Embedding Model

```
DECISION GUIDE:
───────────────────────────────────────────────────────────────────────

Need free, no API key, local/private?
  → Sentence Transformers (all-MiniLM-L6-v2) or Nomic via Ollama

Need best quality, willing to pay?
  → OpenAI text-embedding-3-large (3072 dims)

Need balance of quality and cost?
  → OpenAI text-embedding-3-small (1536 dims, $0.02/1M tokens)

Need to search across multiple languages?
  → paraphrase-multilingual-MiniLM (50+ languages, free local)

Need optimised for Q&A retrieval?
  → Cohere embed-english-v3.0 (use input_type!)

Need fastest possible search (sacrifice accuracy)?
  → Use text-embedding-3-small with reduced dimensions (256 dims)

MODEL COMPARISON:
┌──────────────────────────────────────┬──────┬──────────────┬─────────────────────┐
│ Model                                │ Dims │ Cost         │ Best For            │
├──────────────────────────────────────┼──────┼──────────────┼─────────────────────┤
│ text-embedding-3-small               │ 1536 │ $0.02/1M     │ General production  │
│ text-embedding-3-large               │ 3072 │ $0.13/1M     │ High accuracy       │
│ text-embedding-ada-002 (legacy)      │ 1536 │ $0.10/1M     │ Legacy systems      │
│ all-MiniLM-L6-v2 (local)            │  384 │ Free         │ Local, privacy      │
│ all-mpnet-base-v2 (local)           │  768 │ Free         │ Better local        │
│ nomic-embed-text (local via Ollama)  │  768 │ Free         │ Best free local     │
│ embed-english-v3.0 (Cohere)         │ 1024 │ Pay per use  │ RAG + reranking     │
└──────────────────────────────────────┴──────┴──────────────┴─────────────────────┘
```

---

## Key Points for Exam Prep

```
EMBEDDINGS CHEAT SHEET:
  - One-hot: sparse, no meaning. Embeddings: dense, captures meaning
  - Cosine similarity: angle between vectors (-1 to 1)
  - 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite
  - OpenAI text-embedding-3-small: 1536 dims, $0.02/1M tokens
  - Cohere: use input_type="search_document" vs "search_query"
  - Sentence Transformers: free, local, good quality
  - Batch embeddings: embed 100 at a time, not 1 by 1
  - Cache embeddings: don't re-embed unchanged documents
  - Use embeddings for: semantic search, RAG, clustering, recommendations
```

## Practice Questions

1. Why does one-hot encoding fail to capture semantic similarity?
2. What does cosine similarity measure and what is its range?
3. When would you use `input_type="search_query"` vs `"search_document"` in Cohere?
4. How many dimensions does `text-embedding-3-small` produce?
5. What is contrastive learning and how does it train embedding models?
6. Why is batching important when embedding thousands of documents?
7. What does embedding caching save you?
8. Explain the "king - man + woman = queen" property of embeddings.
9. When would you choose local Sentence Transformers over OpenAI embeddings?
10. What does reducing embedding dimensions from 1536 to 256 trade off?
11. How would you build a document recommendation system with embeddings?
12. What is the difference between semantic search and keyword search?
13. How many dimensions does `text-embedding-3-large` produce?
14. What is the Nomic embed model and how do you run it locally?
15. How do you cluster 1000 documents into topics using embeddings?

---
*Next: [08 — Vector Databases](../08-vector-databases/README.md)*
