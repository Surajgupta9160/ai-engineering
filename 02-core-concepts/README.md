# 02 — Core Concepts of AI Engineering
## A Complete Beginner-to-Expert Guide

---

## Table of Contents
1. [Large Language Models (LLMs) — What They Really Are](#1-large-language-models)
2. [How Training Works — Teaching a Model](#2-how-training-works)
3. [How Inference Works — Using a Model](#3-how-inference-works)
4. [Embeddings — Giving Words Meaning as Numbers](#4-embeddings)
5. [Vector Databases — Searching by Meaning](#5-vector-databases)
6. [RAG — Giving LLMs Your Private Data](#6-rag)
7. [Prompt Engineering — Talking to AI Effectively](#7-prompt-engineering)
8. [AI Agents — Autonomous AI Workers](#8-ai-agents)
9. [How Everything Connects](#9-how-everything-connects)
10. [Key Points & Practice Questions](#10-key-points--practice-questions)

---

## 1. Large Language Models

### Start From Scratch: What IS a Language Model?

Before "Large" Language Models, we had simpler language models. Let's build intuition from the ground up.

**A language model answers one question:** *"Given this text so far, what word comes next?"*

```
Simple language model:
─────────────────────
"The cat sat on the ___"

Word            Probability
"mat"           45%
"floor"         20%
"roof"          12%
"chair"         8%
"dog"           0.1%   ← unlikely but possible
"purple"        0.01%  ← very unlikely
"xyzzy"         ~0%    ← makes no sense
```

That's it. A language model is fundamentally a **next-word prediction machine**.

### What Makes it "Large"?

A "large" language model is trained on so much text and has so many parameters (adjustable numbers/weights) that it develops surprisingly broad capabilities.

```
Scale of Modern LLMs:

Training Data:
  Your book shelf: ~1,000 books
  GPT-3 training: ~570 GB of internet text
               = equivalent of ~500 million books
  GPT-4 training: Unknown (much more)

Parameters (like "neurons" or "adjustable knobs"):
  Human brain:  ~100 billion neurons
  GPT-2 (2019):     1.5 billion parameters
  GPT-3 (2020):   175 billion parameters
  GPT-4 (2023):   ~1.76 trillion (estimated)
  Llama 3.1 405B: 405 billion parameters

What "parameters" means:
  Imagine tuning 175 billion tiny dials to perfect settings
  Each dial slightly influences what the model outputs
  Training = finding the right settings for all dials
```

### Why LLMs Are Surprisingly Capable

Here's the mind-blowing part: just by training to predict next words, LLMs accidentally learn:

```
Task                    How LLM learned it
────────────────────────────────────────────────────────────────
Translation             Saw millions of bilingual texts
Code generation         Trained on GitHub repos
Math                    Saw many worked math problems
Reasoning               Read philosophy, logic textbooks
Summarisation           Read documents + their summaries
Question answering      Read Q&A forums (Reddit, StackOverflow)
Creative writing        Read novels, stories, screenplays
Medical knowledge       Read medical journals and textbooks
Legal knowledge         Read legal texts and case studies
```

**No one explicitly taught these capabilities.** They emerge from scale.

### The Key Insight: Emergent Abilities

```
At small scale:
  GPT-1 (117M params) → Could do simple text completion
  GPT-2 (1.5B params) → Could write coherent paragraphs

At large scale (emergent abilities appear suddenly):
  GPT-3 (175B params) → Suddenly could do:
    • Few-shot learning (learn from examples in prompt)
    • Chain of thought reasoning
    • Multi-step arithmetic
    • Code generation
    • Translation between any languages

These abilities were NOT trained directly —
they EMERGED from scale.

This is like: a child learning to ride a bike
suddenly "gets" balance — it's not taught,
it clicks at a threshold of practice.
```

### What LLMs Cannot Do (Critical for AI Engineers)

```
LLMs CANNOT:
──────────────────────────────────────────────────────────────

❌ Access the internet (unless given tools)
   → Model knows things up to training cutoff only

❌ Remember previous conversations (by default)
   → Each API call is stateless — no memory
   → You must pass history in every call

❌ Do precise math reliably
   → LLMs hallucinate math — "feel" their way to answers
   → For math: use code execution tools instead

❌ Access files or databases (unless given tools)
   → Model only knows what's in its context window

❌ Learn from your conversation
   → Talking to it doesn't update its weights
   → It will make the same mistakes next conversation

❌ Guarantee accuracy
   → LLMs are probabilistic — they may confidently lie
   → This is "hallucination" — covered in detail in Section 03
```

---

## 2. How Training Works

### The Analogy: Teaching by Reading

Imagine how a brilliant student learns:
1. Read millions of books, articles, websites
2. For each sentence, predict what word comes next
3. Check if prediction was right
4. Adjust thinking to do better next time
5. Repeat billions of times

That's roughly how LLMs are trained. Let me be more precise:

### Phase 1: Pre-training

```
Pre-training Overview:
─────────────────────────────────────────────────────────────────

INPUT: Massive text dataset (internet, books, code, etc.)
       Terabytes → Trillions of tokens

PROCESS:
  For each chunk of text:
    1. Show model tokens 1 through N
    2. Ask: "What is token N+1?"
    3. Model makes prediction
    4. Check against actual token N+1
    5. Calculate error (loss)
    6. Use backpropagation to adjust all parameters
    7. Repeat for next chunk

EXAMPLE:
  Text: "The Eiffel Tower is located in"
  Model sees: "The Eiffel Tower is located in"
  Model predicts: "Paris" (70%), "France" (20%), "the" (5%)...
  Actual next token: "Paris"
  → Model was correct! Small adjustment to reinforce
  → If wrong, larger adjustment to correct

SCALE:
  Duration: Weeks to months
  Compute: Thousands of GPUs running simultaneously
  Cost: $50M - $100M+ for frontier models
  Iterations: Billions of parameter updates

RESULT: A "base model" — knows a lot, but not helpful
  It will complete text but won't follow instructions.
  Ask "What is the capital of France?" and it might
  continue with "was discussed in the 1800s..." (text completion)
  rather than answering "Paris."
```

### Visualising Backpropagation (Simply)

```
The model has millions of parameters (knobs):

Before training:
  Knob 1: 0.73   Knob 2: -0.21   Knob 3: 0.44   ...

Model predicts wrong → Calculate how wrong (loss = 5.2)

Backpropagation asks:
  "If I change Knob 1 slightly, does the loss go up or down?"
  "If I change Knob 2 slightly, does the loss go up or down?"
  ... (does this for ALL parameters simultaneously)

Gradient descent:
  Move every knob in the direction that reduces loss

After one update:
  Knob 1: 0.71   Knob 2: -0.19   Knob 3: 0.46   ...

After billions of updates:
  All knobs are in positions that minimise prediction error
  → Model is now very good at predicting next tokens
```

### Phase 2: Supervised Fine-Tuning (SFT)

```
Problem: The base model just completes text.
         It doesn't know how to be an assistant.

Solution: Show it thousands of examples of
          "instruction → ideal response" pairs

Example training pairs:
┌─────────────────────────────────────────────────────────────┐
│ Input:    "Translate 'Hello' to French"                     │
│ Output:   "Bonjour"                                        │
├─────────────────────────────────────────────────────────────┤
│ Input:    "Write a Python function to reverse a string"     │
│ Output:   "def reverse_string(s): return s[::-1]"          │
├─────────────────────────────────────────────────────────────┤
│ Input:    "What is the capital of Australia?"               │
│ Output:   "The capital of Australia is Canberra."          │
└─────────────────────────────────────────────────────────────┘

After SFT: Model knows how to follow instructions
           But it might still be rude, biased, or harmful
```

### Phase 3: RLHF (Reinforcement Learning from Human Feedback)

```
Problem: SFT model follows instructions but may be:
  • Unhelpful (gives technically correct but useless answer)
  • Harmful (provides dangerous information)
  • Dishonest (confidently wrong)
  • Sycophantic (tells you what you want to hear)

Solution: RLHF — teach the model human preferences

Step 1: Generate multiple responses to each prompt
  Prompt: "How do I lose weight quickly?"
  Response A: "Take diet pills and fast for a week"  (risky)
  Response B: "Create a caloric deficit through balanced diet and exercise" (good)
  Response C: "I can't help with health questions" (too cautious)

Step 2: Humans rank the responses
  A < C < B  (B is best, A is worst)

Step 3: Train a "reward model" to predict human rankings

Step 4: Use reinforcement learning to train the LLM
  → Optimise the LLM to produce responses that
    the reward model rates highly
  → Constrain it to not change too much from SFT

RESULT: An aligned model — helpful, harmless, honest
        This is what GPT-4, Claude, Gemini are
```

### Training Summary Timeline

```
Months 1-6:  PRE-TRAINING
  Objective: Predict next token
  Dataset: Entire internet + books + code
  Cost: $50M+
  Result: Base model (knows everything, helps with nothing)

Week 1-4:    SUPERVISED FINE-TUNING
  Objective: Follow instructions
  Dataset: 100K instruction-response pairs (human-written)
  Cost: $50K-$500K
  Result: Chat model (helpful but potentially harmful)

Week 1-8:    RLHF / DPO
  Objective: Align with human values
  Dataset: Ranked response pairs from human raters
  Cost: $100K-$1M
  Result: Production model (helpful, harmless, honest)

Total for a frontier model: 6-12 months, $100M-$1B+
```

---

## 3. How Inference Works

### What Happens When You Send a Message to ChatGPT

Step by step, here's exactly what happens:

```
You type: "What is the speed of light?"
Press Enter
          │
          ▼
STEP 1: TOKENISATION
  Your text is split into tokens
  "What", " is", " the", " speed", " of", " light", "?"
  = 7 tokens

  Each token → integer ID from vocabulary
  "What" → 2061
  " is"  → 318
  " the" → 262
  ...

          │
          ▼
STEP 2: EMBEDDING
  Each integer ID → dense vector (list of numbers)
  2061 → [0.21, -0.08, 0.64, 0.15, ..., 0.33]  (768 or more numbers)

  These vectors contain "meaning" as numbers that the
  model can process mathematically.

          │
          ▼
STEP 3: FORWARD PASS THROUGH TRANSFORMER
  The vectors pass through many "transformer layers"
  Each layer applies "attention" (which tokens relate to which?)
  and "feed-forward" operations (process the features)

  In GPT-4: ~96 layers
  In Llama 3.1 8B: 32 layers

  This step is where the "thinking" happens.

          │
          ▼
STEP 4: OUTPUT DISTRIBUTION
  After all layers, the model outputs probabilities
  for every possible next token (50,000+ tokens)

  "The"     → 45%
  "Light"   → 12%
  "Its"     → 8%
  "299"     → 6%
  ...

          │
          ▼
STEP 5: SAMPLING
  Pick one token based on probabilities
  (temperature=0: always pick highest; temperature=1: random sampling)

  Let's say "The" is selected.

          │
          ▼
STEP 6: REPEAT
  Append "The" to context, run forward pass again
  → " speed"
  → " of"
  → " light"
  → " is"
  → " approximately"
  → " 299"
  → ",792"
  → ",458"
  → " metres"
  → " per"
  → " second"
  → "."
  → [END TOKEN]

Final output: "The speed of light is approximately
               299,792,458 metres per second."
```

### Why Inference is Much Cheaper Than Training

```
Training:
  • Updates ALL parameters billions of times
  • Requires gradient computation (expensive)
  • Must process entire dataset
  • Need many GPUs, many days
  • Cost: $50M+ for frontier models

Inference:
  • Parameters are FROZEN (no updates)
  • Just a forward pass (no gradient)
  • Process one request at a time
  • Can run on fewer GPUs
  • Cost: fractions of a penny per query

Analogy:
  Training = Writing a book (takes months)
  Inference = Reading the book (takes minutes)
  You write it once, anyone can read it cheaply.
```

### Inference Latency Components

```
User sends request
       │  Network time: 20-50ms
       ▼
API Server receives request
       │  Queuing time: 0-100ms (depends on load)
       ▼
Tokenisation: ~1ms

Forward pass (first token): 100-500ms
  ← This is "Time to First Token" (TTFT)
  ← Users hate waiting for this

Generate subsequent tokens: 20-100ms per token
  ← For a 200-word response (~267 tokens): 5-27 seconds

Stream tokens to user: continuous as generated
  ← This is why streaming matters! User sees progress.

Total for 200-word response: 2-8 seconds typical
```

---

## 4. Embeddings

### The Problem Computers Have With Words

Computers love numbers. They struggle with words.

```
Computer sees:
  "cat"   = just a string of characters: [c, a, t]
  "dog"   = just a string: [d, o, g]
  "puppy" = just a string: [p, u, p, p, y]

Computer doesn't know:
  • cat and kitten are similar (both are felines)
  • dog and puppy are similar (puppy is a young dog)
  • cat and dog are somewhat related (both are common pets)
  • cat and planet are unrelated

For AI to work with language, it needs to represent
MEANING as numbers — not just characters.
```

### Embeddings: Meaning as Coordinates

An embedding is like GPS coordinates for meaning. Just as GPS tells you where something is in physical space, an embedding tells you where something is in "meaning space."

```
Physical space example:
  New York:  (40.7° N, 74.0° W)
  London:    (51.5° N, 0.1° W)
  Tokyo:     (35.7° N, 139.7° E)

  Distance(New York, London) = 5,570 km
  Distance(New York, Tokyo) = 10,836 km
  → New York is "closer" to London than Tokyo

Embedding space example:
  "cat":    [0.21, -0.08, 0.64, ...]  (768 numbers)
  "kitten": [0.20, -0.07, 0.65, ...]  (768 numbers)
  "car":    [-0.82, 0.45, -0.23, ...] (768 numbers)

  Distance(cat, kitten) = small  → They're similar!
  Distance(cat, car)    = large  → They're different!
```

### How Embeddings Capture Meaning

The magic is in the numbers. Each dimension in an embedding vector captures some aspect of meaning:

```
Simplified 2D example to illustrate the concept:
(Real embeddings are 768-3072 dimensions)

Dimension 1: "Animal-ness" (negative = not animal, positive = animal)
Dimension 2: "Size" (negative = small, positive = large)

                        Animal axis →
                    -2      -1       0       1       2
             2 ┤           horse
  Large       │                     dog
  ↑         1 ┤                          cat
  Size      0 ┼────────────────────────────── gerbil──►
  axis      -1 ┤    car    pencil
  ↓        -2 ┤    truck
             -3 ┤    ship

Words near each other in this space are "similar"
in those dimensions.
```

### Why Embeddings Are Powerful

```
Traditional keyword search:
  Query: "automobile"
  Searches for: documents containing exactly "automobile"
  Misses: "car", "vehicle", "sedan", "SUV"

Embedding-based semantic search:
  Query: "automobile"
  Embeds query: [0.82, 0.15, -0.43, ...]
  Finds documents with SIMILAR embeddings
  Matches: "car" ✓, "vehicle" ✓, "sedan" ✓, "SUV" ✓
           Even "motor vehicle" ✓ (didn't share a word!)

This is the foundation of:
  • RAG (retrieve relevant documents)
  • Semantic search
  • Recommendation systems
  • Clustering similar content
  • Anomaly detection
```

### Creating Embeddings (Code Example)

```python
import os
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_embedding(text: str) -> list[float]:
    """
    Convert text to a vector of numbers.

    text-embedding-3-small produces 1536 numbers per text.
    Similar texts will have similar vectors.
    """
    # Clean the text (embeddings don't like newlines mid-text)
    text = text.replace("\n", " ")

    # Call the OpenAI embeddings API
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )

    # Extract the vector (list of 1536 floats)
    return response.data[0].embedding

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measure similarity between two vectors.
    Returns value between -1 and 1:
      1.0 = identical meaning
      0.0 = unrelated
     -1.0 = opposite meaning
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    # Cosine similarity formula
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Let's test with some examples
texts = [
    "I love my dog",
    "My puppy is adorable",      # Similar to above (both about dogs)
    "Python is a programming language",  # Different topic
    "JavaScript is used for web development",  # Similar to Python line
]

embeddings = [get_embedding(t) for t in texts]

# Print similarity matrix
print("Similarity Matrix:")
print(f"{'':40}", end="")
for t in texts:
    print(f"{t[:20]:22}", end="")
print()

for i, t1 in enumerate(texts):
    print(f"{t1[:40]:40}", end="")
    for j, t2 in enumerate(texts):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:22.3f}", end="")
    print()

# Expected output (approximate):
#                                I love my dog  My puppy is ador  Python is a prog  JavaScript is us
# I love my dog                      1.000          0.876            0.231             0.219
# My puppy is adorable               0.876          1.000            0.198             0.187
# Python is a programming lang       0.231          0.198            1.000             0.812
# JavaScript is used for web...      0.219          0.187            0.812             1.000
```

---

## 5. Vector Databases

### Why Regular Databases Don't Work

Imagine you have 1 million documents, each turned into a 1536-number embedding vector.

```
Problem: Find the 5 most similar documents to a query.

Naive approach (what a regular DB would do):
  Compare query vector to ALL 1 million vectors
  = 1,000,000 × 1,536 multiplications
  = 1.5 billion operations
  = Takes several seconds per query ← Too slow!

Optimised approach (what vector DBs do):
  Build an index that groups similar vectors together
  Jump directly to the relevant region of the space
  Only compare to a small subset of vectors
  = ~100 comparisons instead of 1,000,000
  = Milliseconds per query ← Fast enough!

The trade-off: Approximate Nearest Neighbor (ANN)
  → Might miss the absolute best match 1% of the time
  → But 100x-1000x faster
  → Usually worth it in practice
```

### The Library Analogy for Vector DBs

```
Traditional database = card catalogue with exact titles
  "Find books about 'dogs'" → Only books with "dogs" in title

Vector database = librarian who understands topics
  "Find books about 'dogs'" → Returns:
    • "The Dog Owner's Complete Guide"
    • "Understanding Canine Behaviour"
    • "Your First Puppy"
    • "Man's Best Friend: A History"
    (found these even though some don't say "dogs"!)
```

### How a Vector DB Works Internally

```
INDEXING PHASE (build once, use many times):

1. You provide text documents
   ["The cat sat on the mat", "Dogs are loyal", ...]

2. Each document gets embedded into a vector
   "The cat sat on the mat" → [0.21, -0.08, 0.64, ...]

3. Vectors are stored with an index structure (HNSW, IVF, etc.)
   Think of it like a map with neighborhoods

   HIGH DIMENSIONAL SPACE (simplified to 2D):

        • dog article       • puppy guide
              • canine book
                                           • car manual
                                     • truck review
          • cat article                       • vehicle guide
        • kitten guide

   Nearby = similar meaning

QUERYING PHASE (happens per request):

1. Your query: "what makes dogs good companions?"

2. Embed the query: [0.19, -0.06, 0.62, ...]

3. Navigate the index to find nearby vectors
   → Finds dog article, puppy guide, canine book

4. Return the nearest K documents (K=3 in this case)

5. Use these documents in your LLM prompt
```

### Using Chroma (Simple Vector DB)

```python
# Step 1: Install
# pip install chromadb openai

import chromadb
from openai import OpenAI
import os

# Initialize clients
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
chroma_client = chromadb.Client()  # In-memory for learning

# Step 2: Create a collection (like a table in SQL)
collection = chroma_client.create_collection(
    name="company_knowledge",
    # We'll handle embeddings manually for clarity
)

# Step 3: Some documents to store
documents = [
    "Our return policy allows returns within 30 days of purchase.",
    "Free shipping on all orders over $50.",
    "Customer support is available Monday-Friday, 9am-6pm EST.",
    "We offer a 2-year warranty on all electronics.",
    "Our headquarters is located in San Francisco, California.",
]

# Step 4: Create embeddings for each document
def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=[text], model="text-embedding-3-small"
    )
    return response.data[0].embedding

embeddings = [embed_text(doc) for doc in documents]

# Step 5: Store documents + embeddings in Chroma
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": "policy_docs", "index": i} for i in range(len(documents))]
)

print(f"Stored {collection.count()} documents in vector DB")

# Step 6: Query the vector DB
query = "Can I return something I bought last week?"

# Embed the query
query_embedding = embed_text(query)

# Find the 3 most relevant documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print(f"\nQuery: {query}")
print(f"\nMost relevant documents:")
for i, (doc, distance) in enumerate(zip(
    results['documents'][0],
    results['distances'][0]
)):
    # Distance of 0 = identical, larger = less similar
    similarity = 1 - distance  # Convert distance to similarity
    print(f"\n{i+1}. Similarity: {similarity:.3f}")
    print(f"   Document: {doc}")

# Output:
# 1. Similarity: 0.891
#    Document: Our return policy allows returns within 30 days...
# 2. Similarity: 0.623
#    Document: We offer a 2-year warranty on all electronics...
# 3. Similarity: 0.589
#    Document: Customer support is available Monday-Friday...
```

---

## 6. RAG

### The Problem RAG Solves

```
Scenario: You work at AcmeCorp.
          You want an AI assistant that knows:
          - Your product documentation
          - Company policies
          - Internal knowledge base
          - Recent news about your company

Problem 1: LLMs don't know your private data
  GPT-4 was trained on the public internet.
  It has never seen your internal documentation.

Problem 2: LLMs have a training cutoff
  GPT-4's training data ends in early 2024.
  It doesn't know about things that happened after.

Problem 3: Fine-tuning is expensive and slow
  Training takes weeks and costs thousands of dollars.
  What if your docs change frequently?

Solution: RAG
  Don't train the model on your data.
  Instead, find the relevant bits of your data
  EACH TIME the user asks a question,
  and include them in the prompt.
```

### RAG Step by Step

```
PHASE 1: INDEXING (Do this once, or when data changes)
─────────────────────────────────────────────────────────

Your Documents                                     Vector Database
┌─────────────────┐                               ┌─────────────────┐
│ policy.pdf      │   1. Extract text             │                 │
│ product_guide   │ → 2. Split into chunks  ───►  │ [chunk1_vector] │
│ .md             │   3. Create embeddings        │ [chunk2_vector] │
│ faq.txt         │   4. Store vectors + text     │ [chunk3_vector] │
└─────────────────┘                               │ ...             │
                                                  └─────────────────┘


PHASE 2: QUERYING (Do this for every user question)
────────────────────────────────────────────────────

User: "What's the return deadline?"
         │
         ▼
1. Embed the question
   "What's the return deadline?" → [0.45, -0.12, 0.71, ...]
         │
         ▼
2. Search vector DB for similar chunks
   Found: "Returns must be made within 30 days of purchase date."
          "Items must be unused and in original packaging."
         │
         ▼
3. Build the prompt:
   ┌────────────────────────────────────────────────────┐
   │ SYSTEM: You are a helpful assistant for AcmeCorp.  │
   │         Answer only using the provided context.    │
   │                                                    │
   │ CONTEXT:                                          │
   │ [Returns must be made within 30 days...]          │
   │ [Items must be unused...]                         │
   │                                                    │
   │ USER: What's the return deadline?                  │
   └────────────────────────────────────────────────────┘
         │
         ▼
4. Send to LLM → Get answer
   "You have 30 days from your purchase date to return
    an item. The item must be unused and in its original
    packaging."
         │
         ▼
5. Return answer to user ✓
```

### Simple RAG Implementation

```python
import os
from openai import OpenAI
import chromadb
from typing import List

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
chroma_client = chromadb.Client()

# ──────────────────────────────────────────
# INDEXING: Store your knowledge base
# ──────────────────────────────────────────

def create_knowledge_base(documents: List[str], collection_name: str):
    """
    Takes a list of text documents and stores them
    in a vector database for later retrieval.
    """
    # Create a collection in Chroma
    collection = chroma_client.create_collection(collection_name)

    print(f"Embedding {len(documents)} documents...")

    # Embed all documents
    embeddings = []
    for doc in documents:
        response = openai_client.embeddings.create(
            input=[doc], model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)

    # Store in vector DB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(documents))]
    )

    print(f"Knowledge base created with {collection.count()} documents")
    return collection

# ──────────────────────────────────────────
# RETRIEVAL: Find relevant documents
# ──────────────────────────────────────────

def retrieve_relevant_docs(
    query: str,
    collection,
    n_results: int = 3
) -> List[str]:
    """
    Given a user's question, find the most relevant
    documents from our knowledge base.
    """
    # Embed the query
    query_response = openai_client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    # Search vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results['documents'][0]  # List of most relevant docs

# ──────────────────────────────────────────
# GENERATION: Answer using retrieved docs
# ──────────────────────────────────────────

def rag_answer(query: str, collection) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant docs
    2. Build prompt with context
    3. Generate answer
    """
    # Step 1: Retrieve
    relevant_docs = retrieve_relevant_docs(query, collection)

    # Step 2: Format context
    context = "\n\n".join([
        f"Document {i+1}: {doc}"
        for i, doc in enumerate(relevant_docs)
    ])

    # Step 3: Generate
    system_prompt = """You are a helpful customer service assistant.
    Answer questions ONLY based on the provided context.
    If the context doesn't contain the answer, say "I don't have
    that information in my knowledge base."
    Always be concise and accurate."""

    user_message = f"""Context:
{context}

Question: {query}

Please answer based on the context above."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0  # We want factual, consistent answers
    )

    return response.choices[0].message.content

# ──────────────────────────────────────────
# EXAMPLE USAGE
# ──────────────────────────────────────────

# Your company's knowledge base
company_docs = [
    "Our return policy allows returns within 30 days of purchase. Items must be in original condition.",
    "Free shipping is offered on all orders over $50. Standard shipping takes 5-7 business days.",
    "Customer support is available Monday through Friday, 9am to 6pm Eastern Time.",
    "We offer a 2-year manufacturer warranty on all electronics products.",
    "Orders can be tracked at tracking.acmecorp.com using your order number.",
    "We accept Visa, Mastercard, American Express, and PayPal.",
    "Our store is located at 123 Main Street, San Francisco, CA 94102.",
    "Gift wrapping is available for $5 per item at checkout.",
]

# Create knowledge base (do this once)
kb = create_knowledge_base(company_docs, "acmecorp_kb")

# Ask questions
questions = [
    "Can I return something I bought 3 weeks ago?",
    "How long does shipping take?",
    "What payment methods do you accept?",
    "What's your store address?",
    "Do you offer gift wrapping?",
]

for question in questions:
    print(f"\nQ: {question}")
    answer = rag_answer(question, kb)
    print(f"A: {answer}")
```

---

## 7. Prompt Engineering

### What is a Prompt?

A **prompt** is the text you send to an LLM to get a response. Prompt engineering is the skill of crafting prompts that reliably produce the output you want.

```
Think of prompting like giving instructions to a very smart intern:

Bad instructions:
  "Write something about our product"
  → Intern: writes a 3-page essay with random details

Good instructions:
  "Write a 3-bullet-point summary of our product for
   a busy executive. Each bullet should be under 10 words.
   Focus on: problem solved, key benefit, target user.
   Product info: [details]"
  → Intern: produces exactly what you needed
```

### The Three Message Types

```python
messages = [
    {
        "role": "system",          # ← Instructions for HOW to behave
        "content": "You are a..."  #   Runs before every conversation
    },
    {
        "role": "user",            # ← What the human says
        "content": "Question..."
    },
    {
        "role": "assistant",       # ← What the AI said (conversation history)
        "content": "Answer..."     #   Needed for multi-turn chat
    }
]
```

### Prompt Techniques (Preview — Full coverage in Section 15)

```
1. Zero-shot: Just ask
   "Translate 'Hello' to Spanish"

2. Few-shot: Show examples first
   "cat → gato, dog → perro, house → ?"

3. Chain of Thought: Ask to reason step by step
   "What is 15% tip on a $87 bill? Think step by step."

4. Persona: Give it a role
   "You are a friendly Python tutor..."

5. Format specification: Tell it HOW to respond
   "Respond as JSON with keys: name, age, occupation"
```

---

## 8. AI Agents

### The Limitations of a Plain LLM Call

A plain LLM call is like asking a brilliant professor who:
- Has read every book ever written ✓
- Has no internet connection ✗
- Can't look up current information ✗
- Can't run code ✗
- Can't access your files ✗
- Can't send emails ✗
- Can only talk to you in this one conversation ✗

### What an Agent Adds

An **AI Agent** gives the LLM:
1. **Tools** (ability to DO things)
2. **Memory** (ability to REMEMBER things)
3. **Planning** (ability to DECIDE what to do next)

```
Agent = LLM + Tools + Memory + Planning Loop

Plain LLM:
  User: "What's the weather in New York?"
  LLM:  "I don't have real-time weather data."

Agent with weather tool:
  User: "What's the weather in New York?"
  Agent THINKS: I need to get weather data
  Agent CALLS: get_weather_tool(city="New York")
  Tool RETURNS: "72°F, partly cloudy"
  Agent RESPONDS: "It's 72°F and partly cloudy in New York."
```

### The Agent Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                                │
│                                                                 │
│  User Goal: "Research Tesla's Q3 2025 earnings and             │
│              write a 3-bullet summary"                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ITERATION 1:                                           │   │
│  │  THINK: I need to find Tesla's Q3 2025 earnings data    │   │
│  │  ACT:   search_web("Tesla Q3 2025 earnings report")     │   │
│  │  OBSERVE: Found: Revenue $25.2B, Net Income $2.17B,     │   │
│  │           Deliveries 462,890 vehicles...                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ITERATION 2:                                           │   │
│  │  THINK: I have the data. Now I'll write the summary.    │   │
│  │  ACT:   [No tool needed — write directly]               │   │
│  │  OUTPUT: • Revenue of $25.2B, up 8% YoY                 │   │
│  │           • Net income $2.17B despite pricing pressure   │   │
│  │           • 462,890 deliveries, record quarter           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  DONE — Goal achieved after 2 iterations                       │
└─────────────────────────────────────────────────────────────────┘
```

### A Simple Agent From Scratch

```python
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Define tools (what the agent can do) ──

def get_current_time() -> str:
    """Returns the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    IMPORTANT: In production, never use eval() on untrusted input!
    This is for educational purposes only.
    """
    try:
        # Only allow safe characters
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_word_count(text: str) -> str:
    """Count the number of words in a text."""
    count = len(text.split())
    return f"{count} words"

# Tool registry: maps name → function
TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_word_count": get_word_count,
}

# Tool definitions for the LLM (JSON Schema format)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Input should be a valid math expression like '2 + 2 * 3'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_word_count",
            "description": "Count the number of words in a text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to count words in"}
                },
                "required": ["text"]
            }
        }
    }
]

def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """
    Run an agent that can use tools to answer questions.

    The agent loop:
    1. Send message to LLM
    2. If LLM wants to use a tool, execute it
    3. Send tool result back to LLM
    4. Repeat until LLM gives final answer
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_query}")
    print(f"{'='*60}")

    # Start with just the user's message
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed to answer accurately."},
        {"role": "user", "content": user_query}
    ]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Ask the LLM what to do
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto"  # LLM decides whether to use tools
        )

        assistant_message = response.choices[0].message

        # Case 1: LLM is done, gives final answer
        if not assistant_message.tool_calls:
            print(f"AGENT: {assistant_message.content}")
            return assistant_message.content

        # Case 2: LLM wants to use tools
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"TOOL CALL: {tool_name}({tool_args})")

            # Execute the tool
            if tool_name in TOOLS:
                if tool_args:
                    result = TOOLS[tool_name](**tool_args)
                else:
                    result = TOOLS[tool_name]()
            else:
                result = f"Error: Unknown tool '{tool_name}'"

            print(f"TOOL RESULT: {result}")

            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max iterations reached without completing the task."

# Test the agent
run_agent("What time is it right now?")
run_agent("What is 15% of 847.50?")
run_agent("If I have a 500-word essay and need it to be exactly 250 words, how many words do I need to cut?")
run_agent("What is the current time and what is 2 to the power of 10?")
```

---

## 9. How Everything Connects

### The Complete Picture

Now let's see how all these concepts work together in a real AI application:

```
┌─────────────────────────────────────────────────────────────────────┐
│              COMPLETE AI APPLICATION ARCHITECTURE                    │
│                                                                     │
│  USER: "What were our sales figures for Q3 2025?"                  │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    YOUR APPLICATION                          │  │
│  │                                                              │  │
│  │  1. SAFETY CHECK                                            │  │
│  │     Is this a safe, appropriate question?                   │  │
│  │     → Yes, proceed                                          │  │
│  │                                                              │  │
│  │  2. RETRIEVE CONTEXT (RAG)                                  │  │
│  │     Embed question → Search vector DB                       │  │
│  │     → Found: "Q3 2025 Sales Report.pdf (chunks 3,7,12)"    │  │
│  │                                                              │  │
│  │  3. BUILD PROMPT                                            │  │
│  │     System: "You are a business analyst..."                 │  │
│  │     Context: [Q3 report chunks]                             │  │
│  │     User: "What were our sales figures for Q3 2025?"        │  │
│  │                                                              │  │
│  │  4. CALL LLM API                                            │  │
│  │     Send to GPT-4o or Claude                                │  │
│  │     → "Q3 2025 revenue was $12.4M, up 18% YoY..."          │  │
│  │                                                              │  │
│  │  5. OUTPUT VALIDATION                                       │  │
│  │     Is the response factual and appropriate?                │  │
│  │     → Yes, send to user                                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│           │                                                         │
│           ▼                                                         │
│  USER SEES: "According to the Q3 2025 report, revenue was          │
│              $12.4M, representing 18% year-over-year growth..."     │
│                                                                     │
│  ──────────────────────────────────────────────────────────────    │
│                                                                     │
│  BACKGROUND COMPONENTS:                                            │
│                                                                     │
│  Vector DB ──────────────────────────────── (stores embeddings)   │
│  Embedding Model ────────────────────────── (creates embeddings)  │
│  LLM API ────────────────────────────────── (generates response)  │
│  Prompt Templates ───────────────────────── (consistent prompts)  │
│  Logging/Monitoring ─────────────────────── (track quality/cost)  │
└─────────────────────────────────────────────────────────────────────┘
```

### When to Use Each Concept

```
Decision Guide:
─────────────────────────────────────────────────────────────────

User asks a simple question about a well-known topic:
  → Just call the LLM API directly
  → No RAG needed (model already knows it)
  → Example: "What is machine learning?"

User asks about YOUR private/recent data:
  → Use RAG
  → Embed your docs, store in vector DB, retrieve + generate
  → Example: "What's our refund policy?"

User needs to do a multi-step task with real-world actions:
  → Use an Agent with tools
  → Agent can search web, run code, query databases, etc.
  → Example: "Research our competitors and write a comparison"

User needs a consistent persona / specialised behavior:
  → Use a well-crafted system prompt
  → Example: A customer service bot that only discusses products

User needs AI to learn your specific style/format:
  → Use fine-tuning
  → Train the model on your examples
  → Example: A model that always outputs in your company's voice
```

---

## 10. Key Points & Practice Questions

### Master Reference Card

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CORE CONCEPTS CHEAT SHEET                        │
│                                                                      │
│  LLM                                                                │
│  • Next token prediction machine trained on massive text data       │
│  • Emergent capabilities from scale (translation, coding, etc.)     │
│  • Cannot: access internet, run code, remember (without tools)      │
│                                                                      │
│  TRAINING vs INFERENCE                                              │
│  • Training: learn weights from data (months, millions $$$)         │
│  • Inference: use frozen weights to generate (ms, pennies)          │
│  • As AI engineer: you do inference, not training (usually)         │
│                                                                      │
│  EMBEDDINGS                                                         │
│  • Text → dense vector of numbers that captures meaning             │
│  • Similar meaning → similar vectors                                │
│  • Measured with cosine similarity (range: -1 to 1)                │
│                                                                      │
│  VECTOR DATABASE                                                    │
│  • Stores embeddings + text                                         │
│  • Enables semantic search (find by meaning, not keyword)           │
│  • ANN = approximate but fast; exact = slow but perfect            │
│                                                                      │
│  RAG (Retrieval-Augmented Generation)                               │
│  • Two phases: index (offline) + query (per request)               │
│  • Gives LLMs access to your private/recent data                   │
│  • No training needed — just retrieval + prompting                 │
│                                                                      │
│  AGENT                                                              │
│  • LLM + tools + memory + planning loop                            │
│  • Can act autonomously over multiple steps                         │
│  • ReAct pattern: Think → Act → Observe → Repeat                  │
└──────────────────────────────────────────────────────────────────────┘
```

### Practice Questions

**Beginner:**
1. Explain in your own words what a "next token prediction machine" means.
2. What is the difference between training and inference?
3. Why are embeddings useful for computers working with language?
4. What problem does a vector database solve?
5. What does RAG stand for and what problem does it solve?

**Intermediate:**
6. Why can't you just use a SQL database for semantic search?
7. Explain the two phases of a RAG system and what happens in each.
8. How is an AI Agent different from a regular LLM API call?
9. What is cosine similarity and why is it used for comparing embeddings?
10. Why is pre-training so expensive compared to inference?

**Advanced:**
11. When would you choose RAG over fine-tuning, and why?
12. What are the limitations of the current LLM architecture that affect how you design applications?
13. Explain what "emergent abilities" means and give three examples.
14. How would you design a system that combines RAG with an agent?
15. What are the three training phases of a production LLM and what does each accomplish?

---

*Next: [03 — Terminology](../03-terminology/README.md)*
