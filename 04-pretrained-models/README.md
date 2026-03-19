# 04 — Pre-trained Models: Complete Guide to All Major LLMs

---

## Why This Section Matters

As an AI Engineer, choosing the right model is one of the most important decisions you make. The wrong choice costs you:
- 10x too much money (using GPT-4 when GPT-4o mini suffices)
- Poor quality (using a cheap model for a complex task)
- Privacy violations (sending sensitive data to wrong provider)
- Vendor lock-in (not designing for model swapping)

This section covers every major model family with deep detail so you can make informed decisions.

---

## 1. OpenAI Models

### The Model Family Tree

```
OpenAI's models through history:

GPT-1 (2018)    117M params — first GPT, could generate text
GPT-2 (2019)    1.5B params — "too dangerous to release" (OpenAI)
GPT-3 (2020)    175B params — GPT-3 changed everything; first API
GPT-3.5 (2022)  — Fine-tuned GPT-3; powered ChatGPT at launch
GPT-4 (2023)    ~1.7T params — Multimodal, major quality leap
GPT-4o (2024)   — "Omni" — text + vision + audio, faster + cheaper
GPT-4o mini     — Smaller, much cheaper, 80% quality of GPT-4o
o1 (2024)       — Reasoning model; "thinks" before answering
o3 (2025)       — More powerful reasoning model
o4-mini (2025)  — Fast, cheap reasoning
```

### GPT-4o — The Workhorse

**GPT-4o** ("o" = omni) is OpenAI's flagship general-purpose model. It handles text, images, and audio in a single model.

```
Key Stats:
  Context window:  128,000 tokens (~96,000 words / 192 pages)
  Max output:      16,384 tokens (~12,000 words)
  Modalities:      Text, Images (input), Audio (input/output via API)
  Knowledge:       Up to early 2024
  Pricing (2025):  $2.50 / 1M input tokens
                   $10.00 / 1M output tokens

Strengths:
  ✓ Excellent instruction following
  ✓ Strong coding ability
  ✓ Good mathematical reasoning
  ✓ Multimodal (can see images)
  ✓ Fast response time

Weaknesses:
  ✗ More expensive than mini
  ✗ Knowledge cutoff (no real-time data)
  ✗ Can hallucinate on obscure facts
```

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Basic text completion
def ask_gpt4o(question: str, system: str = "You are a helpful assistant.") -> str:
    """
    Call GPT-4o with a question and return the answer.

    Parameters:
        question: The user's question
        system: Instructions that define the AI's behavior

    Returns:
        The AI's response as a string
    """
    response = client.chat.completions.create(
        model="gpt-4o",           # Use the latest GPT-4o
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        temperature=0.7,          # Balanced creativity
        max_tokens=1000,          # Limit response length
    )
    return response.choices[0].message.content

# Example usage
answer = ask_gpt4o("Explain the difference between SQL and NoSQL databases")
print(answer)

# With image input
import base64

def analyze_image(image_path: str, question: str) -> str:
    """Ask GPT-4o a question about an image."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine file extension for media type
    ext = image_path.split(".")[-1].lower()
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "gif": "image/gif",
                  "webp": "image/webp"}.get(ext, "image/jpeg")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}",
                        "detail": "high"  # "low" for cheaper, "high" for detailed
                    }
                },
                {"type": "text", "text": question}
            ]
        }]
    )
    return response.choices[0].message.content

# result = analyze_image("screenshot.png", "What error is shown in this screenshot?")
```

### GPT-4o mini — When to Choose It

**GPT-4o mini** is designed for high-volume, cost-sensitive applications where you don't need the absolute best quality.

```
Key Stats:
  Context window:  128,000 tokens
  Max output:      16,384 tokens
  Pricing:         $0.15 / 1M input tokens  ← 16.7x cheaper than GPT-4o
                   $0.60 / 1M output tokens ← 16.7x cheaper

Cost comparison:
  1 million calls with 500 tokens each:
    GPT-4o:      500M tokens × $2.50/1M = $1,250
    GPT-4o mini: 500M tokens × $0.15/1M = $75

  Savings: $1,175 per million calls

Quality vs Cost Decision:
  Use GPT-4o when:      Use GPT-4o mini when:
  ─────────────────     ────────────────────────
  Complex reasoning     Simple classification
  Code generation       Routing/classification
  Long documents        High-volume extraction
  Important answers     Quick summaries
  Customer-facing       Internal tools
```

```python
# Cost-aware model selection
def choose_model(task_complexity: str) -> str:
    """Select model based on task complexity."""
    if task_complexity == "simple":
        # Classification, extraction, simple Q&A
        return "gpt-4o-mini"
    elif task_complexity == "medium":
        # General chat, summarization
        return "gpt-4o-mini"  # Try mini first, upgrade if quality insufficient
    else:
        # Complex reasoning, code, analysis
        return "gpt-4o"
```

### o1 and o3 — The Reasoning Models

**o1** and **o3** are a completely different category of models. They're designed for tasks that require careful, step-by-step reasoning.

```
What makes reasoning models different:

Standard GPT-4o:
  Sees question → immediately starts generating answer
  Fast but can make reasoning errors

o1/o3 models:
  Sees question → THINKS internally for several seconds
  Internal "chain of thought" reasoning (not shown to user)
  Then generates answer based on its reasoning
  Slower but much better at hard problems

Example:
  Question: "A bat and a ball cost $1.10 in total.
             The bat costs $1.00 more than the ball.
             How much does the ball cost?"

  GPT-4o (common wrong answer): "$0.10"
  (Intuitive answer: $1.00 + $0.10 = $1.10 ✓... wait)
  (Correct: ball=5¢, bat=$1.05, total=$1.10)

  o1 (correct): Thinks carefully...
  "Let ball = x. Then bat = x + 1.00.
   Total: x + (x + 1.00) = 1.10
   2x + 1.00 = 1.10
   2x = 0.10
   x = 0.05
   Ball costs $0.05 (5 cents)."
```

```
When to Use Reasoning Models:
  ✓ Hard math problems
  ✓ Complex logic puzzles
  ✓ Multi-step reasoning
  ✓ Scientific analysis
  ✓ Code debugging that requires tracing logic
  ✓ Planning and strategy

When NOT to Use Reasoning Models:
  ✗ Simple Q&A (overkill)
  ✗ Creative writing (not designed for this)
  ✗ Speed-critical applications (thinking takes time)
  ✗ High-volume cheap tasks (very expensive)

Cost comparison:
  GPT-4o:   $2.50/1M input
  o1:       $15.00/1M input  ← 6x more expensive
  o3:       $10.00/1M input  ← 4x more expensive
```

---

## 2. Anthropic Claude

### Philosophy and Approach

Anthropic was founded by former OpenAI employees with a focus on AI safety. Claude is trained using **Constitutional AI** — a method where the model is given a set of principles and is trained to follow them.

```
Constitutional AI (Simplified):
  1. Give model a "constitution" (set of principles)
     Example: "Be helpful. Avoid harmful content.
               Be honest. Respect privacy."

  2. Have model evaluate its own responses against constitution
     "Does this response violate any principles? If so, revise."

  3. Use these self-critiques to train the reward model

  Result: Model that internalises principles rather than just
          following rules mechanically.

This is why Claude:
  • Gives more nuanced refusals (explains why, offers alternatives)
  • Less likely to be "jailbroken" with clever prompts
  • More willing to engage with complex/sensitive topics appropriately
  • Tends to be more verbose and thorough in responses
```

### Claude 3.5 Sonnet — The Current Best

```
Key Stats:
  Context window:   200,000 tokens (LARGEST of major models as of 2025)
  Knowledge cutoff: April 2024
  Pricing:          $3.00 / 1M input tokens
                    $15.00 / 1M output tokens

What 200K context means practically:
  A full 300-page book = ~75,000 tokens
  Claude can process the entire book in ONE call

  A large codebase = often 100K-300K tokens
  Claude can review entire codebases

Strengths:
  ✓ BEST at instruction following (follows complex instructions precisely)
  ✓ Excellent coding (benchmarks: #1 or #2 consistently)
  ✓ Very large context window (200K)
  ✓ Nuanced, thoughtful responses
  ✓ Good at staying in character / persona
  ✓ Less likely to refuse legitimate requests

Weaknesses:
  ✗ More expensive than GPT-4o for output
  ✗ Can be verbose (thorough but sometimes too long)
  ✗ No internet access by default
```

```python
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def ask_claude(question: str, system: str = None) -> str:
    """
    Call Claude 3.5 Sonnet.

    Note: Anthropic's API is slightly different from OpenAI's:
    - system is a separate parameter, not a message with role
    - Response is in message.content[0].text (not choices[0].message.content)
    - Uses 'max_tokens' (required, not optional)
    """
    kwargs = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,   # REQUIRED for Claude (not optional)
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    # System prompt is separate in Claude (not a message)
    if system:
        kwargs["system"] = system

    message = client.messages.create(**kwargs)

    # Response structure is different from OpenAI
    return message.content[0].text

# Example
answer = ask_claude(
    question="Write a Python function that validates email addresses",
    system="You are a senior Python developer. Write clean, well-documented code."
)
print(answer)

# Streaming with Claude
def ask_claude_streaming(question: str) -> None:
    """Stream Claude's response token by token."""
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()  # New line at end

# Long document analysis (Claude's specialty)
def analyze_long_document(document_text: str, question: str) -> str:
    """
    Use Claude's 200K context to analyze a very long document.
    This is Claude's killer feature — other models can't do this.
    """
    prompt = f"""Please analyze the following document and answer the question.

Document:
{document_text}

Question: {question}

Please provide a thorough analysis."""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

### Claude Model Tiers

```
CLAUDE 3 HAIKU — Fastest and Cheapest
  Price: $0.25/1M input, $1.25/1M output
  Best for: High-volume, simple tasks
  Use: Classification, quick extraction, simple Q&A

CLAUDE 3.5 SONNET — Best Balance (Most Popular)
  Price: $3/1M input, $15/1M output
  Best for: Coding, complex instructions, document analysis
  Use: Most production applications

CLAUDE 3 OPUS — Slowest and Most Expensive
  Price: $15/1M input, $75/1M output
  Best for: Absolute highest quality needed
  Use: Critical analysis, complex research tasks

Decision rule:
  Default to 3.5 Sonnet → if too expensive, use Haiku
  → if quality insufficient, use Opus
```

---

## 3. Google Gemini

### The Google Advantage

Gemini is Google's answer to GPT-4 and Claude. Its key differentiators:

```
1. MASSIVE CONTEXT: 1 million tokens
   What that means:
   - Fit ~750 pages of text in ONE request
   - Entire codebases
   - Hours of video transcripts
   - Years of conversation history

2. NATIVE MULTIMODAL
   Gemini was designed from the ground up for:
   Text + Images + Video + Audio + Code + Documents

3. GOOGLE ECOSYSTEM
   - Integrates with Google Search (grounding)
   - Works with Google Cloud services
   - Good for Google Workspace (Docs, Sheets) integration

4. COMPETITIVE PRICING
   Gemini 1.5 Flash: Very cheap ($0.075/1M input)
```

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ── Basic text generation ──

def ask_gemini(question: str, model_name: str = "gemini-1.5-pro") -> str:
    """
    Call Google's Gemini model.

    model_name options:
      "gemini-1.5-pro"    — Most capable, 1M context
      "gemini-1.5-flash"  — Fast and cheap, 1M context
      "gemini-2.0-flash"  — Latest, fast, 1M context
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(question)
    return response.text

print(ask_gemini("What is the theory of relativity?"))

# ── Generation configuration ──

from google.generativeai.types import GenerationConfig

def ask_gemini_configured(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """Call Gemini with custom configuration."""
    model = genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.9,
            top_k=40,
        )
    )
    response = model.generate_content(question)
    return response.text

# ── Multi-turn chat ──

def gemini_chat_example():
    """Gemini maintains conversation state internally."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat()  # Creates a stateful chat session

    # First message
    response1 = chat.send_message("My name is Alice. I'm learning Python.")
    print(f"Gemini: {response1.text}")

    # Second message — Gemini remembers "Alice" and "Python"
    response2 = chat.send_message("What should I learn next?")
    print(f"Gemini: {response2.text}")
    # Output will reference Alice learning Python

# ── Multimodal (image + text) ──

import PIL.Image

def analyze_image_gemini(image_path: str, question: str) -> str:
    """
    Gemini is natively multimodal — very natural image handling.
    Compare to OpenAI where you must base64-encode images manually.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")

    # Simply open the image — no base64 encoding needed!
    image = PIL.Image.open(image_path)

    # Pass both image and question together
    response = model.generate_content([image, question])
    return response.text

# result = analyze_image_gemini("chart.png", "What trend does this chart show?")

# ── Long context — Gemini's killer feature ──

def analyze_long_text(text: str, question: str) -> str:
    """
    Gemini 1.5 Pro can handle up to 1M tokens.
    This is equivalent to ~750,000 words — an entire book.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = f"""Analyze the following text:

{text}

Question: {question}"""

    # Even if text is 500 pages, Gemini can handle it!
    response = model.generate_content(prompt)
    return response.text
```

### Gemini Model Comparison

```
GEMINI 1.5 FLASH — Best value
  Context: 1M tokens
  Price: $0.075/1M input, $0.30/1M output (VERY cheap!)
  Speed: Fast
  Best for: High-volume tasks, document processing, Q&A

GEMINI 1.5 PRO — Most capable
  Context: 1M tokens
  Price: $1.25/1M input, $5.00/1M output
  Speed: Medium
  Best for: Complex analysis, long documents, multimodal

GEMINI 2.0 FLASH — Latest (2025)
  Context: 1M tokens
  Price: ~$0.10/1M input (experimental pricing)
  Speed: Very fast
  Best for: Latest capabilities, general use
```

---

## 4. Meta LLaMA — Open Source Champion

### Why Open Source Matters

```
The open-source revolution in AI:

2023: GPT-4 released → only accessible via API
      Meta released LLaMA 1 → anyone could run it!
      Community immediately fine-tuned it for chat

2024: LLaMA 3.1 released
      405B parameter model — competes with GPT-4
      FREE to download and run
      Can fine-tune on your own data
      Can deploy anywhere

Impact for AI Engineers:
  BEFORE LLaMA: AI = paying OpenAI/Google forever
  AFTER LLaMA:  AI = downloading free model weights
                     and running on your own hardware

Use cases enabled by open source:
  • HIPAA-compliant healthcare AI (data never leaves hospital)
  • On-device AI (smartphone, laptop, no internet needed)
  • Military/government AI (classified environments)
  • Cost at scale (10M requests/day: $0 vs $25,000/month)
  • Research without API restrictions
  • Custom fine-tuning (impossible with closed models)
```

### LLaMA 3.1 Family

```
MODEL SIZES AND HARDWARE REQUIREMENTS:

LLaMA 3.1 8B
  Parameters: 8 billion
  Context: 128K tokens
  File size: ~4.7 GB (4-bit quantized)
  RAM needed: 8 GB
  Speed: Fast
  Hardware: MacBook Air, most modern laptops with 8GB+ RAM
  Quality: Surprisingly good for size; similar to GPT-3.5

LLaMA 3.1 70B
  Parameters: 70 billion
  Context: 128K tokens
  File size: ~40 GB (4-bit quantized)
  RAM needed: 48 GB
  Speed: Medium
  Hardware: Mac Studio with 96GB RAM, or 2× A100 GPUs
  Quality: Close to GPT-4 on many tasks

LLaMA 3.1 405B
  Parameters: 405 billion
  Context: 128K tokens
  File size: ~230 GB (4-bit quantized)
  RAM needed: 250+ GB
  Speed: Slow
  Hardware: Multiple high-end GPUs / server cluster
  Quality: Competitive with GPT-4o
```

### Running LLaMA with Ollama

```python
# First: install Ollama from https://ollama.com
# Then run in terminal: ollama pull llama3.1

import ollama
import os

def ask_llama(question: str, model: str = "llama3.1") -> str:
    """
    Call LLaMA 3.1 running locally via Ollama.

    Advantages:
    - FREE (no API costs)
    - Private (data never leaves your machine)
    - Works offline
    - No rate limits

    Disadvantages:
    - Need good hardware
    - Slower than cloud APIs
    - Need to download large model files
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question}]
    )
    return response['message']['content']

# Example usage
answer = ask_llama("What is machine learning?")
print(answer)

# With system prompt
response = ollama.chat(
    model="llama3.1",
    messages=[
        {
            "role": "system",
            "content": "You are a Python tutor. Explain things clearly with examples."
        },
        {
            "role": "user",
            "content": "How do Python decorators work?"
        }
    ]
)
print(response['message']['content'])

# Streaming with Ollama
for chunk in ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "Tell me about space exploration"}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
print()

# Using OpenAI SDK with Ollama (compatibility mode!)
from openai import OpenAI

# Point OpenAI SDK at your local Ollama server
local_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Placeholder key, ignored by Ollama
)

response = local_client.chat.completions.create(
    model="llama3.1",  # Use local model name
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
# Same code structure as OpenAI — easy to switch!
```

---

## 5. Mistral AI

### The French Open-Source Model Company

Mistral AI (Paris-based) has built a reputation for extremely efficient models — achieving impressive performance with smaller parameter counts.

### Mixture of Experts (MoE) — Mixtral's Secret

```
Standard Dense Model (like GPT-3):
  Every forward pass:
    ALL neurons fire for EVERY token
    175B model: all 175B parameters active

Mixtral 8×7B — Mixture of Experts:
  Architecture:
    8 "expert" sub-networks, each ~7B parameters
    A "router" decides which 2 experts handle each token

  Every forward pass:
    Only 2 of 8 experts active per token
    Cost: like running a 14B model (2 × 7B)
    Quality: like having a 56B model (8 × 7B)

  The math:
    Total parameters: 56B (8 × 7B)
    Active parameters: 14B (2 × 7B active)
    → Model quality benefits from 56B total knowledge
    → Inference cost is only 14B
    → 4x efficient!

Visual:
  Token "hello" arrives
        │
        ▼
    [Router: decides which 2 experts to use]
        │
    ┌───┴───┐
    │       │
  Expert 3  Expert 7   ← Only these 2 fire
    │       │
    └───┬───┘
        │
    Final output

  Experts 1,2,4,5,6,8 → inactive (saving compute)
```

```python
# Using Mistral API
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

def ask_mistral(question: str, model: str = "mistral-large-latest") -> str:
    """
    Call Mistral AI models.

    Available models:
    "open-mistral-7b"         — Tiny, very cheap, open source
    "open-mixtral-8x7b"       — MoE, great quality/cost ratio
    "open-mixtral-8x22b"      — Larger MoE, stronger
    "mistral-small-latest"    — Balanced
    "mistral-medium-latest"   — Better quality
    "mistral-large-latest"    — Best quality
    """
    response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=question)]
    )
    return response.choices[0].message.content

# Or use via OpenAI-compatible API
from openai import OpenAI

mistral_client = OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key=os.environ["MISTRAL_API_KEY"]
)

response = mistral_client.chat.completions.create(
    model="mistral-large-latest",
    messages=[{"role": "user", "content": "Explain neural networks"}]
)
print(response.choices[0].message.content)
```

---

## 6. Cohere — Enterprise RAG Specialist

### Why Cohere Is Different

Most AI companies focus on chat models. Cohere focuses on **enterprise search and retrieval** use cases.

```
Cohere's Product Focus:
  ┌─────────────────────────────────────────────────────────────┐
  │  Command R/R+     → Best models for RAG applications       │
  │  Embed v3         → Best embedding models for search       │
  │  Rerank           → Dramatically improves search quality   │
  │  Classify         → Text classification pipeline           │
  └─────────────────────────────────────────────────────────────┘

The Cohere Workflow (optimized for enterprise):

  Query → Embed query → Search DB → Rerank results → Generate answer
             ↑               ↑           ↑                  ↑
          Cohere          Any DB      Cohere             Command R
          Embed v3                    Rerank
```

```python
import cohere
import os

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

# ── Text Generation ──
def ask_cohere(question: str) -> str:
    response = co.chat(
        model="command-r-plus",
        message=question
    )
    return response.text

# ── Embeddings (Cohere's strength) ──
def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings for documents.
    IMPORTANT: Use input_type="search_document" for documents
               Use input_type="search_query" for queries
    This improves retrieval quality significantly.
    """
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"   # For documents being indexed
    )
    return response.embeddings

def embed_query(query: str) -> list[float]:
    """Create embedding for a search query."""
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"       # For search queries
    )
    return response.embeddings[0]

# ── Rerank — Cohere's Most Valuable Feature ──
def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 3
) -> list[dict]:
    """
    Rerank documents by relevance to query.

    Use case:
      1. Get 20 candidates from vector DB (fast ANN search)
      2. Rerank with Cohere to find the best 3
      Result: Better quality than vector search alone

    How it works internally:
      Cross-encoder: looks at (query, document) pair together
      More accurate than embedding similarity (which looks at them separately)
    """
    reranked = co.rerank(
        query=query,
        documents=documents,
        model="rerank-english-v3.0",
        top_n=top_n
    )

    results = []
    for item in reranked.results:
        results.append({
            "document": documents[item.index],
            "relevance_score": item.relevance_score,
            "rank": item.index
        })
    return results

# Example: Improved RAG with reranking
def rag_with_reranking(
    query: str,
    documents: list[str],
    initial_k: int = 20,
    final_k: int = 3
) -> str:
    """
    Better RAG pipeline:
    1. Fast ANN search for 20 candidates
    2. Rerank to get best 3
    3. Generate answer from top 3

    Vs naive RAG (top-3 from vector search only):
    - Reranking catches relevant docs that ANN missed
    - Removes false positives from vector search
    """
    # Step 1: Get initial candidates (vector search)
    # ... (vector DB query) ...
    initial_candidates = documents[:initial_k]  # Simplified

    # Step 2: Rerank
    reranked = rerank_documents(query, initial_candidates, top_n=final_k)
    top_docs = [r["document"] for r in reranked]

    # Step 3: Generate
    context = "\n\n".join(top_docs)
    response = co.chat(
        model="command-r-plus",
        message=f"Context:\n{context}\n\nQuestion: {query}",
    )
    return response.text
```

---

## 7. DeepSeek — The Disruptor

### Why DeepSeek Shocked the AI World

```
January 2025: DeepSeek-R1 released
  • Open source reasoning model
  • Performance comparable to OpenAI o1
  • Training cost: ~$6 million
    vs. OpenAI's estimated $100M+ for o1

Why this matters:
  Proved that "bigger budget = better AI" isn't always true
  Showed efficient architecture choices matter enormously
  Opened up reasoning models to the open-source community

DeepSeek innovations:
  • Multi-Head Latent Attention (MLA) — reduces memory usage
  • Mixture of Experts architecture — efficient inference
  • Distillation from larger to smaller models
  • Group Query Attention — faster inference
```

```python
# DeepSeek via their API
from openai import OpenAI

deepseek_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"]
)

def ask_deepseek(question: str, use_reasoning: bool = False) -> str:
    """
    Call DeepSeek models.

    Models:
    "deepseek-chat"   — General purpose chat model
    "deepseek-coder"  — Coding specialist
    "deepseek-r1"     — Reasoning model (like o1)
    """
    model = "deepseek-reasoner" if use_reasoning else "deepseek-chat"

    response = deepseek_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# Or run locally with Ollama
response = ollama.chat(
    model="deepseek-r1:8b",  # 8B version, downloadable
    messages=[{"role": "user", "content": "Solve: 2x + 5 = 13"}]
)
```

---

## 8. Model Selection Decision Guide

```
STEP-BY-STEP MODEL SELECTION:

Step 1: Is data privacy critical?
   YES → Open source (Llama, Mistral), run locally
   NO  → Continue to Step 2

Step 2: What's your budget?
   Very limited → Gemini Flash, GPT-4o mini, Haiku
   Moderate     → GPT-4o, Claude 3.5 Sonnet, Gemini Pro
   No limit     → GPT-4o, Claude 3.5, o1/o3

Step 3: What type of task?
   Hard math/logic/coding → o1, o3, DeepSeek-R1
   Long documents (>100K) → Gemini 1.5 Pro (1M), Claude 3.5 (200K)
   Coding                 → Claude 3.5 Sonnet, GPT-4o
   RAG/Search             → Cohere Command R+
   Image understanding    → GPT-4o, Gemini Pro, Claude 3.5
   Video understanding    → Gemini 1.5 Pro
   Speech/Audio           → Whisper (OpenAI), Gemini

Step 4: Consider volume
   High volume (>1M requests/day) → GPT-4o mini, Gemini Flash
   Low volume                     → Any model that meets quality bar
```

---

## 9. Benchmark Comparison

```
Popular AI benchmarks (higher = better):

Benchmark   What it tests            GPT-4o  Claude 3.5  Gemini 1.5P  Llama 3.1 405B
──────────────────────────────────────────────────────────────────────────────────────
MMLU        General knowledge        88.7    88.3         85.9          88.6
            1000+ questions across
            many subjects

HumanEval   Code generation          90.2    92.0*        84.1          89.0
            164 Python problems

GSM8K       Math word problems       96.1    96.4         90.8          96.8
            Grade school math

MATH        Hard math problems       76.6    71.1         67.7          73.8

* Claude 3.5 Sonnet outperforms GPT-4o on coding

Key insight: Models are within 5% of each other on most benchmarks.
Real-world performance varies by specific task.
Always test on YOUR use case!
```

---

## 10. Pricing Reference Table (2025)

| Model | Input (per 1M) | Output (per 1M) | Context | Notes |
|-------|---------------|-----------------|---------|-------|
| GPT-4o | $2.50 | $10.00 | 128K | Best general purpose |
| GPT-4o mini | $0.15 | $0.60 | 128K | Best value cheap model |
| o1 | $15.00 | $60.00 | 128K | Complex reasoning |
| o3 mini | $1.10 | $4.40 | 200K | Fast reasoning |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 200K | Best instruction following |
| Claude 3 Haiku | $0.25 | $1.25 | 200K | Cheapest Claude |
| Gemini 1.5 Flash | $0.075 | $0.30 | 1M | Cheapest quality model |
| Gemini 1.5 Pro | $1.25 | $5.00 | 1M | 1M context window |
| Llama 3.1 8B | $0.10* | $0.10* | 128K | Open source, cheap via API |
| Llama 3.1 70B | $0.90* | $0.90* | 128K | Open source, strong |
| Mistral Large | $2.00 | $6.00 | 128K | Good instruction following |
| DeepSeek Chat | $0.14 | $0.28 | 64K | Very cheap |
| Cohere Command R+ | $2.50 | $10.00 | 128K | Best for RAG |

*Via Together AI or similar hosted providers

---

## Practice Questions

**Beginner:**
1. What does "GPT-4o" stand for and what does "omni" mean?
2. What is the biggest difference between Claude and GPT-4o?
3. Why would someone run LLaMA locally instead of using the OpenAI API?
4. What makes Gemini 1.5 Pro unique compared to other models?
5. What is Mixture of Experts and which model uses it?

**Intermediate:**
6. When would you choose o1 over GPT-4o?
7. Why does Cohere have a special "search_document" vs "search_query" input type?
8. Explain how Constitutional AI differs from RLHF.
9. If you need to process a 400-page PDF, which model(s) could handle it?
10. What is the cost difference for 1 million calls between GPT-4o and GPT-4o mini?

**Advanced:**
11. Design a model routing strategy for a customer service app that handles both simple FAQs and complex billing disputes.
12. Why might a 70B open-source model be preferable to GPT-4o in a HIPAA-regulated healthcare application?
13. Explain why DeepSeek-R1's competitive performance at a fraction of the training cost is significant.
14. How would you build a fallback system that tries GPT-4o first, then falls back to Claude, then to a local Llama model?
15. What are the trade-offs of using the cheapest model vs the best model in a user-facing application?

---
*Next: [05 — How LLMs Work](../05-how-llms-work/README.md)*
