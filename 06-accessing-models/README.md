# 06 — Accessing AI Models: Complete Practical Guide

---

## 1. Overview: Ways to Access LLMs

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ACCESS METHOD COMPARISON                          │
│                                                                     │
│  METHOD          COST       PRIVACY    SETUP    QUALITY            │
│  ─────────────────────────────────────────────────────────         │
│  Cloud APIs      Pay/token  Medium     Easy     Best               │
│  (OpenAI etc.)                                                      │
│                                                                     │
│  HuggingFace     Free tier  Medium     Medium   Variable           │
│  Inference API                                                      │
│                                                                     │
│  Ollama          Free       Full       Easy     Good               │
│  (local)                                                            │
│                                                                     │
│  LM Studio       Free       Full       Easy     Good               │
│  (local GUI)                                                        │
│                                                                     │
│  Third-party     Pay/token  Medium     Easy     Good               │
│  (Together, Groq)                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Decision Guide

```python
def choose_access_method(requirements: dict) -> str:
    """
    Given requirements, recommend the best access method.
    """
    if requirements.get("needs_privacy") or requirements.get("sensitive_data"):
        return "Ollama or LM Studio (local)"

    if requirements.get("lowest_cost") and requirements.get("high_volume"):
        return "Together AI or Groq (open models via API)"

    if requirements.get("best_quality"):
        return "OpenAI GPT-4o or Anthropic Claude 3.5"

    if requirements.get("beginners"):
        return "OpenAI API (most documentation, easiest)"

    if requirements.get("no_internet"):
        return "Ollama (fully offline)"

    return "OpenAI API (safe default)"
```

---

## 2. OpenAI API — Complete Tutorial

### Setup

```bash
# Install the OpenAI Python library
pip install openai

# Set API key as environment variable (NEVER hardcode in source code!)
# On Mac/Linux:
export OPENAI_API_KEY="sk-proj-..."

# On Windows Command Prompt:
set OPENAI_API_KEY=sk-proj-...

# Or in a .env file (with python-dotenv):
# OPENAI_API_KEY=sk-proj-...
```

```python
import os
from openai import OpenAI

# Initialize client — reads key from OPENAI_API_KEY environment variable
client = OpenAI()

# Alternatively, pass explicitly (not recommended for production):
# client = OpenAI(api_key="sk-...")

# NEVER do this:
# client = OpenAI(api_key="sk-abc123...")  # BAD: key exposed in source code
```

### All Chat Completion Parameters Explained

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    # ── REQUIRED ──
    model="gpt-4o",                    # Which model to use
    messages=[                          # The conversation
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python decorators."}
    ],

    # ── SAMPLING CONTROLS ──
    temperature=0.7,                   # Randomness: 0=deterministic, 2=very random
                                       # Use 0 for facts/code, 0.7 for chat, 1+ for creative

    top_p=0.9,                         # Nucleus sampling: only sample from top 90% probability
                                       # Either use temperature OR top_p, not both (OpenAI recommendation)

    max_tokens=1000,                   # Maximum tokens to GENERATE in the response
                                       # Default: varies by model. Always set this!
                                       # Cost = input tokens + output tokens

    # ── REPETITION CONTROLS ──
    frequency_penalty=0.0,             # Reduce repetition of exact phrases (-2 to 2)
                                       # 0 = no penalty
                                       # 0.5 = moderate reduction of repeated phrases
                                       # 2.0 = strong avoidance of any repeated phrase
                                       # Useful for long-form content that tends to repeat

    presence_penalty=0.0,              # Encourage topic diversity (-2 to 2)
                                       # 0 = no penalty
                                       # 0.5 = moderate encouragement to mention new topics
                                       # 2.0 = strongly encourages breadth over depth
                                       # Useful for brainstorming / diverse idea generation

    # ── STOPPING CONDITIONS ──
    stop=None,                         # Stop generating when these strings are encountered
                                       # Example: stop=["\n", "END", "###"]
                                       # Useful for structured outputs where you know the end marker

    # ── MULTIPLE COMPLETIONS ──
    n=1,                               # How many different responses to generate
                                       # n=3 generates 3 different answers to choose from
                                       # Costs n× more tokens

    # ── STREAMING ──
    stream=False,                      # Stream tokens as generated (True for real-time output)

    # ── OUTPUT FORMAT ──
    response_format={"type": "text"},  # "text" (default) or "json_object" (force JSON output)
                                       # Must tell model to output JSON in the prompt too!

    # ── REPRODUCIBILITY ──
    seed=42,                           # Integer seed for reproducible outputs
                                       # Same seed + same prompt = more likely same output
                                       # Not 100% guaranteed due to hardware floating point

    # ── LOGGING/DEBUGGING ──
    logprobs=False,                    # Return log probabilities of each output token
                                       # Useful for: calibration, uncertainty estimation

    # ── ACCOUNTABILITY ──
    user="user_123abc",                # Your user's unique identifier
                                       # Used by OpenAI for abuse monitoring
                                       # Hash your user IDs for privacy
)

print(response.choices[0].message.content)
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

### Multi-Turn Conversation

```python
from openai import OpenAI

client = OpenAI()

def build_chatbot():
    """
    A simple chatbot that remembers the conversation history.

    KEY INSIGHT: LLMs have NO memory between calls.
    You must pass the ENTIRE conversation history each time.
    The messages list IS the memory.
    """
    # Start with the system message (defines behavior)
    conversation_history = [
        {
            "role": "system",
            "content": """You are a friendly Python tutor.
                         - Explain things clearly with examples
                         - Ask if the student understood
                         - Encourage when they get things right"""
        }
    ]

    print("Python Tutor Chatbot (type 'quit' to exit)")
    print("-" * 40)

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Tutor: Goodbye! Keep coding!")
            break

        if not user_input:
            continue

        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Send FULL history to API (model sees entire conversation)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,  # <-- all messages, not just current
            temperature=0.7,
            max_tokens=500
        )

        # Extract the assistant's response
        assistant_reply = response.choices[0].message.content

        # Add assistant response to history (so model "remembers" it)
        conversation_history.append({
            "role": "assistant",
            "content": assistant_reply
        })

        print(f"Tutor: {assistant_reply}")

build_chatbot()
```

### Streaming Responses

```python
from openai import OpenAI

client = OpenAI()

def stream_response(prompt: str) -> None:
    """
    Stream tokens as they're generated.

    WHY streaming matters for UX:
    Without streaming:  User waits 10 seconds, then sees 500-word answer at once
    With streaming:     User sees first word in 0.5s, watches text appear
                        → Feels much faster even if total time is the same

    How it works:
    API sends chunks (delta) as they're generated
    Each chunk contains a few tokens
    We print them immediately
    """
    print("Response: ", end="", flush=True)

    # stream=True returns a generator instead of a complete response
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True    # ← Enable streaming
    )

    # Iterate over the stream
    full_response = ""
    for chunk in stream:
        # Each chunk may or may not have content
        delta = chunk.choices[0].delta

        if delta.content:          # Check if this chunk has text
            print(delta.content, end="", flush=True)
            full_response += delta.content

    print()  # New line when done
    return full_response

# Test it
stream_response("Explain the water cycle in simple terms.")
```

### Structured Output with Pydantic

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

client = OpenAI()

# Define the EXACT structure you want
class PersonInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(ge=0, le=150, description="Age in years")
    occupation: str
    skills: List[str] = Field(description="List of professional skills")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall tone of the description"
    )

def extract_person_info(text: str) -> PersonInfo:
    """
    Extract structured person information from unstructured text.
    Returns a validated Pydantic model (not just a dict).
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Extract person information from the text."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format=PersonInfo    # ← Pass the Pydantic class
    )

    # Returns a validated PersonInfo object (not raw JSON!)
    return response.choices[0].message.parsed

# Test
text = "John Smith is a 35-year-old software engineer who loves Python, ML, and cloud computing."
person = extract_person_info(text)

# Access as typed attributes
print(f"Name: {person.name}")        # John Smith
print(f"Age: {person.age}")          # 35
print(f"Occupation: {person.occupation}")  # software engineer
print(f"Skills: {person.skills}")    # ['Python', 'ML', 'cloud computing']
print(f"Type: {type(person.age)}")   # <class 'int'>  ← properly typed!
```

### Error Handling

```python
from openai import OpenAI, RateLimitError, AuthenticationError, APIConnectionError, APIError
import time

client = OpenAI()

def robust_api_call(messages: list, max_retries: int = 3) -> str:
    """
    API call with proper error handling and retry logic.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            # Too many requests — wait and retry with exponential backoff
            wait_time = 2 ** attempt   # 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait_time}s before retry {attempt+1}/{max_retries}")
            time.sleep(wait_time)

        except AuthenticationError as e:
            # Wrong API key — don't retry, fail immediately
            raise ValueError("Invalid API key. Check OPENAI_API_KEY environment variable.") from e

        except APIConnectionError as e:
            # Network issue — wait and retry
            wait_time = 2 ** attempt
            print(f"Connection error. Waiting {wait_time}s: {e}")
            time.sleep(wait_time)

        except APIError as e:
            # Other API error (500, etc.) — retry
            print(f"API error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise

    raise Exception(f"Failed after {max_retries} attempts")
```

### Async Usage for Parallel Requests

```python
import asyncio
from openai import AsyncOpenAI

# AsyncOpenAI for async usage
client = AsyncOpenAI()

async def ask_llm_async(prompt: str, model: str = "gpt-4o") -> str:
    """Async version of API call."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

async def process_many_prompts(prompts: list[str]) -> list[str]:
    """
    Process multiple prompts IN PARALLEL — much faster than sequential.

    Sequential:   10 requests × 3 seconds each = 30 seconds total
    Parallel:     10 requests simultaneously = ~3 seconds total
    """
    # Create tasks for all prompts simultaneously
    tasks = [ask_llm_async(prompt) for prompt in prompts]

    # Wait for all to complete in parallel
    results = await asyncio.gather(*tasks)

    return list(results)

# Example usage
async def main():
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Go?",
        "What is Rust?",
        "What is Java?",
    ]

    print("Processing 5 prompts in parallel...")
    start = asyncio.get_event_loop().time()
    answers = await process_many_prompts(prompts)
    elapsed = asyncio.get_event_loop().time() - start

    for prompt, answer in zip(prompts, answers):
        print(f"\nQ: {prompt}")
        print(f"A: {answer[:100]}...")

    print(f"\nTotal time: {elapsed:.2f}s (vs ~{len(prompts)*3}s sequential)")

asyncio.run(main())
```

---

## 3. Google Gemini API

```bash
pip install google-generativeai
```

```python
import google.generativeai as genai
import os
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

# Configure with API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ── Model with full configuration ──

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",        # or "gemini-1.5-flash", "gemini-2.0-flash"

    generation_config=GenerationConfig(
        temperature=0.7,                 # Same as OpenAI temperature
        max_output_tokens=1000,          # Max tokens to generate
        top_p=0.9,                       # Nucleus sampling
        top_k=40,                        # Top-K sampling (Gemini also supports this)
    ),

    # Safety settings — control what content the model will/won't generate
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },

    system_instruction="You are a helpful expert who explains things clearly."
)

# ── Basic call ──
response = model.generate_content("What is photosynthesis?")
print(response.text)

# ── Chat with history ──
chat = model.start_chat()          # Creates stateful chat session
r1 = chat.send_message("Hi, I'm learning about neural networks")
r2 = chat.send_message("What should I learn first?")   # Gemini remembers the context
print(r2.text)

# ── Streaming ──
for chunk in model.generate_content("Write a poem about the ocean", stream=True):
    print(chunk.text, end="", flush=True)
print()

# ── Check safety ratings ──
response = model.generate_content("Tell me about medication dosages")
if response.candidates:
    for candidate in response.candidates:
        print("Safety ratings:", candidate.safety_ratings)
```

---

## 4. Hugging Face

### What Is Hugging Face?

```
Hugging Face is like GitHub but for AI models.
It hosts 500,000+ models you can download and run.

Key sections:
  Models     → Browse and download model weights
  Datasets   → Datasets for training
  Spaces     → Live demos of models
  Hub        → Platform for sharing models
  Inference  → Cloud API for running models without hardware
```

### Cloud Inference (InferenceClient)

```python
from huggingface_hub import InferenceClient

# Get free token from huggingface.co/settings/tokens
client = InferenceClient(token=os.environ["HF_TOKEN"])

# Text generation (many models available)
output = client.text_generation(
    "The future of AI is",
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=200,
    temperature=0.7
)
print(output)

# Chat format (instruction-tuned models)
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is machine learning?"}
]

completion = client.chat_completion(
    messages=messages,
    model="meta-llama/Llama-3.1-70B-Instruct",  # 70B via cloud!
    max_tokens=500
)
print(completion.choices[0].message.content)
```

### Local Inference with Transformers

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ── Easiest: pipeline API ──
# Downloads model automatically, handles everything

generator = pipeline(
    "text-generation",        # Task type
    model="microsoft/phi-2",  # Small model (~1.7B params, ~2GB download)
    device="cpu",             # Use "cuda" if you have NVIDIA GPU
    torch_dtype=torch.float32 # or torch.float16 for faster
)

result = generator(
    "Python is a programming language that",
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,         # Must be True for temperature to work
    pad_token_id=50256      # Avoid warning
)
print(result[0]["generated_text"])

# ── More control: AutoModel + AutoTokenizer ──
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"       # Auto-place on available hardware
)

# Tokenize input
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Generate
with torch.no_grad():       # No gradients needed for inference
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )

# Decode back to text
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### Transformers.js — Run in Browser/Node.js

```javascript
// Install: npm install @xenova/transformers

import { pipeline } from '@xenova/transformers';

// Model downloads and runs entirely in the browser!
// No server needed, no API keys, no cost

async function runInBrowser() {
    // Load a small model (downloads ~25MB)
    const generator = await pipeline(
        'text-generation',
        'Xenova/gpt2'  // Small GPT-2 for demo
    );

    const result = await generator('The meaning of life is', {
        max_new_tokens: 50,
        temperature: 0.7,
    });

    console.log(result[0].generated_text);
}

// Run in Node.js
runInBrowser().catch(console.error);
```

---

## 5. Ollama — Run Models Locally

### Installation

```bash
# macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Windows:
# Download installer from https://ollama.com/download

# Verify installation
ollama --version
```

### Pulling and Running Models

```bash
# Download and run Llama 3.1 (8B model, ~4.7GB)
ollama pull llama3.1          # Download only
ollama run llama3.1           # Download and run immediately

# Other popular models
ollama pull mistral           # Mistral 7B
ollama pull codellama         # Llama fine-tuned for code
ollama pull phi3              # Microsoft Phi-3, small but capable
ollama pull nomic-embed-text  # Embedding model (for RAG)
ollama pull llama3.1:70b      # 70B model (needs ~40GB RAM)

# See downloaded models
ollama list

# Remove a model
ollama rm llama3.1
```

### Python Client

```python
import ollama

# Simple generation
response = ollama.generate(
    model='llama3.1',
    prompt='What is the capital of Japan?'
)
print(response['response'])

# Chat format (with conversation history)
response = ollama.chat(
    model='llama3.1',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain Docker in one sentence.'}
    ]
)
print(response['message']['content'])

# Streaming
for chunk in ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'Count to 10'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
print()

# Embeddings (for RAG)
embedding = ollama.embeddings(
    model='nomic-embed-text',
    prompt='The quick brown fox'
)
vector = embedding['embedding']     # List of ~768 floats
print(f"Embedding dimension: {len(vector)}")
```

### OpenAI-Compatible Endpoint

```python
from openai import OpenAI

# Point the OpenAI SDK at Ollama's local server!
# Ollama serves at http://localhost:11434
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"    # Placeholder — Ollama ignores this
)

# Now use exact same code as OpenAI, but runs locally!
response = client.chat.completions.create(
    model="llama3.1",   # Must match ollama model name
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is Kubernetes?"}
    ]
)
print(response.choices[0].message.content)

# Benefits:
# - Free (no API costs)
# - Private (data never leaves your machine)
# - Offline capable
# - No rate limits
```

---

## 6. LM Studio

### Setup

```
1. Download LM Studio from https://lmstudio.ai
2. Install (Mac, Windows, Linux)
3. Open LM Studio
4. Click "Models" tab → search for models from HuggingFace
5. Download a model (recommend: llama-3.1-8b-instruct Q4_K_M)
6. Go to "Local Server" tab
7. Load the model and click "Start Server"
8. Server runs on http://localhost:1234
```

### Using with Python

```python
from openai import OpenAI

# LM Studio exposes an OpenAI-compatible API!
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"   # Placeholder
)

# Works exactly like OpenAI
response = client.chat.completions.create(
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    messages=[
        {"role": "user", "content": "Hello! Tell me a joke."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

---

## 7. LiteLLM — Universal LLM Interface

```bash
pip install litellm
```

```python
from litellm import completion

# The SAME code works for ANY provider
# Just change the model name!

# OpenAI
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Anthropic Claude (same code!)
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Gemini (same code!)
response = completion(
    model="gemini/gemini-1.5-pro",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Local Ollama (same code!)
response = completion(
    model="ollama/llama3.1",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Get the response text the same way regardless of provider
print(response.choices[0].message.content)

# ── Fallback configuration ──
from litellm import completion

def call_with_fallback(message: str) -> str:
    """
    Try primary model; if it fails, use fallback models.
    Critical for production reliability.
    """
    models_to_try = [
        "gpt-4o",                        # Primary: best quality
        "anthropic/claude-3-5-sonnet",   # Fallback 1: different provider
        "ollama/llama3.1"                # Fallback 2: local, always available
    ]

    for model in models_to_try:
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": message}],
                timeout=30
            )
            print(f"Used model: {model}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Model {model} failed: {e}. Trying next...")

    raise Exception("All models failed")

# ── Cost tracking ──
from litellm import completion_cost

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain AI"}]
)

# Get cost for this call
cost = completion_cost(completion_response=response)
print(f"This call cost: ${cost:.6f}")
# Output: This call cost: $0.000225
```

---

## 8. Best Practices

### Never Hardcode API Keys

```python
# ❌ WRONG — Never do this
client = OpenAI(api_key="sk-proj-abc123xyz...")  # Key visible in code!

# ✅ CORRECT — Environment variables
import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ✅ ALSO CORRECT — .env file with python-dotenv
from dotenv import load_dotenv
load_dotenv()  # Loads .env file
client = OpenAI()  # Reads OPENAI_API_KEY from environment
```

### Always Pin Model Versions

```python
# ❌ WRONG — "gpt-4" may point to different versions over time
response = client.chat.completions.create(model="gpt-4", ...)

# ✅ CORRECT — Pin to specific version
response = client.chat.completions.create(model="gpt-4o-2024-11-20", ...)
# Now your app behaves consistently even as OpenAI releases new versions
```

### Rate Limiting and Retry

```python
import time
from functools import wraps
from openai import RateLimitError

def with_retry(max_retries=3, base_delay=1):
    """Decorator that adds retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                    print(f"Rate limited. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@with_retry(max_retries=3)
def call_api(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---

## Key Points for Exam Prep

```
OPENAI API:
  - 3 message roles: system, user, assistant
  - temperature (0=deterministic, 2=random)
  - frequency_penalty reduces repeated phrases
  - presence_penalty encourages new topics
  - response_format: "json_object" forces JSON
  - stream=True for token-by-token delivery
  - Async with AsyncOpenAI for parallel calls

GEMINI:
  - safety_settings control content filtering
  - start_chat() for stateful conversation
  - Native multimodal (images are easy to pass)

HUGGING FACE:
  - Hub = GitHub for models
  - InferenceClient for cloud, transformers for local
  - Pipeline API = simplest way to run locally

OLLAMA:
  - Local models, free, private, offline
  - OpenAI-compatible endpoint at localhost:11434
  - ollama pull <model> to download

LITELLM:
  - Same code for all providers
  - Just change model name
  - Built-in fallback and cost tracking
```

---

## Practice Questions

1. Why should you never hardcode API keys in source code?
2. What does `frequency_penalty` do and when would you use it?
3. How do you maintain conversation history when calling the OpenAI API?
4. What is streaming and why does it improve perceived speed?
5. What is the difference between `temperature` and `top_p`?
6. How do you use the OpenAI SDK with Ollama running locally?
7. What is LiteLLM and what problem does it solve?
8. How do you generate multiple completions with different variations?
9. What does `seed` do in the OpenAI API?
10. How do you handle RateLimitError in production?
11. What is the difference between sync and async API calls?
12. How do you count tokens before making an API call?
13. What does `response_format: json_object` do and what must you also include?
14. How does `presence_penalty` differ from `frequency_penalty`?
15. How do you ensure your app behaves consistently across OpenAI model updates?

---
*Next: [07 — Embeddings](../07-embeddings/README.md)*
