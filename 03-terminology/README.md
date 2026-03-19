# 03 — AI Terminology: Complete Beginner-to-Expert Glossary

---

## 1. Tokens — The Atomic Unit of LLMs

### What Is a Token?

A **token** is the basic chunk of text that an LLM reads and generates. It is NOT the same as a word. Tokens are sub-word units determined by a **tokenizer** — a vocabulary of ~50,000–100,000 common sub-strings built during model training.

**The key rule:** LLMs never see raw text. Everything is converted to tokens first.

```
Text you write:    "Hello, world! How are you today?"
                        │
                        ▼ Tokenizer
Tokens produced:   ["Hello", ",", " world", "!", " How", " are", " you", " today", "?"]
Token IDs:         [15496,   11,   995,     0,   1374,   389,   345,    1981,      30]
Count:             9 tokens
```

### Why Spaces Appear at the Start of Tokens

You'll notice " world" has a leading space. This is intentional — it distinguishes the word in the middle of a sentence from the word at the beginning:

```
"world" at sentence start → token: "world"  (ID: 10603)
"world" after a space     → token: " world" (ID: 995)

These are different tokens with different IDs.
```

### The Byte Pair Encoding (BPE) Algorithm

BPE is how the vocabulary is built. It starts with individual characters and merges the most common pairs:

```
Step 1: Start with characters
  Corpus: "cat", "cats", "catalog"
  Vocabulary: c, a, t, s, l, o, g

Step 2: Count pairs
  Most common pair: "c" + "a" = "ca" (appears 3 times)
  Merge them into one token: "ca"

Step 3: Re-tokenize and count again
  "ca" + "t" = "cat" (appears 2 times)
  Merge: "cat"

Step 4: Continue until vocabulary size is reached (50,000 tokens)

Result: Common words become single tokens
  "the" → 1 token
  "hello" → 1 token
  "supercalifragilistic" → many tokens (rare word, split up)
```

### Token Count Examples

```python
import tiktoken  # pip install tiktoken

# Load the tokenizer for GPT-4o
enc = tiktoken.encoding_for_model("gpt-4o")

examples = [
    "Hello",                               # 1 token
    "Hello world",                         # 2 tokens
    "Hello, world!",                       # 4 tokens
    "The quick brown fox",                 # 4 tokens
    "supercalifragilisticexpialidocious",   # 7 tokens (rare word = more tokens)
    "ChatGPT",                             # 3 tokens: "Chat", "G", "PT"
    "2024-01-15",                          # 5 tokens: "2024", "-", "01", "-", "15"
    "def fibonacci(n):",                   # 6 tokens
]

for text in examples:
    tokens = enc.encode(text)
    print(f"{len(tokens):3d} tokens | {text}")

# Output:
#   1 tokens | Hello
#   2 tokens | Hello world
#   4 tokens | Hello, world!
#   4 tokens | The quick brown fox
#   7 tokens | supercalifragilisticexpialidocious
#   3 tokens | ChatGPT
#   5 tokens | 2024-01-15
#   6 tokens | def fibonacci(n):
```

### Language Differences in Tokenization

English has the most efficient tokenization because most models are trained on predominantly English text:

```
"hello" (English)    → 1 token
"hola" (Spanish)     → 1 token  (common enough to have its own token)
"你好" (Chinese)     → 3 tokens  (each character may be multiple tokens)
"مرحبا" (Arabic)     → 5 tokens  (right-to-left, different encoding)

Practical impact:
  Translate "hello" to Chinese: costs MORE tokens for the output
  APIs in non-English languages cost more per word

  Same sentence:
  English: "The weather is nice today" = 5 tokens = $0.000013
  Japanese: 「今日は天気がいいですね」 = 12 tokens = $0.000031
```

### Token Counting Rules of Thumb

```
Approximate conversions:
  1 token  ≈  4 characters
  1 token  ≈  0.75 words (English)
  100 tokens ≈ 75 words ≈ half a page
  1,000 tokens ≈ 750 words ≈ 1.5 pages
  10,000 tokens ≈ 7,500 words ≈ 15 pages
  100,000 tokens ≈ 75,000 words ≈ a full novel

A typical:
  Tweet: 20-50 tokens
  Email: 100-500 tokens
  Short article: 500-1,500 tokens
  Research paper: 5,000-15,000 tokens
  Book chapter: 5,000-20,000 tokens
```

### Why Tokens Matter for AI Engineers

```
1. COST: You pay per token
   GPT-4o: $2.50 per million input tokens
   If your prompt is 500 tokens → costs $0.00125
   If your app handles 100,000 requests/day → $125/day just in prompts

2. CONTEXT LIMIT: Models have a max token limit per call
   GPT-4o: 128,000 tokens max
   Claude 3.5: 200,000 tokens max
   If conversation + context + docs exceeds limit → ERROR

3. LATENCY: More output tokens = slower response
   Generating 100 tokens ≈ 1-3 seconds
   Generating 2,000 tokens ≈ 20-60 seconds
   Always limit max_tokens for faster responses

4. PROMPT DESIGN: Be concise
   Every wasted token costs money and uses context space
```

---

## 2. Context Window

### The Analogy: Working Memory

Think of the context window like your working memory — the amount of information you can hold in your head at once while working on a problem.

```
Human working memory: ~7 items at once (Miller's Law)
GPT-4o context window: 128,000 tokens (~96,000 words)

Like a physical whiteboard:
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW (128K tokens)                  │
│                                                                 │
│  Everything here is "in memory" and influences the response:    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ SYSTEM PROMPT (500 tokens)                                │  │
│  │ "You are a helpful customer service assistant..."         │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ CONVERSATION HISTORY (15,000 tokens)                      │  │
│  │ [Previous 30 messages back and forth]                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ RETRIEVED DOCUMENTS (40,000 tokens)                       │  │
│  │ [RAG chunks from knowledge base]                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ CURRENT USER MESSAGE (200 tokens)                         │  │
│  │ "What is your return policy for electronics?"             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  TOTAL USED: ~55,700 tokens    REMAINING: ~72,300 tokens        │
└─────────────────────────────────────────────────────────────────┘
```

### Context Window Sizes (2025)

| Model | Context Window | Equivalent Pages |
|-------|---------------|-----------------|
| GPT-4o | 128,000 tokens | ~96 pages |
| GPT-4o mini | 128,000 tokens | ~96 pages |
| Claude 3.5 Sonnet | 200,000 tokens | ~150 pages |
| Gemini 1.5 Pro | 1,000,000 tokens | ~750 pages |
| Gemini 1.5 Flash | 1,000,000 tokens | ~750 pages |
| Llama 3.1 8B | 128,000 tokens | ~96 pages |
| Mistral Large | 128,000 tokens | ~96 pages |

### What Happens When You Exceed the Context Window

```
Option A: The API returns an error
  openai.BadRequestError: This model's maximum context length is
  128000 tokens. Your messages resulted in 135000 tokens.

Option B: The model truncates (silently drops early content)
  Some implementations drop the oldest messages automatically
  Risk: model loses important earlier context

Strategies to Handle Long Contexts:
────────────────────────────────────

1. SLIDING WINDOW
   Keep only the most recent N messages
   ┌──────────────────────────────────────────────────────┐
   │ msg1 msg2 msg3 msg4 msg5 msg6 ... msg47 msg48 msg49 │
   └──────────────────────────────────────────────────────┘
                                   ↑
                            Only keep last 20

2. SUMMARISATION
   When context gets long, ask LLM to summarise old messages
   Then replace those messages with the summary
   "Earlier in our conversation: The user asked about..."

3. RAG (for knowledge, not conversation)
   Instead of putting all documents in context,
   only retrieve the relevant 3-5 chunks per question

4. COMPRESSION
   Use a small, cheap model (GPT-4o-mini) to compress long
   content before including in the main context
```

### Context Window Management in Code

```python
from openai import OpenAI
import tiktoken

client = OpenAI()
enc = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(messages: list) -> int:
    """Count total tokens in a message list."""
    total = 0
    for msg in messages:
        # Each message has ~4 tokens overhead
        total += 4
        total += len(enc.encode(msg.get("content", "")))
    return total

def manage_context(
    messages: list,
    max_tokens: int = 100_000,  # Leave buffer below 128K
    system_message: str = None
) -> list:
    """
    Trim conversation history to fit within context window.
    Always keeps: system message + most recent messages.
    """
    # Separate system message
    system = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    # Count tokens while adding messages from newest to oldest
    kept = []
    token_count = count_tokens(system)  # Always include system

    for msg in reversed(conversation):
        msg_tokens = count_tokens([msg])
        if token_count + msg_tokens < max_tokens:
            kept.insert(0, msg)
            token_count += msg_tokens
        else:
            # Add a note that history was truncated
            kept.insert(0, {
                "role": "system",
                "content": "[Note: Earlier conversation history was omitted due to length]"
            })
            break

    return system + kept

# Usage
conversation_history = []

def chat(user_message: str) -> str:
    """Chat function with automatic context management."""
    # Add user message
    conversation_history.append({"role": "user", "content": user_message})

    # Trim if needed
    managed_history = manage_context(
        conversation_history,
        system_message="You are a helpful assistant."
    )

    # Call API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."}
        ] + managed_history
    )

    assistant_reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply
```

---

## 3. Temperature

### The Analogy: Volume Knob for Creativity

Temperature controls how "creative" or "random" the model is. Think of it like a volume knob that goes from 0 (completely deterministic) to 2 (wildly random).

```
Temperature = 0.0 → Always picks the most likely word
              Like asking an expert a factual question:
              they'll give the same correct answer every time.

Temperature = 0.7 → Balanced creativity
              Like asking a creative professional:
              each response is good but slightly different.

Temperature = 1.5 → Very creative / random
              Like asking a very excited brainstormer:
              exciting ideas but might go off-track.

Temperature = 2.0 → Nearly random
              Output can be nonsensical or incoherent.
```

### The Math Behind Temperature (Intuitive Version)

Before sampling, the model produces "logits" — raw scores for each possible next token. Temperature is applied to make the distribution sharper or flatter:

```
Example: Model has computed scores for next word after "The sky is"
Raw logits: blue=5.2, beautiful=3.1, green=1.8, purple=0.5

Step 1: Convert to probabilities with softmax
  At Temperature=1.0 (standard):
    blue:      e^5.2 / sum = 181/213 = 85%
    beautiful: e^3.1 / sum =  22/213 = 10%
    green:     e^1.8 / sum =   6/213 =  3%
    purple:    e^0.5 / sum =   2/213 =  1%

Step 2: Temperature changes the division
  Divide logits by temperature BEFORE softmax:

  Temperature = 0.1 (very sharp):
    blue=52.0, beautiful=31.0, green=18.0, purple=5.0
    → blue:99.9%, others: ~0% (nearly deterministic)

  Temperature = 0.7 (sharper than default):
    blue=7.4, beautiful=4.4, green=2.6, purple=0.7
    → blue:93%, beautiful:6%, others:1%

  Temperature = 1.0 (default):
    blue=85%, beautiful=10%, green=3%, purple=1%

  Temperature = 1.5 (flatter):
    blue=55%, beautiful:25%, green:13%, purple:7%

  Temperature = 2.0 (very flat):
    blue=38%, beautiful:28%, green:20%, purple:14%
```

### Temperature Guidelines by Task

```
Task                          Recommended Temperature
────────────────────────────────────────────────────────
Factual Q&A                   0.0  (always same answer)
Code generation               0.0-0.2  (prefer correctness)
Data extraction               0.0  (consistent output)
Mathematical reasoning        0.0  (no creativity needed)
Summarisation                 0.3-0.5  (consistent but fluent)
Translation                   0.3  (accurate but natural)
Customer service chatbot      0.5-0.7  (helpful and consistent)
General conversation          0.7  (natural feeling)
Creative writing              0.8-1.0  (varied and interesting)
Brainstorming ideas           1.0-1.2  (lots of variety)
Poetry / very creative tasks  1.0-1.5  (embrace variety)
```

---

## 4. Top-P (Nucleus Sampling)

### The Problem With Temperature Alone

Temperature alone has a flaw: when the model is uncertain (many options seem equally likely), it can still sample very low-probability tokens.

```
Uncertain distribution (model doesn't know what comes next):
  word_a: 8%   word_b: 7%   word_c: 6%  ... (200+ options each at 1-8%)

  With temperature=1.0, we might sample one of these tiny-probability words.
  Result: output may seem random or incoherent.
```

### How Top-P Solves This

Top-P (also called nucleus sampling) says: **only sample from the smallest set of words whose combined probability exceeds P.**

```
Example with Top-P = 0.9:

Word         Probability    Cumulative Probability
──────────────────────────────────────────────────
"Paris"      0.72           0.72
"France"     0.11           0.83
"the"        0.06           0.89
"a"          0.03           0.92   ← Stop here (exceeded 0.90)
"beautiful"  0.02           0.94
...          ...            ...

The "nucleus" = {Paris, France, the}
Only sample from these 3 words.
"a", "beautiful", and all other words are excluded.

Why this is smart:
- When model is confident ("Paris" = 72%), nucleus is small = coherent
- When model is uncertain (many words at 5% each), it picks from
  only the top words = still coherent
```

### Top-P vs Temperature Comparison

```
Temperature (static):
  Always adjusts by same factor regardless of distribution shape
  May include many unlikely tokens when distribution is flat

Top-P (adaptive):
  Dynamically adjusts based on distribution shape
  Always includes just enough tokens to cover P% of probability mass

Best practice: Use BOTH together
  temperature=0.7, top_p=0.9  ← Common production setting
  temperature=0.0, top_p=1.0  ← For factual/deterministic tasks
```

---

## 5. Top-K Sampling

### What Is Top-K?

Top-K is simpler than Top-P: just keep the **K most probable tokens** and sample only from those.

```
Example with Top-K = 5:

Before Top-K:                After Top-K (K=5):
  "Paris"      72%             "Paris"      72% ✓ (top 5)
  "France"     11%             "France"     11% ✓
  "the"         6%             "the"         6% ✓
  "a"           3%             "a"           3% ✓
  "beautiful"   2%             "beautiful"   2% ✓
  "city"        1%             "city"        1% ✗ (excluded)
  "old"         0.8%           "old"        0.8% ✗
  ... (50,000 other words)

Only sample from {Paris, France, the, a, beautiful}
Re-normalise their probabilities to sum to 100%:
  Paris: 72/94=77%, France: 11/94=12%, etc.
```

### Top-K vs Top-P

```
Top-K weakness:
  K=50 always includes 50 words, even if
  the 50th word has 0.001% probability
  → Can include very unlikely words

Top-P strength:
  Adapts to the distribution
  When confident: might include only 3 words
  When uncertain: might include 50 words
  → More natural, less likely to produce garbage

Recommendation:
  Most practitioners prefer Top-P over Top-K
  OpenAI recommends either Top-P or temperature, not both
  Anthropic uses Top-P by default
```

---

## 6. Hallucination — The Most Important Failure Mode

### What Is Hallucination?

**Hallucination** is when an LLM confidently states something that is false, made-up, or not grounded in reality.

```
EXAMPLE 1: Fake citation
User: "What academic papers discuss quantum entanglement in AI?"
LLM:  "Here are some key papers:
       1. 'Quantum Coherence in Neural Networks' by Dr. John Smith (2021)
          Published in Nature Neuroscience, DOI: 10.1038/s41583-021-...
       2. 'Entanglement-Based Deep Learning' by..."

       ← These papers may not exist! The model generated
         plausible-sounding but fake academic citations.

EXAMPLE 2: Wrong facts
User: "What year did Tesla go public?"
LLM:  "Tesla went public in 2011."
      ← WRONG! Tesla went public in 2010 (June 29, 2010).
        Stated with complete confidence.

EXAMPLE 3: Invented features
User: "Does iPhone 15 have satellite messaging?"
LLM:  "Yes, iPhone 15 supports satellite messaging in emergency
       situations through Apple's Emergency SOS via satellite."
      ← This happens to be true for iPhone 14+, but the model
        may not have verified the specific iPhone 15 detail.
```

### Why Hallucination Happens

```
Root cause: LLMs are probability machines, not truth machines.

The model doesn't "know" things — it predicts likely continuations:

Question: "Who wrote the novel 'The Art of Prompting'?"

The model's internal process:
  "Questions like this typically have the form:
   '[Book title]' was written by [Author Name] in [Year]..."

  Most likely completion:
   "'The Art of Prompting' was written by James Chen in 2022..."

  The model doesn't KNOW this is false.
  It generated the most statistically plausible pattern.

  This is called: confabulation — filling in gaps with
  plausible-sounding but invented information.
```

### Types of Hallucination

```
1. FACTUAL ERRORS
   Stating wrong facts about real things
   "Paris is the capital of Germany"

2. ENTITY FABRICATION
   Inventing people, papers, companies, products
   "According to Dr. Sarah Johnson from MIT..."

3. TEMPORAL ERRORS
   Wrong dates, sequences, timelines
   "Python was created in 2001" (actually 1991)

4. SOURCE FABRICATION
   Citing non-existent books, papers, articles, URLs
   "As stated in IEEE Transactions Vol 45, pp. 123-145..."

5. CONFIDENT EXTRAPOLATION
   Going beyond what the training data supports
   Guessing about things that are uncertain

6. CONTEXT DRIFT
   Misremembering earlier parts of a long conversation
   "You earlier said X" (when the user didn't say X)
```

### How to Reduce Hallucination

```
Strategy 1: GROUNDING (most effective)
  Give the model the correct information in the prompt
  "Based on the following document: [doc text], answer: [question]"
  → Model answers FROM your text, not from memory

Strategy 2: RAG
  Retrieve relevant documents, include in context
  → Model cites real documents, not invented ones

Strategy 3: TEMPERATURE = 0
  Make the model deterministic
  → Still can hallucinate, but less randomly

Strategy 4: ASK FOR CITATIONS
  "List the sources you're basing this on"
  → Makes model commit to verifiable claims

Strategy 5: ASK MODEL TO SAY "I DON'T KNOW"
  In system prompt: "If you don't know, say 'I don't have
  reliable information about this.'"
  → Reduces overconfident fabrication

Strategy 6: VERIFICATION LOOP
  Use another LLM call to verify the first answer
  "Is this answer factually correct given this context: [doc]"
  → Catch errors before showing to user

Strategy 7: STRUCTURED OUTPUT WITH CONFIDENCE
  Ask: "Answer the question and rate your confidence 0-100"
  → Low confidence answers can be flagged for review
```

---

## 7. System Prompt vs User Message vs Assistant Message

### The Three Roles Explained

Every LLM conversation consists of messages, each with a **role**:

```
┌─────────────────────────────────────────────────────────────────┐
│                  THREE MESSAGE ROLES                             │
│                                                                 │
│  SYSTEM (role: "system")                                       │
│  ─────────────────────────────────────────────────────────     │
│  • Written by YOU (the developer)                               │
│  • Sets the AI's persona, rules, and constraints                │
│  • Processed first, before any user messages                    │
│  • Users typically don't see this                               │
│  • Example: "You are a customer service agent for AcmeCorp.    │
│    Only discuss our products. Be friendly and professional."    │
│                                                                 │
│  USER (role: "user")                                           │
│  ─────────────────────────────────────────────────────────     │
│  • Written by the end user (human turn)                         │
│  • The actual questions, requests, or input                     │
│  • Example: "What is your return policy?"                       │
│                                                                 │
│  ASSISTANT (role: "assistant")                                 │
│  ─────────────────────────────────────────────────────────     │
│  • Written by the AI (or by you when building history)          │
│  • Previous AI responses in a conversation                      │
│  • Include these to give the model memory of what it said       │
│  • Example: "Our return policy allows returns within 30 days..."│
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Turn Conversation Structure

```python
from openai import OpenAI

client = OpenAI()

def build_conversation_example():
    """Shows how multi-turn conversations are structured."""

    messages = [
        # System message: set the rules ONCE at the beginning
        {
            "role": "system",
            "content": """You are a friendly math tutor for high school students.
                          - Explain concepts step by step
                          - Use simple language
                          - Encourage the student when they try
                          - Never just give the answer; guide them to find it"""
        },

        # First user message
        {"role": "user", "content": "I don't understand how to solve quadratic equations"},

        # First AI response (what the AI said)
        {"role": "assistant", "content": "Great question! A quadratic equation has the form ax² + bx + c = 0. The easiest method is the quadratic formula. Have you seen it before?"},

        # Second user message
        {"role": "user", "content": "No, I haven't. Can you show me?"},

        # Second AI response
        {"role": "assistant", "content": "Of course! The formula is: x = (-b ± √(b² - 4ac)) / 2a. Let's try it with a simple example: x² + 5x + 6 = 0. Can you identify what a, b, and c are?"},

        # Third user message (current)
        {"role": "user", "content": "Is a=1, b=5, c=6?"}
    ]

    # The API remembers nothing — you must pass the FULL history each time
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content

# The model's answer will reference: "Yes, that's exactly right! Now plug in..."
# It knows the context because we passed all previous messages.
```

---

## 8. Latency, TTFT, and TPS

### Latency = Total Time for a Response

```
Anatomy of LLM response time:

[Request sent]
     │
     │ Network: 20-100ms (your internet + server location)
     ▼
[API server receives request]
     │
     │ Queue: 0-500ms (how busy the server is)
     ▼
[Processing starts]
     │
     │ Tokenization: ~1ms
     ▼
[First token generated] ← THIS is TTFT (Time to First Token)
     │                     ~100-800ms for most models
     │ Token by token generation:
     │ ~20-100ms per token
     ▼
[Final token generated]
     │
     │ Network: 20-100ms back to you
     ▼
[You see complete response]

TOTAL for a 200-word response (~270 tokens):
  Best case: 0.5s + (270 × 20ms) = 5.9 seconds
  Typical:   0.8s + (270 × 50ms) = 14.3 seconds
  Busy:      2.0s + (270 × 80ms) = 23.6 seconds
```

### TTFT — Time to First Token

TTFT is the time until the user sees the **first** character of the response.

```
Why TTFT matters:
  Without streaming:  User stares at blank screen for 15 seconds
                      Then sees entire response appear at once
                      Feels SLOW even if total time is 15s

  With streaming:     User sees first word after 0.8s
                      Then watches text appear word by word
                      Feels FAST even if total time is 15s

  → Always use streaming for any user-facing application!
  → TTFT is the metric users actually perceive as "speed"
```

### TPS — Tokens Per Second

TPS measures throughput — how fast tokens are generated once generation starts.

```
Typical TPS values:
  GPT-4o:       ~80-120 TPS
  GPT-4o mini:  ~150-200 TPS
  Claude 3.5:   ~80-100 TPS
  Groq (Llama): ~500-800 TPS  (extremely fast inference hardware)
  Local Llama:  ~20-50 TPS    (depends on your hardware)

Time to generate 500 tokens (about 375 words):
  GPT-4o at 100 TPS:    5 seconds
  Groq at 600 TPS:      0.83 seconds
  Local at 30 TPS:      16.7 seconds
```

---

## 9. Model Types: Base vs Instruction-Tuned vs Chat

```
BASE MODEL (also called "pre-trained model" or "foundation model"):
─────────────────────────────────────────────────────────────────
  Training: Predicts next token on internet text
  Behavior: Completes text, doesn't follow instructions

  Input:  "The capital of France is"
  Output: "Paris. It is known for its romantic atmosphere and the
          Eiffel Tower, which was built in..."

  Not useful for chatbots directly.
  Examples: GPT-3 base, Llama-3.1 base

INSTRUCTION-TUNED MODEL (also called "Instruct" model):
────────────────────────────────────────────────────────
  Training: Base + SFT (Supervised Fine-Tuning)
  Behavior: Follows instructions, answers questions

  Input:  "What is the capital of France?"
  Output: "The capital of France is Paris."

  Good for API use. Still may not have great safety.
  Examples: Llama-3.1-8B-Instruct, Mistral-Instruct

CHAT / ALIGNED MODEL:
──────────────────────
  Training: Base + SFT + RLHF (or DPO)
  Behavior: Helpful, harmless, honest — production-ready

  Input:  "How do I hack into my neighbor's WiFi?"
  Output: "I'm not able to help with accessing someone else's
          network without permission. That would be illegal..."

  What you get from OpenAI/Anthropic/Google APIs.
  Examples: GPT-4o, Claude 3.5, Gemini 1.5 Pro
```

---

## 10. Quantization

### What Is Quantization?

LLMs store their parameters (weights) as numbers. Quantization reduces the precision of these numbers to save memory and increase speed.

```
Full precision (FP32):
  Each weight stored as 32-bit floating point
  Range: ±3.4×10^38, very precise
  Memory: 4 bytes per parameter

Half precision (FP16 or BF16):
  Each weight stored as 16-bit floating point
  Range: smaller, slightly less precise
  Memory: 2 bytes per parameter
  Quality: Nearly identical to FP32 in practice

INT8 (8-bit quantization):
  Each weight stored as 8-bit integer
  Range: -128 to 127
  Memory: 1 byte per parameter
  Quality: Slight degradation, often unnoticeable

INT4 (4-bit quantization):
  Each weight stored as 4-bit integer
  Range: -8 to 7
  Memory: 0.5 bytes per parameter
  Quality: Noticeable but acceptable degradation

Example — Llama 3.1 70B model:
  FP32:  70B × 4 bytes = 280 GB RAM needed  ← Impossible on most hardware
  FP16:  70B × 2 bytes = 140 GB RAM needed  ← Needs 2× A100 80GB GPUs
  INT8:  70B × 1 byte  = 70 GB RAM needed   ← Fits on 1 A100 80GB GPU
  INT4:  70B × 0.5B    = 35 GB RAM needed   ← Fits on 1 consumer GPU!
```

### Practical Impact of Quantization

```
For local inference with Ollama:

Model: Llama 3.1 8B

Q8_0 (8-bit):   ~8 GB download,  ~8 GB RAM,  Good quality
Q6_K (6-bit):   ~6 GB download,  ~6 GB RAM,  Very good quality
Q4_K_M (4-bit): ~5 GB download,  ~5 GB RAM,  Good quality (recommended)
Q3_K_M (3-bit): ~3.5 GB,         ~4 GB RAM,  Acceptable quality
Q2_K  (2-bit):  ~2.5 GB,         ~3 GB RAM,  Degraded quality

For most purposes, Q4_K_M offers the best quality/size tradeoff.
```

---

## 11. Fine-Tuning vs Prompting vs RAG (Quick Reference)

```
┌───────────────────────────────────────────────────────────────────────┐
│              COMPARISON: WHEN TO USE EACH APPROACH                    │
│                                                                       │
│  APPROACH       │ COST        │ SPEED  │ BEST FOR                    │
│  ───────────────┼─────────────┼────────┼─────────────────────────    │
│  Prompting      │ Free        │ Fast   │ Simple tasks, common needs  │
│  (just write    │ (beyond API │        │ Quick iteration             │
│   a prompt)     │ call cost)  │        │ Changing requirements       │
│  ───────────────┼─────────────┼────────┼─────────────────────────    │
│  RAG            │ Vector DB + │ Fast   │ Private/recent data         │
│  (retrieve +    │ Embed calls │        │ Large document sets         │
│   generate)     │ (~$10-100/m)│        │ Need citations/sources      │
│  ───────────────┼─────────────┼────────┼─────────────────────────    │
│  Fine-tuning    │ $100-$1000+ │ Fast   │ Consistent style/tone       │
│  (train on      │ for training│ after  │ Specialised domain vocab    │
│   your data)    │             │ train  │ Reduce prompt length        │
└───────────────────────────────────────────────────────────────────────┘

The Golden Rule:
  Try prompting first → if not good enough, try RAG
  → if still not good enough, consider fine-tuning
```

---

## 12. Open Source vs Closed Source Models

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DETAILED COMPARISON                               │
│                                                                      │
│              CLOSED SOURCE              OPEN SOURCE                  │
│              (GPT-4, Claude, Gemini)    (Llama, Mistral)             │
│ ─────────────────────────────────────────────────────────────────    │
│  Model weights   Not available          Downloadable (usually)       │
│  Training data   Not disclosed          Often disclosed              │
│  Modifications   Not possible           Possible (fine-tune)         │
│  Cost structure  Pay per token          Pay for compute              │
│  Privacy         Data goes to vendor    Data stays with you          │
│  GDPR/HIPAA      Complex agreements     Full control                 │
│  Maintenance     Vendor handles         You handle                   │
│  Performance     State of the art       Near state of the art        │
│  Availability    Depends on vendor      Always available             │
│  Customisation   Limited (system prompt)│ Unlimited                  │
│ ─────────────────────────────────────────────────────────────────    │
│  Best for:       Production apps that   Sensitive data, cost         │
│                  need best quality      at scale, custom needs       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 13. Complete Terminology Reference Table

| Term | Simple Definition | Why It Matters |
|------|------------------|----------------|
| **Token** | Sub-word chunk (~4 chars); the unit LLMs process | Determines cost and context limits |
| **Context Window** | Max tokens in one API call (includes all text) | Limits how much history/data you can include |
| **Temperature** | Randomness knob (0=deterministic, 2=wild) | Set to 0 for facts, 0.7 for chat, 1+ for creativity |
| **Top-P** | Only sample from tokens covering P% probability | Use 0.9 for most tasks; prevents garbage tokens |
| **Top-K** | Only sample from top K probable tokens | Less common than Top-P; simpler |
| **TTFT** | Time to First Token — when user sees first character | Critical for UX; use streaming to improve perceived speed |
| **TPS** | Tokens Per Second — generation speed | More TPS = shorter total wait time |
| **Hallucination** | LLM confidently stating false information | Use RAG, grounding, and verification to mitigate |
| **Grounding** | Providing real data for LLM to base answers on | RAG is the main grounding technique |
| **System Prompt** | Developer-set instructions processed before user input | Controls persona, rules, output format |
| **Base Model** | Model trained only on text prediction | Not suitable for direct use; needs fine-tuning |
| **Instruction-tuned** | Base + SFT to follow instructions | Can use for API calls but less safe than chat models |
| **Chat Model** | Base + SFT + RLHF for production use | What GPT-4, Claude, Gemini are |
| **Parameters** | The numerical weights in the model | More params = more capacity but more compute |
| **Quantization** | Reducing weight precision to save memory | INT4 = 8x smaller than FP32; some quality loss |
| **Fine-tuning** | Further training on specific data | Changes model behaviour permanently |
| **RAG** | Retrieve relevant docs, include in prompt | Gives model access to your private/recent data |
| **RLHF** | Training technique using human preference rankings | Makes models helpful, harmless, honest |
| **Constitutional AI** | Anthropic's method of using written principles for alignment | Claude's approach to safety |
| **Embeddings** | Dense vector representation of text meaning | Enables semantic search and RAG |
| **Vector DB** | Database optimised for storing/searching embeddings | Stores knowledge base for RAG |
| **Inference** | Using a trained model to generate output | What you do when calling the API |
| **Training** | Teaching model from data by adjusting weights | Done by model providers, very expensive |

---

## Practice Questions

**Beginner:**
1. What is the difference between a token and a word?
2. What does it mean when we say an LLM has a 128K context window?
3. What temperature would you use for a chatbot that answers tax questions?
4. What is hallucination in the context of AI?
5. What is the difference between a base model and a chat model?

**Intermediate:**
6. Why does Chinese text cost more tokens than English text?
7. Explain the difference between Top-P and Top-K sampling.
8. What is TTFT and why does it matter more than total latency?
9. Why does INT4 quantization make a 70B model runnable on consumer hardware?
10. When would you choose RAG over fine-tuning?

**Advanced:**
11. Design a context window management strategy for a long-running customer service chatbot.
12. Explain why temperature=0 doesn't completely eliminate hallucination.
13. What are the trade-offs between model precision (FP16 vs INT4)?
14. How does the RLHF training phase affect a model's tendency to hallucinate?
15. If a model has 1 trillion parameters but poor training data, will it outperform a 70B model with excellent training data?

---
*Next: [04 — Pre-trained Models](../04-pretrained-models/README.md)*
