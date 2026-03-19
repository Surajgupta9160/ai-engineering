# 18 — AI Application Architecture: Building Production AI Systems

---

## 1. The Five Core Architecture Patterns

```
PATTERN 1: SIMPLE LLM CALL
  User → API → LLM → Response
  Use when: Single-turn Q&A, translation, summarisation
  Complexity: Minimal

PATTERN 2: CHATBOT (stateful conversation)
  User → API → [history management] → LLM → Response
                      ↑
              Store history across turns
  Use when: Multi-turn conversation
  Complexity: Low

PATTERN 3: RAG APPLICATION
  User → API → [embed query] → [vector search] → [LLM + context] → Response
  Use when: Answer from private/recent documents
  Complexity: Medium

PATTERN 4: AGENT APPLICATION
  User → API → [Agent Loop] → Response
                 ↓
              Plan → Tool → Observe → Repeat
  Use when: Multi-step tasks, real-world actions
  Complexity: High

PATTERN 5: PIPELINE (document processing)
  Input → Step1(LLM) → Step2(code) → Step3(LLM) → Output
  Use when: Document processing, data extraction, structured workflows
  Complexity: Medium
```

---

## 2. Full-Stack Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION AI APPLICATION STACK                    │
│                                                                      │
│  BROWSER/CLIENT                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  React/Next.js                                                   │ │
│  │  • Chat UI with streaming (EventSource)                          │ │
│  │  • File upload (for RAG)                                        │ │
│  │  • Feedback buttons (👍/👎)                                      │ │
│  │  • Error states and loading indicators                           │ │
│  └───────────────────────────┬─────────────────────────────────────┘ │
│                              │ HTTPS / WebSocket                      │
│  API LAYER                   │                                       │
│  ┌───────────────────────────▼─────────────────────────────────────┐ │
│  │  FastAPI / Express                                               │ │
│  │  • Authentication (JWT / API keys)                               │ │
│  │  • Rate limiting (per user, per IP)                              │ │
│  │  • Input validation                                              │ │
│  │  • Streaming SSE endpoints                                       │ │
│  └───────────┬───────────────────────────────────────────────────── │ │
│              │                                                       │ │
│  AI LAYER    │                                                       │ │
│  ┌───────────▼──────────────────────────────────────────────────┐  │ │
│  │  LLM Orchestration (LangChain / custom)                      │  │ │
│  │  • Prompt management and templating                           │  │ │
│  │  • RAG retrieval pipeline                                     │  │ │
│  │  • Agent loop / tool execution                                │  │ │
│  │  • Response streaming                                         │  │ │
│  └────────┬──────────────────────────────────────────────────── │  │ │
│           │                                                      │  │ │
│  ┌────────▼──────┐  ┌─────────────┐  ┌──────────┐  ┌────────┐ │  │ │
│  │   LLM APIs    │  │  Vector DB  │  │  Redis   │  │  SQL   │ │  │ │
│  │ GPT-4o/Claude │  │  (Chroma/   │  │ (cache + │  │ (chat  │ │  │ │
│  │               │  │   Pinecone) │  │  session)│  │ history│ │  │ │
│  └───────────────┘  └─────────────┘  └──────────┘  └────────┘ │  │ │
│                                                                  │  │ │
│  OBSERVABILITY                                                   │  │ │
│  LangSmith / Langfuse (tracing, feedback, metrics)              │  │ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. FastAPI Backend — Complete AI App

```python
# ============================================================
# PRODUCTION FASTAPI AI APPLICATION
# Covers: chat, streaming, document upload, RAG query
# ============================================================

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import asyncio
import json
import os
import tempfile
from pathlib import Path

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

app = FastAPI(title="AI Application API", version="1.0.0")

# CORS — allow frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise shared resources
oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# ── Request/Response Models ──

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    history: list[dict] = []   # Previous messages for context


class QueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"


# ── Endpoints ──

@app.get("/health")
async def health_check():
    """Health check — required for load balancers."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    """
    Basic chat endpoint.
    Maintains conversation history passed by the client.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ] + request.history + [
        {"role": "user", "content": request.message}
    ]

    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )

    return {
        "reply": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint.
    Returns tokens as they are generated — much better UX for long responses.

    Response format: Server-Sent Events (SSE)
    Each event: "data: {token}\n\n"
    Final event: "data: [DONE]\n\n"
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ] + request.history + [
        {"role": "user", "content": request.message}
    ]

    async def generate_tokens():
        """Async generator that yields tokens as SSE events."""
        try:
            stream = oai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True,
                max_tokens=500
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    # SSE format: "data: TOKEN\n\n"
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"

            # Signal end of stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """
    Upload and index a document for RAG.
    Supports: PDF, TXT, MD files.
    """
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load document
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        documents = loader.load()

        # Add source metadata
        for doc in documents:
            doc.metadata["source_file"] = file.filename

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        # Index in vector DB
        vectorstore.add_documents(chunks)

        return {
            "status": "indexed",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "pages": len(documents)
        }

    finally:
        os.unlink(tmp_path)  # Always cleanup temp file


@app.post("/query")
async def rag_query(request: QueryRequest) -> dict:
    """
    Query the indexed documents using RAG.
    Returns an answer with source citations.
    """
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(
        request.question,
        k=3
    )

    if not docs:
        return {
            "answer": "I don't have any information about that in my knowledge base.",
            "sources": []
        }

    # Build context from retrieved documents
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source_file', 'unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])

    # Generate answer
    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Answer questions using ONLY the provided context. Cite sources. Say 'I don't know' if the context doesn't contain the answer."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {request.question}"
            }
        ],
        temperature=0
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:200]
            }
            for doc in docs
        ]
    }
```

---

## 4. React Streaming Consumer (TypeScript)

```typescript
// ============================================================
// REACT STREAMING COMPONENT
// Consumes the /chat/stream SSE endpoint
// ============================================================

import { useState, useCallback, useRef } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

function useStreamingChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentToken, setCurrentToken] = useState('');

  const sendMessage = useCallback(async (userMessage: string) => {
    // Add user message immediately
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsStreaming(true);
    setCurrentToken('');

    try {
      // POST to streaming endpoint
      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          history: messages  // Pass conversation history
        })
      });

      // Read the SSE stream
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let assistantReply = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);  // Remove "data: " prefix

          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            if (parsed.token) {
              assistantReply += parsed.token;
              setCurrentToken(assistantReply);  // Show partial response
            }
          } catch {
            // Ignore parse errors on incomplete chunks
          }
        }
      }

      // Add complete assistant response to history
      setMessages(prev => [...prev, { role: 'assistant', content: assistantReply }]);

    } catch (error) {
      console.error('Streaming error:', error);
    } finally {
      setIsStreaming(false);
      setCurrentToken('');
    }
  }, [messages]);

  return { messages, isStreaming, currentToken, sendMessage };
}

// ── Usage in a component ──
function ChatInterface() {
  const [input, setInput] = useState('');
  const { messages, isStreaming, currentToken, sendMessage } = useStreamingChat();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {/* Show streaming response in real-time */}
        {isStreaming && currentToken && (
          <div className="message assistant streaming">
            {currentToken}
            <span className="cursor">▊</span>
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          {isStreaming ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}
```

---

## 5. Conversation Memory Management

```python
from langchain.memory import (
    ConversationBufferMemory,       # Keep all messages
    ConversationBufferWindowMemory, # Keep last N exchanges
    ConversationSummaryMemory,      # Summarise old messages
    ConversationTokenBufferMemory,  # Limit by token count
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# ── Buffer: Keep everything (simple, hits context limit) ──
buffer_memory = ConversationBufferMemory(return_messages=True)

# ── Window: Keep last 5 exchanges (prevents context overflow) ──
window_memory = ConversationBufferWindowMemory(
    k=5,                  # Keep last 5 exchanges (10 messages)
    return_messages=True
)

# ── Summary: Compress old messages (best for long conversations) ──
summary_memory = ConversationSummaryMemory(
    llm=llm,              # Uses LLM to write summaries
    return_messages=True
)

# ── Token-limited: Most precise approach ──
token_memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=8000,  # Keep at most 8K tokens of history
    return_messages=True
)

# Choosing the right strategy:
# Short conversations (<20 turns):        ConversationBufferMemory
# Long conversations or cost-sensitive:   ConversationBufferWindowMemory
# Very long + important context:         ConversationSummaryMemory
# Precise token budget management:        ConversationTokenBufferMemory
```

---

## 6. Resilience Patterns

```python
import time
from functools import wraps
from openai import OpenAI, RateLimitError, APIConnectionError

client = OpenAI()


# ── Circuit Breaker ──
class CircuitBreaker:
    """
    Circuit breaker prevents cascading failures.
    After N failures, "opens" the circuit and returns errors immediately
    instead of hitting a broken service.

    States:
      CLOSED: Normal operation, requests pass through
      OPEN:   Too many failures, requests fail immediately
      HALF-OPEN: Testing if service recovered
    """

    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def can_call(self) -> bool:
        """Check if we should attempt a call."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False

        return True  # HALF_OPEN: try one call

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            print(f"Circuit OPENED after {self.failure_count} failures")


circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

def protected_llm_call(messages: list) -> str:
    if not circuit.can_call():
        raise Exception("Circuit breaker OPEN: service unavailable")
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, timeout=15
        )
        circuit.record_success()
        return response.choices[0].message.content
    except Exception as e:
        circuit.record_failure()
        raise
```

---

## 7. Security Checklist

```python
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
import time

app = FastAPI()
security = HTTPBearer()
SECRET_KEY = os.environ["JWT_SECRET_KEY"]


# ── 1. Authentication ──
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token on every protected endpoint."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


# ── 2. Rate Limiting ──
from collections import defaultdict

request_counts: dict = defaultdict(list)

def rate_limit(max_requests: int = 60, window_seconds: int = 60):
    """Limit requests per user per minute."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user_id = request.headers.get("X-User-ID", request.client.host)
            now = time.time()

            # Clean old requests
            request_counts[user_id] = [
                t for t in request_counts[user_id]
                if now - t < window_seconds
            ]

            if len(request_counts[user_id]) >= max_requests:
                raise HTTPException(429, "Rate limit exceeded")

            request_counts[user_id].append(now)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


# ── 3. Input Validation ──
from pydantic import BaseModel, validator

class SecureChatRequest(BaseModel):
    message: str
    session_id: str

    @validator("message")
    def validate_message(cls, v):
        if len(v) > 10000:
            raise ValueError("Message too long (max 10,000 chars)")
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


# ── 4. PII Scrubbing ──
import re

PII_PATTERNS = {
    "email":        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "ssn":          r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card":  r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b',
    "phone":        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
}

def scrub_pii(text: str) -> str:
    """Remove PII before storing logs."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type.upper()} REDACTED]", text)
    return text
```

---

## 8. Docker Deployment

```dockerfile
# Dockerfile for AI application
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```yaml
# docker-compose.yml — full stack
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    volumes:
      - ./vector_db:/app/vector_db  # Persist vector DB

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_DB=aiapp
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

---

## 9. Production Checklist

```
PRODUCTION AI APP CHECKLIST
════════════════════════════════════════════════════════════

SECURITY ✓/✗
□ API keys in environment variables (never in code)
□ JWT authentication on all endpoints
□ Rate limiting per user and per IP
□ Input validation (length, type, format)
□ PII redaction before logging
□ Content moderation on outputs
□ HTTPS only in production
□ CORS configured correctly

RELIABILITY ✓/✗
□ Retry with exponential backoff (3-5 attempts)
□ Timeout on all external calls (15-30s)
□ Model fallback chain configured
□ Circuit breaker for external services
□ Graceful degradation (cached response if API down)
□ Health check endpoint (/health)
□ Database connection pooling

PERFORMANCE ✓/✗
□ Streaming responses for chat
□ Response caching (exact match + semantic)
□ Async/await for all I/O
□ Embedding batch processing
□ Vector DB index optimized
□ CDN for static assets

COST MANAGEMENT ✓/✗
□ Cost tracking per endpoint/user
□ Budget alerts configured
□ Model routing by complexity
□ Context window managed (no unnecessary history)
□ Output length limits set

OBSERVABILITY ✓/✗
□ Request/response logging (PII masked)
□ LLM call tracing (LangSmith/Langfuse)
□ Error rate monitoring
□ Latency percentiles (p50, p95, p99)
□ User feedback collection (thumbs up/down)
□ Alerting on anomalies
□ Cost per request tracked

QUALITY ✓/✗
□ Golden test set for regression testing
□ Prompt version management
□ A/B testing framework
□ Hallucination detection for critical paths
□ Human review queue for flagged outputs
```

---

## Key Points for Exam Prep

```
AI APP ARCHITECTURE CHEAT SHEET:
  - 5 patterns: simple, chatbot, RAG, agent, pipeline
  - FastAPI: use StreamingResponse for /chat/stream
  - SSE format: "data: TOKEN\n\n", end with "data: [DONE]\n\n"
  - React: use fetch() with reader.read() to consume SSE
  - Memory: buffer (all) → window (last N) → summary (compressed)
  - Circuit breaker: CLOSED → OPEN (N failures) → HALF_OPEN (recovery)
  - JWT + rate limiting on all endpoints
  - Scrub PII before logging
  - Docker: always run as non-root user
  - Always: health endpoint, retry logic, model fallback
```

## Practice Questions

1. What are the 5 core AI app architecture patterns and when do you use each?
2. How does Server-Sent Events (SSE) work for streaming AI responses?
3. What is a circuit breaker and when does it open?
4. When would you use ConversationSummaryMemory vs ConversationBufferWindowMemory?
5. How do you authenticate a FastAPI endpoint with JWT?
6. What is PII scrubbing and when should you apply it?
7. How does smart model routing reduce costs in a mixed workload?
8. What should a Docker Dockerfile include for security?
9. What metrics should you alert on in production?
10. How would you implement multi-tenancy in a RAG application?
11. What is the purpose of the X-Accel-Buffering header for SSE?
12. How does the React streaming consumer handle partial JSON chunks?
13. Design the data flow for a document upload → RAG query application.
14. What is the difference between rate limiting and circuit breaking?
15. List 5 items from the production security checklist and explain each.

---

*These notes cover the complete AI Engineer roadmap from beginner to expert.*
*Start with Section 01 and work through sequentially, or jump to specific topics.*
