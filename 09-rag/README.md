# 09 — RAG (Retrieval-Augmented Generation): Complete Guide

---

## The Story: Why Jane Needed RAG

Jane is a lawyer at a large firm. She needs to answer a client's question: *"Does our standard services agreement allow the client to terminate early without penalty?"*

```
WITHOUT RAG:
──────────────────────────────────────────────────────────────
Jane → asks ChatGPT → "I can't access your specific contracts.
                        Generally, termination clauses vary..."
    ↑ USELESS — The model doesn't know her firm's contracts.

WITH RAG:
──────────────────────────────────────────────────────────────
Jane → asks AI system → searches 500+ contract PDFs in seconds
                      → finds Clause 12.3 in services_agreement_v4.pdf
                      → AI answers: "Yes, Section 12.3 allows early
                         termination with 30 days notice, without penalty,
                         if the client provides written notice..."
    ↑ USEFUL — The model answered from HER firm's actual documents.

This is RAG.
```

---

## 1. What Problems RAG Solves

```
PROBLEM 1: LLM DOESN'T KNOW YOUR PRIVATE DATA
  LLMs are trained on the public internet.
  They have NEVER seen:
    - Your internal documents
    - Your company policies
    - Your customer data
    - Your proprietary knowledge

PROBLEM 2: TRAINING CUTOFF
  GPT-4's training ended in early 2024.
  It doesn't know:
    - Events after that date
    - Updated regulations
    - Recent product releases

PROBLEM 3: HALLUCINATION
  Without real data, LLMs make things up.
  With RAG: answers are grounded in real documents.
  You can even cite the source!

PROBLEM 4: FINE-TUNING IS EXPENSIVE
  Training a model on your data costs:
    - Time: weeks
    - Money: $1,000 - $100,000+
    - And it becomes stale when data changes

RAG IS CHEAPER AND ALWAYS FRESH
  - No training required
  - Update documents, not the model
  - See results immediately
```

---

## 2. RAG vs Fine-Tuning vs Long Context

```
┌──────────────────────────────────────────────────────────────────────┐
│              CHOOSING YOUR APPROACH                                   │
├────────────────┬──────────────────────────────────────────────────── │
│ Approach       │ Use When                                             │
├────────────────┼───────────────────────────────────────────────────  │
│ Prompting      │ • Task is straightforward                           │
│ (no retrieval) │ • Info is in LLM's training data                    │
│                │ • Small amount of context needed                    │
├────────────────┼───────────────────────────────────────────────────  │
│ RAG            │ • Private/proprietary documents                     │
│                │ • Frequently changing data                          │
│                │ • Large knowledge base (1000s of docs)              │
│                │ • Need citations/sources in answers                 │
│                │ • Recent information beyond training cutoff         │
├────────────────┼───────────────────────────────────────────────────  │
│ Fine-tuning    │ • Need specific tone/style consistency              │
│                │ • Specialised domain vocabulary                     │
│                │ • Reduce expensive prompt length at scale           │
│                │ • Task not improvable by prompting                  │
├────────────────┼───────────────────────────────────────────────────  │
│ Long context   │ • Small set of documents (<500K tokens)             │
│ (e.g., 1M)     │ • Don't know which parts are relevant               │
│                │ • Want entire document in context always            │
└────────────────┴───────────────────────────────────────────────────  │
```

---

## 3. Core RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG ARCHITECTURE                             │
│                                                                     │
│  PHASE 1: INDEXING (Run once, update when docs change)             │
│  ──────────────────────────────────────────────────────            │
│                                                                     │
│  ┌──────────┐   Extract   ┌───────────┐   Split   ┌────────────┐  │
│  │ Raw Docs │────────────►│   Text    │──────────►│  Chunks    │  │
│  │ PDF,Word │             │           │           │ 500 chars  │  │
│  │ HTML,CSV │             └───────────┘           │ each       │  │
│  └──────────┘                                     └─────┬──────┘  │
│                                                         │          │
│                                                    Embed each      │
│                                                         │          │
│                                                    ┌────▼───────┐  │
│                                                    │ Vector DB  │  │
│                                                    │[0.2,-0.1,..]│ │
│                                                    │[0.8, 0.3,..]│ │
│                                                    │[...] + text │ │
│                                                    └────────────┘  │
│                                                                     │
│  PHASE 2: QUERYING (Every user request)                            │
│  ──────────────────────────────────────────────────────            │
│                                                                     │
│  User: "What is the return policy?"                                 │
│    │                                                                │
│    ├─1. Embed query: [0.45, -0.12, ...]                            │
│    │                                                                │
│    ├─2. Search vector DB → top 3 similar chunks                    │
│    │     "Returns within 30 days..."                               │
│    │     "Items must be in original packaging..."                  │
│    │     "Refund processed within 5-7 business days..."            │
│    │                                                                │
│    ├─3. Build prompt:                                              │
│    │     SYSTEM: Use context to answer. Cite sources.              │
│    │     CONTEXT: [chunk1] [chunk2] [chunk3]                       │
│    │     USER: What is the return policy?                          │
│    │                                                                │
│    └─4. LLM response:                                              │
│          "You can return items within 30 days, in original          │
│           packaging. Refunds take 5-7 business days. [Source:      │
│           returns_policy.pdf]"                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: Document Ingestion

### Loading Documents

```python
# ============================================================
# LOADING DOCUMENTS FROM VARIOUS SOURCES
# ============================================================

# ── PDFs ──
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("contract.pdf")
pages = loader.load()    # Returns list of Document objects

# Each page is a Document with:
#   page.page_content = the text on that page
#   page.metadata = {"source": "contract.pdf", "page": 3}
for page in pages[:2]:
    print(f"Page {page.metadata['page']}: {page.page_content[:100]}...")

# Load entire folder of PDFs
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./documents/",       # Folder path
    glob="**/*.pdf",      # Match all PDFs recursively
    loader_cls=PyPDFLoader
)
all_docs = loader.load()
print(f"Loaded {len(all_docs)} pages from {len(set(d.metadata['source'] for d in all_docs))} files")

# ── Word Documents ──
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("report.docx")
docs = loader.load()

# ── HTML / Web Pages ──
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Load a web page and extract clean text
loader = WebBaseLoader(
    web_paths=["https://docs.python.org/3/tutorial/"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(class_=("body"))  # Extract only the main body
    }
)
web_docs = loader.load()

# ── CSV Files ──
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="products.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["product_id", "name", "description", "price"]
    }
)
csv_docs = loader.load()
# Each row becomes its own Document with metadata["row"] = row_number

# ── Plain Text ──
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt", encoding="utf-8")
text_docs = loader.load()
```

### Chunking — The Most Important RAG Decision

```python
# ============================================================
# CHUNKING STRATEGIES
#
# WHY chunking matters:
#   Too large  → chunk has noise, wastes context window, worse retrieval
#   Too small  → loses surrounding context, answer is incomplete
#
# The sweet spot: 300-800 characters for most use cases
# ============================================================

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

sample_text = """
Welcome to AcmeCorp's employee handbook.

Section 1: Working Hours
Our standard working hours are 9am to 6pm, Monday through Friday.
Employees may work flexible hours with manager approval.
Remote work is allowed up to 3 days per week.

Section 2: Vacation Policy
All full-time employees receive 15 days of paid vacation per year.
Vacation days accrue at 1.25 days per month.
Unused days may be carried over up to a maximum of 10 days.

Section 3: Code of Conduct
All employees must maintain professional conduct at all times.
Harassment of any kind will not be tolerated.
"""

# ── STRATEGY 1: Recursive Character Splitting (RECOMMENDED) ──
# Tries to split on: \n\n, then \n, then " ", then ""
# This keeps paragraphs together as much as possible

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # Target ~300 characters per chunk
    chunk_overlap=50,        # 50 characters overlap between chunks
                             # Overlap ensures no information is lost at boundaries
    separators=[
        "\n\n",              # Split on paragraphs first (preferred)
        "\n",                # Then on lines
        ". ",                # Then on sentences
        " ",                 # Then on words
        "",                  # Last resort: split mid-word
    ]
)

chunks = recursive_splitter.split_text(sample_text)
print(f"Recursive splitter: {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):\n{chunk[:100]}...")

# ── STRATEGY 2: Fixed Character Splitting ──
# Simple but can split mid-sentence; use overlap to compensate

fixed_splitter = CharacterTextSplitter(
    separator="\n",      # Split on newlines
    chunk_size=300,
    chunk_overlap=50
)
fixed_chunks = fixed_splitter.split_text(sample_text)

# ── STRATEGY 3: Semantic Chunking (Best Quality, More Expensive) ──
# Splits at semantic boundaries (topic changes) rather than arbitrary positions
# Uses embeddings to detect when the topic shifts

semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # Split when similarity drops below percentile
    breakpoint_threshold_amount=95           # Split at 95th percentile jumps
)

# Note: This calls the embeddings API, so it has a cost
semantic_chunks = semantic_splitter.split_text(sample_text)

# ── Chunk Size Guide ──
# Use Case                    chunk_size   overlap
# ─────────────────────────────────────────────────
# Precise Q&A                 200-400      50-100
# General knowledge base      400-800      50-100
# Long document summarisation 800-1500     100-200
# Legal/technical documents   500-1000     100-150
```

### Adding Metadata and Storing

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

def ingest_documents(
    file_paths: list[str],
    collection_name: str = "knowledge_base",
    persist_dir: str = "./chroma_db"
) -> Chroma:
    """
    Complete document ingestion pipeline:
    1. Load documents from files
    2. Split into chunks
    3. Add metadata
    4. Create embeddings
    5. Store in vector database

    Returns:
        Chroma vectorstore ready for querying
    """
    print("Step 1: Loading documents...")
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"  Loaded {len(docs)} pages from {path}")
        all_docs.extend(docs)

    print(f"\nStep 2: Splitting {len(all_docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)

    # Enrich metadata for better filtering later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)
        # source and page are already set by PyPDFLoader

    print(f"  Created {len(chunks)} chunks")

    print("\nStep 3: Creating embeddings and storing in vector DB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # from_documents embeds each chunk and stores in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir   # Save to disk for reuse
    )

    print(f"  Stored {vectorstore._collection.count()} chunks in vector DB")
    return vectorstore
```

---

## 5. Phase 2: Retrieval

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load existing vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ── Basic similarity search ──
query = "What is the vacation policy?"
results = vectorstore.similarity_search(query, k=3)

for doc in results:
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Text: {doc.page_content[:200]}")
    print("---")

# ── Similarity search with scores ──
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results_with_scores:
    # Score is L2 distance (lower = more similar for Chroma default)
    similarity = 1 / (1 + score)   # Convert to 0-1 range
    print(f"Similarity: {similarity:.3f} | {doc.page_content[:100]}")

# ── Metadata filtering ──
# Only search within a specific document
results = vectorstore.similarity_search(
    query,
    k=3,
    filter={"source": "employee_handbook.pdf"}
)

# ── MMR (Maximal Marginal Relevance) ──
# Returns diverse results — avoids returning 3 nearly identical chunks
results = vectorstore.max_marginal_relevance_search(
    query,
    k=3,           # Return 3 results
    fetch_k=20,    # Fetch top 20 first, then pick 3 most diverse
    lambda_mult=0.5  # 0 = max diversity, 1 = max relevance
)
```

---

## 6. Phase 3: Generation

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Build the RAG chain ──
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Custom prompt ensures:
# 1. Model uses ONLY the provided context
# 2. Model says "I don't know" if context doesn't have answer
# 3. Responses cite the source
RAG_PROMPT = """You are a helpful assistant answering questions about our company.
Answer ONLY based on the provided context.
If the context does not contain enough information to answer, say:
"I don't have that information in my knowledge base."

CONTEXT:
{context}

QUESTION: {question}

ANSWER (be concise, cite sources when possible):"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",   # "stuff" = put all retrieved chunks in one prompt
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True   # Include which docs were used
)

# ── Query ──
result = qa_chain.invoke({"query": "What is the vacation policy?"})

print("ANSWER:", result["result"])
print("\nSOURCES:")
for doc in result["source_documents"]:
    print(f"  - {doc.metadata.get('source')} (page {doc.metadata.get('page', '?')})")
```

---

## 7. Complete RAG From Scratch (No Framework)

```python
"""
Build a complete RAG system from scratch — no LangChain.
This shows exactly what's happening under the hood.
"""

import os
import json
from openai import OpenAI
import chromadb
from typing import List, Dict

openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./scratch_rag_db")


def get_embedding(text: str) -> List[float]:
    """Convert text to a vector using OpenAI embeddings."""
    response = openai_client.embeddings.create(
        input=[text.replace("\n", " ")],   # Clean text
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap   # Step size with overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


def index_documents(documents: List[Dict[str, str]], collection_name: str) -> chromadb.Collection:
    """
    Index documents into the vector database.

    Args:
        documents: List of {"text": "...", "source": "...", "metadata": {...}}
        collection_name: Name for the Chroma collection

    Returns:
        Chroma collection ready for search
    """
    # Get or create collection
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    collection = chroma_client.create_collection(collection_name)

    all_texts, all_ids, all_embeddings, all_metadatas = [], [], [], []
    doc_id = 0

    for doc in documents:
        # Split document into chunks
        chunks = chunk_text(doc["text"])
        print(f"  '{doc['source']}': {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks):
            # Create embedding for this chunk
            embedding = get_embedding(chunk)

            all_texts.append(chunk)
            all_ids.append(f"doc_{doc_id}")
            all_embeddings.append(embedding)
            all_metadatas.append({
                "source": doc["source"],
                "chunk_index": chunk_idx,
                **doc.get("metadata", {})
            })
            doc_id += 1

    # Batch insert into Chroma
    collection.add(
        documents=all_texts,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )

    print(f"  Total: {collection.count()} chunks indexed")
    return collection


def retrieve(
    query: str,
    collection: chromadb.Collection,
    k: int = 3,
    min_similarity: float = 0.3
) -> List[Dict]:
    """
    Find the most relevant document chunks for a query.

    Args:
        query: User's question
        collection: Chroma collection to search
        k: Number of results to return
        min_similarity: Minimum similarity threshold (0-1)

    Returns:
        List of {"text": "...", "source": "...", "similarity": 0.85}
    """
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances", "metadatas"]
    )

    retrieved = []
    for text, distance, metadata in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        # Chroma returns L2 distance; convert to similarity
        similarity = 1 / (1 + distance)
        if similarity >= min_similarity:
            retrieved.append({
                "text": text,
                "source": metadata.get("source", "unknown"),
                "similarity": round(similarity, 3),
                "metadata": metadata
            })

    return retrieved


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """
    Generate an answer using retrieved context.

    Args:
        query: User's question
        context_chunks: Retrieved relevant document chunks

    Returns:
        AI-generated answer grounded in context
    """
    if not context_chunks:
        return "I don't have enough information to answer that question."

    # Format context with source attribution
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n[Source {i} — {chunk['source']}]:\n{chunk['text']}\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions ONLY using the "
                "provided sources. If the answer is not in the sources, say "
                "'I don't have that information in my knowledge base.'\n"
                "Always cite your source as [Source N] when referencing content."
            )
        },
        {
            "role": "user",
            "content": f"Sources:{context_text}\n\nQuestion: {query}"
        }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0    # Deterministic for factual Q&A
    )

    return response.choices[0].message.content


def rag_query(
    query: str,
    collection: chromadb.Collection,
    k: int = 3
) -> Dict:
    """
    Complete RAG pipeline: retrieve then generate.

    Returns:
        {"answer": "...", "sources": [...], "num_chunks_used": N}
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, collection, k=k)

    # Step 2: Generate answer using chunks
    answer = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": answer,
        "sources": [c["source"] for c in chunks],
        "num_chunks_used": len(chunks),
        "chunks": chunks   # Include for debugging
    }


# ── Example usage ──
docs = [
    {
        "text": """
        Return Policy:
        We offer a 30-day return policy on all items. Products must be in
        their original packaging and unused condition. To initiate a return,
        contact customer service at returns@acmecorp.com with your order number.
        Refunds are processed within 5-7 business days to the original payment method.
        """,
        "source": "return_policy.txt"
    },
    {
        "text": """
        Shipping Information:
        Standard shipping takes 5-7 business days and costs $4.99.
        Free shipping is available on orders over $50.
        Express shipping (2-3 business days) costs $12.99.
        International shipping is available to 40+ countries.
        """,
        "source": "shipping_info.txt"
    },
]

print("Indexing documents...")
collection = index_documents(docs, "customer_service_kb")

# Query the system
questions = [
    "Can I return something I bought 3 weeks ago?",
    "How much is express shipping?",
    "What payment methods do you accept?",  # Not in docs — should say "don't know"
]

for question in questions:
    result = rag_query(question, collection)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
```

---

## 8. Advanced RAG: HyDE

**Hypothetical Document Embeddings** — generate a hypothetical answer, then use THAT for search.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def hyde_search(
    query: str,
    collection,
    k: int = 3
) -> list:
    """
    HyDE: Generate a hypothetical document that answers the query,
    then use its embedding to search the knowledge base.

    WHY this helps:
    - Query: "What is the return deadline?" → short, sparse vector
    - Hypothetical doc: "You can return items within 30 days..." → rich vector
    - Rich vector matches real documents better than sparse query

    Args:
        query: User's question
        collection: Chroma collection
        k: Number of results

    Returns:
        Retrieved documents
    """
    # Step 1: Generate hypothetical answer (ignoring actual knowledge)
    hypo_response = client.chat.completions.create(
        model="gpt-4o-mini",    # Use cheap model for hypothesis
        messages=[{
            "role": "user",
            "content": (
                f"Write a brief passage that would answer this question. "
                f"Do not say 'I don't know' — write as if you know the answer.\n\n"
                f"Question: {query}"
            )
        }],
        max_tokens=200,
        temperature=0.5
    )

    hypothetical_document = hypo_response.choices[0].message.content
    print(f"Hypothetical document: {hypothetical_document[:100]}...")

    # Step 2: Embed the hypothetical document (not the original query)
    response = client.embeddings.create(
        input=[hypothetical_document],
        model="text-embedding-3-small"
    )
    hypo_embedding = response.data[0].embedding

    # Step 3: Search with hypothetical embedding
    results = collection.query(
        query_embeddings=[hypo_embedding],
        n_results=k
    )

    return results["documents"][0]
```

---

## 9. Advanced RAG: Multi-Query Retrieval

```python
def multi_query_retrieve(
    query: str,
    collection,
    k_per_query: int = 3
) -> list[str]:
    """
    Generate multiple query variations → retrieve for each → merge results.

    WHY this helps:
    Single query may miss relevant docs because of vocabulary mismatch.
    Multiple queries with different phrasing cast a wider net.

    Example:
      Original: "return policy"
      Generated also: "refund rules", "how to send back a product",
                      "money back guarantee terms"
    """
    # Step 1: Generate query variations
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Generate 3 different ways to ask this question. "
                f"Return ONLY the 3 questions, one per line.\n\n"
                f"Original question: {query}"
            )
        }],
        temperature=0.7,
        max_tokens=200
    )

    # Parse the 3 generated queries
    generated_queries = [
        line.strip()
        for line in response.choices[0].message.content.strip().split("\n")
        if line.strip()
    ][:3]

    all_queries = [query] + generated_queries
    print(f"Queries: {all_queries}")

    # Step 2: Retrieve for each query
    all_docs = set()   # Use set to deduplicate

    for q in all_queries:
        results = collection.query(
            query_texts=[q],
            n_results=k_per_query
        )
        for doc in results["documents"][0]:
            all_docs.add(doc)   # Deduplication happens here

    return list(all_docs)
```

---

## 10. RAG Evaluation with RAGAS

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,         # Is answer supported by context?
    answer_relevancy,     # Does answer address the question?
    context_precision,    # Are retrieved chunks relevant?
    context_recall,       # Did we retrieve all needed chunks?
)
from datasets import Dataset

def evaluate_rag_system(
    test_questions: list[str],
    ground_truth_answers: list[str],
    rag_system_fn,    # Your RAG function: question -> {"answer": ..., "contexts": [...]}
) -> dict:
    """
    Evaluate RAG quality using RAGAS metrics.

    Metrics:
      faithfulness     (0-1): Is every claim in the answer supported by context?
      answer_relevancy (0-1): How relevant is the answer to the question?
      context_precision (0-1): Are all retrieved contexts relevant?
      context_recall   (0-1): Did we retrieve all info needed to answer?

    Args:
        test_questions: List of test questions
        ground_truth_answers: The ideal correct answers
        rag_system_fn: Your RAG function to evaluate

    Returns:
        Dictionary of metric scores
    """
    # Run your RAG system on all test questions
    answers = []
    contexts = []

    for question in test_questions:
        result = rag_system_fn(question)
        answers.append(result["answer"])
        contexts.append(result["contexts"])   # List of retrieved text chunks

    # Build evaluation dataset
    eval_data = {
        "question":  test_questions,
        "answer":    answers,
        "contexts":  contexts,        # List of lists of retrieved chunks
        "ground_truth": ground_truth_answers,
    }

    dataset = Dataset.from_dict(eval_data)

    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    return results.to_pandas().to_dict("records")[0]


# Example evaluation
def my_rag(question: str) -> dict:
    """Your RAG system."""
    chunks = retrieve(question, collection, k=3)
    answer = generate_answer(question, chunks)
    return {
        "answer": answer,
        "contexts": [c["text"] for c in chunks]
    }

test_qs = [
    "What is the return policy?",
    "How long does shipping take?",
]
ground_truths = [
    "Returns are allowed within 30 days of purchase.",
    "Standard shipping takes 5-7 business days.",
]

scores = evaluate_rag_system(test_qs, ground_truths, my_rag)
print("\nRAG Evaluation Results:")
for metric, score in scores.items():
    print(f"  {metric}: {score:.3f}")
```

---

## 11. Common RAG Failures and Fixes

```
FAILURE 1: "My RAG returns wrong answers"
  Diagnosis: Poor retrieval quality
  Fixes:
    - Reduce chunk size (smaller = more precise retrieval)
    - Increase k (retrieve more candidates)
    - Use reranking (Cohere Rerank, BGE)
    - Use HyDE for better query representation
    - Check embedding model quality

FAILURE 2: "RAG says 'I don't know' but the answer is in my docs"
  Diagnosis: Retrieval is missing relevant chunks
  Fixes:
    - Check chunk size isn't too large
    - Try multi-query retrieval
    - Verify documents were indexed correctly
    - Check metadata filters aren't too restrictive
    - Verify embedding dimensions match index

FAILURE 3: "Answers are correct but too verbose/wrong format"
  Diagnosis: Prompt template issue
  Fixes:
    - Add format instructions to prompt
    - Specify max length in prompt
    - Use structured output / JSON mode

FAILURE 4: "RAG hallucinates things not in context"
  Diagnosis: LLM ignoring the context grounding instruction
  Fixes:
    - Stronger system prompt: "Answer ONLY from context"
    - Set temperature=0
    - Use shorter, more relevant chunks (less noise)
    - Add "If not in context, say so" instruction explicitly

FAILURE 5: "Too slow at indexing"
  Diagnosis: Embedding large document set one-by-one
  Fixes:
    - Batch embeddings (embed 100 at a time, not 1)
    - Use faster embedding model (text-embedding-3-small)
    - Process documents asynchronously
    - Cache embeddings (don't re-embed unchanged docs)
```

---

## Key Points for Exam Prep

```
RAG CHEAT SHEET:
  - RAG = Retrieve relevant context + Augment the prompt + Generate answer
  - Two phases: indexing (offline) and querying (per request)
  - Better than fine-tuning for: private data, frequent updates, citations
  - Chunk size 300-800 chars + 10-20% overlap = good starting point
  - Recursive text splitting: best default chunking strategy
  - Similarity search returns top-K most relevant chunks
  - Always include "only use context" in the prompt
  - RAGAS: 4 metrics — faithfulness, relevancy, precision, recall
  - HyDE: embed hypothetical answer, not raw query
  - Multi-query: generate variations, merge results
  - Reranking: retrieve 20, rerank to 3 (better quality)
```

## Practice Questions

1. What are the two main phases of a RAG system?
2. Why is chunking strategy called the "most important RAG decision"?
3. What does chunk_overlap do and why is it important?
4. When would you choose RAG over fine-tuning?
5. What is MMR and when would you use it?
6. What is HyDE and why does it improve retrieval?
7. What does "faithfulness" measure in RAGAS?
8. What is multi-query retrieval and when does it help?
9. What metadata should you store alongside chunks?
10. Why should you set temperature=0 in RAG generation?
11. What is the difference between similarity_search and max_marginal_relevance_search?
12. How would you handle multi-tenant RAG (different users, different data)?
13. How do you update a RAG system when documents change?
14. What is contextual compression and when is it useful?
15. Design a RAG evaluation pipeline from scratch.

---
*Next: [10 — Tools & Function Calling](../10-tools-function-calling/README.md)*
