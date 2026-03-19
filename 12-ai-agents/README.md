# 12 — AI Agents: Building Autonomous AI Systems

## Agent vs Simple LLM Call: The Analogy

```
SIMPLE LLM CALL                     AI AGENT
═══════════════════════════════════ ══════════════════════════════════════
Like asking a friend ONE question   Like hiring an employee to do a project

You:    "What is photosynthesis?"   You:    "Research, summarize, and write
                                             a report on photosynthesis and
Friend: [answers immediately]               how it could be used in solar
                                             energy"

One round trip.                     Employee: researches, reads sources,
One question, one answer.                     checks facts, drafts sections,
No decision making required.                  revises, formats the report.

  USER ──> LLM ──> ANSWER           USER ──> AGENT ──> [LOOP] ──> REPORT
                                                │
                                           ┌────▼────┐
                                           │ Think   │
                                           │ Plan    │
                                           │ Search  │
                                           │ Write   │
                                           │ Revise  │
                                           └─────────┘

WHEN TO USE EACH:
  Simple LLM call: Single-step tasks, Q&A, translation, summarization
  Agent:           Multi-step tasks, research, planning, tool-using workflows
```

---

## 4 Core Agent Components

```
AGENT ARCHITECTURE
============================================================

              ┌─────────────────────────────────────────┐
              │                 AGENT                    │
              │                                          │
              │  ┌──────────┐      ┌──────────────────┐ │
              │  │          │      │                  │ │
  USER ──────>│  │  BRAIN   │<────>│  MEMORY          │ │
  TASK        │  │          │      │                  │ │
              │  │ (LLM:    │      │ working memory   │ │
              │  │  gpt-4o, │      │ episodic memory  │ │
              │  │  claude, │      │ semantic memory  │ │
              │  │  etc.)   │      │ procedural memory│ │
              │  │          │      └──────────────────┘ │
              │  │  Reasons │                            │
              │  │  Plans   │      ┌──────────────────┐ │
              │  │  Decides │      │                  │ │
              │  │          │<────>│  TOOLS           │ │
              │  └──────────┘      │                  │ │
              │       │            │ search_web()     │ │
              │       │            │ run_code()       │ │
              │  ┌────▼────────┐   │ read_file()      │ │
              │  │             │   │ call_api()       │ │
              │  │  PLANNING   │   └──────────────────┘ │
              │  │             │                         │
              │  │ ReAct       │                         │
              │  │ Plan-Exec   │                         │
              │  │ Tree-of-    │                         │
              │  │ Thoughts    │                         │
              │  └─────────────┘                        │
              └─────────────────────────────────────────┘
                            │
                      ┌─────▼──────┐
                      │  ANSWER /  │
                      │  ACTION    │
                      └────────────┘

1. BRAIN   — The LLM. Reads context, reasons, and decides what to do next.
2. MEMORY  — Different stores for different types of information.
3. TOOLS   — Functions the agent can call to affect the world.
4. PLANNING — The strategy for deciding what steps to take.
```

---

## ReAct Pattern: Deep Dive

```
REACT = REASON + ACT
============================================================

ReAct interleaves three types of steps:
  THOUGHT  — the model's internal reasoning (not shown to user)
  ACTION   — calling a tool or taking an action
  OBSERVATION — the result of the action

EXAMPLE TRACE — "What is the population of the largest city in Japan?"
════════════════════════════════════════════════════════════════════

Thought 1: I need to find the largest city in Japan by population.
           I should search for this.
Action 1:  search_web("largest city in Japan by population")
Observation 1: Tokyo is the largest city in Japan with a population
               of approximately 13.96 million (city proper) and
               37.4 million in the greater metropolitan area.

Thought 2: I found the answer. The largest city is Tokyo and its
           population is 13.96 million in the city proper.
           The user asked about the city so I'll use city proper.
Action 2:  [FINISH - give final answer]
Observation 2: N/A

Final Answer: The largest city in Japan is Tokyo, with a population
              of approximately 13.96 million people in the city proper.

KEY PROPERTY: The model's THOUGHT is visible in the context window
but not shown to the user. This "chain of thought" allows the model
to reason step by step before acting.
```

```python
# ============================================================
# REACT PATTERN — COMPLETE WORKING IMPLEMENTATION
# ============================================================

import json
import re
from openai import OpenAI

client = OpenAI()


# ---- Tools available to the agent ----

def search_web(query: str) -> str:
    """Mock web search — in production call a real search API."""
    results = {
        "largest city in Japan":     "Tokyo is Japan's largest city with 13.96M people.",
        "capital of France":         "Paris is the capital of France, population 2.1M.",
        "Python programming":        "Python is a high-level programming language by Guido van Rossum.",
        "weather in Tokyo":          "Tokyo is currently 22°C and sunny.",
        "population of New York":    "New York City has a population of about 8.3 million.",
    }
    # Find the best match in our mock results
    for key, value in results.items():
        if any(word.lower() in query.lower() for word in key.split()):
            return value
    return f"No results found for: {query}"


def calculate(expression: str) -> str:
    """Safe math evaluator."""
    import ast, operator
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        raise ValueError("Unsupported expression")
    try:
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


# Registry maps tool names to functions
TOOL_REGISTRY = {
    "search_web": search_web,
    "calculate":  calculate,
}

# Tool descriptions for the system prompt
TOOLS_DESCRIPTION = """
Available tools (call them using the exact JSON format shown):

1. search_web(query: str) -> str
   Search the internet for current information.
   Call: {"tool": "search_web", "query": "your search terms"}

2. calculate(expression: str) -> str
   Evaluate a mathematical expression.
   Call: {"tool": "calculate", "expression": "2 + 2 * 10"}
"""


def run_react_agent(user_task: str, max_steps: int = 8) -> str:
    """
    Execute a task using the ReAct (Reason + Act) pattern.

    The model alternates between:
      THOUGHT:      Reasoning about what to do next
      ACTION:       Calling a tool (JSON format)
      OBSERVATION:  The tool's result (added to context)

    This continues until the model produces FINAL ANSWER: ...

    Args:
        user_task: The user's question or task
        max_steps: Safety limit on tool calls

    Returns:
        The final answer string
    """

    # System prompt instructs the model to use ReAct format
    system_prompt = f"""You are a helpful AI agent. Solve tasks step by step.

{TOOLS_DESCRIPTION}

FORMAT:
For every step, respond in EXACTLY this format:

Thought: [your reasoning about what to do next]
Action: {{"tool": "tool_name", "param": "value"}}

OR when you have the final answer:

Thought: [I now have enough information to answer]
Final Answer: [your complete answer to the user]

RULES:
- Always show your Thought before any Action
- Always use valid JSON for actions
- After each Observation, continue with another Thought
- Only use Final Answer when you are confident in the answer
"""

    # Start with just the user's task
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_task},
    ]

    step_count = 0

    while step_count < max_steps:
        step_count += 1
        print(f"\n[Step {step_count}]")

        # Ask the model what to do next
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,    # Deterministic reasoning
            stop=None,        # Don't stop early
        )

        agent_output = response.choices[0].message.content
        print(agent_output)

        # Add model's output to the conversation
        messages.append({"role": "assistant", "content": agent_output})

        # Check if model gave a final answer
        final_match = re.search(r"Final Answer:\s*(.+)", agent_output, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()

        # Extract action from the model's output
        # Look for JSON after "Action:"
        action_match = re.search(r"Action:\s*(\{.+?\})", agent_output, re.DOTALL)
        if not action_match:
            # Model didn't produce an action — prompt it to continue
            messages.append({
                "role": "user",
                "content": "Please continue. Use an Action or provide Final Answer."
            })
            continue

        # Parse and execute the action
        try:
            action = json.loads(action_match.group(1))
            tool_name = action.pop("tool")   # Remove 'tool' key; rest are args

            if tool_name not in TOOL_REGISTRY:
                observation = f"Error: Unknown tool '{tool_name}'"
            else:
                # Execute the tool with remaining args
                observation = TOOL_REGISTRY[tool_name](**action)

        except json.JSONDecodeError as e:
            observation = f"Error: Could not parse action JSON — {e}"
        except TypeError as e:
            observation = f"Error: Wrong arguments for {tool_name} — {e}"

        print(f"Observation: {observation}")

        # Add observation back to the conversation for the model to see
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    # Safety: max steps reached
    return "I could not complete this task within the step limit. Please rephrase."


# Test the ReAct agent
print("=" * 60)
answer = run_react_agent("What is the population of the largest city in Japan?")
print(f"\nFINAL ANSWER: {answer}")

print("\n" + "=" * 60)
answer2 = run_react_agent(
    "If Tokyo has 13.96 million people and grows 0.5% per year, "
    "what will the population be in 10 years?"
)
print(f"\nFINAL ANSWER: {answer2}")
```

---

## 4 Memory Types with Code

```
MEMORY TYPES IN AI AGENTS
============================================================

1. WORKING MEMORY (in-context)
   What's in the current context window.
   Temporary — lost when the conversation ends.
   Fast, immediate, limited by context length.

   Example: The conversation history, current task, tool outputs.

2. EPISODIC MEMORY (conversation history)
   Records of past interactions, stored externally.
   Persistent across sessions, searchable.
   "Remember what we talked about last week"

3. SEMANTIC MEMORY (knowledge base / vector store)
   General facts and knowledge, stored as embeddings.
   Searched by similarity (RAG).
   "What are the company's refund policies?"

4. PROCEDURAL MEMORY (learned behaviors)
   How to do things — stored as prompts, fine-tuning, or code.
   Changes slowly (requires retraining or prompt updates).
   "Always respond in formal English" (in system prompt)
```

```python
# ============================================================
# MEMORY IMPLEMENTATIONS
# ============================================================

import json
import os
from openai import OpenAI
from datetime import datetime

client = OpenAI()


# ============================================================
# 1. WORKING MEMORY — the messages list (built-in to all LLM apps)
# ============================================================

class WorkingMemory:
    """
    The simplest memory: the conversation history in the context window.

    Limitations:
    - Lost when the conversation ends (no persistence)
    - Limited by model's context window (e.g., 128K tokens for GPT-4o)
    - Old messages need to be summarized or dropped when context fills up
    """

    def __init__(self, system_prompt: str, max_messages: int = 20):
        self.system_prompt = system_prompt
        self.max_messages  = max_messages
        self.messages: list[dict] = []

    def add(self, role: str, content: str):
        """Add a message to working memory."""
        self.messages.append({"role": role, "content": content})

        # When we have too many messages, summarize the oldest ones
        if len(self.messages) > self.max_messages:
            self._compress()

    def _compress(self):
        """
        Summarize the oldest half of the conversation to save context space.
        This is a simple version — production systems use more sophisticated
        sliding-window or hierarchical summarization.
        """
        # Keep the first N messages to summarize
        oldest = self.messages[:self.max_messages // 2]
        self.messages = self.messages[self.max_messages // 2:]

        # Ask the model to summarize the old messages
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following conversation excerpt in 2-3 sentences."
                },
                {
                    "role": "user",
                    "content": json.dumps(oldest)
                }
            ],
        )
        summary = summary_response.choices[0].message.content

        # Prepend the summary as a system message
        self.messages.insert(0, {
            "role":    "system",
            "content": f"[Earlier conversation summary]: {summary}"
        })

    def get_messages(self) -> list[dict]:
        """Return the full message list including the system prompt."""
        return [{"role": "system", "content": self.system_prompt}] + self.messages


# ============================================================
# 2. EPISODIC MEMORY — long-term conversation storage
# ============================================================

class EpisodicMemory:
    """
    Persists conversation episodes across sessions using simple JSON storage.

    In production, use a database (PostgreSQL, MongoDB) and implement
    similarity search to retrieve relevant past episodes.
    """

    def __init__(self, storage_path: str = "memory/episodes.json"):
        self.storage_path = storage_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        self.episodes = self._load()

    def _load(self) -> list[dict]:
        """Load episodes from disk."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path) as f:
                return json.load(f)
        return []

    def _save(self):
        """Persist episodes to disk."""
        with open(self.storage_path, "w") as f:
            json.dump(self.episodes, f, indent=2)

    def save_episode(self, user_id: str, messages: list[dict], summary: str):
        """
        Save a conversation episode.

        Args:
            user_id: Identifier for the user (use a hashed ID in production)
            messages: The conversation messages
            summary: A short summary of what was discussed
        """
        episode = {
            "id":        len(self.episodes) + 1,
            "user_id":   user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary":   summary,
            "messages":  messages,    # In production, store only the summary + key facts
        }
        self.episodes.append(episode)
        self._save()

    def recall(self, user_id: str, limit: int = 5) -> list[dict]:
        """
        Retrieve the most recent episodes for a user.

        In production, add semantic search to retrieve the most RELEVANT
        episodes, not just the most recent ones.
        """
        user_episodes = [e for e in self.episodes if e["user_id"] == user_id]
        # Return summaries of the N most recent episodes
        return [
            {"timestamp": e["timestamp"], "summary": e["summary"]}
            for e in user_episodes[-limit:]
        ]

    def format_for_context(self, user_id: str) -> str:
        """Format past episodes as a string to inject into the system prompt."""
        episodes = self.recall(user_id)
        if not episodes:
            return ""
        lines = ["Past conversation summaries:"]
        for ep in episodes:
            lines.append(f"  [{ep['timestamp'][:10]}] {ep['summary']}")
        return "\n".join(lines)


# ============================================================
# 3. SEMANTIC MEMORY — vector store (knowledge base / RAG)
# ============================================================

class SemanticMemory:
    """
    Store facts as embeddings; retrieve by semantic similarity.

    This is the foundation of RAG (Retrieval-Augmented Generation).
    In production, use: Pinecone, Weaviate, Chroma, pgvector, Qdrant.
    Here we implement a minimal in-memory version.
    """

    def __init__(self):
        self.memories: list[dict] = []   # [{text: str, embedding: list[float]}]

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for text using OpenAI's embedding model.
        The vector captures the semantic meaning of the text.
        """
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def remember(self, text: str, metadata: dict = None):
        """
        Store a fact or piece of information with its embedding.

        Args:
            text: The fact to remember
            metadata: Optional additional info (source, timestamp, etc.)
        """
        embedding = self.embed(text)
        self.memories.append({
            "text":      text,
            "embedding": embedding,
            "metadata":  metadata or {},
        })

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors (range: -1 to 1)."""
        import math
        dot   = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a**2 for a in vec_a))
        mag_b = math.sqrt(sum(b**2 for b in vec_b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the most semantically similar memories to a query.

        Args:
            query: The question or topic to search for
            top_k: Number of most relevant memories to return

        Returns:
            List of {text, similarity_score} dicts, sorted by relevance
        """
        if not self.memories:
            return []

        query_embedding = self.embed(query)

        # Score every stored memory against the query
        scored = [
            {
                "text":       mem["text"],
                "metadata":   mem["metadata"],
                "similarity": self.cosine_similarity(query_embedding, mem["embedding"]),
            }
            for mem in self.memories
        ]

        # Sort by similarity descending, return top K
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]


# ============================================================
# 4. PROCEDURAL MEMORY — stored instructions / system prompts
# ============================================================

class ProceduralMemory:
    """
    Store and retrieve behavioral instructions for the agent.

    Procedural memories are "how to do things" — they are injected
    into the system prompt to shape the agent's behavior.
    """

    def __init__(self):
        # Default procedures — can be extended at runtime
        self.procedures: dict[str, str] = {
            "default_tone": (
                "Always respond in a professional, helpful tone. "
                "Be concise — no more than 3 paragraphs unless the user asks for more."
            ),
            "safety": (
                "Never provide instructions for illegal activities, "
                "self-harm, or content that could endanger others."
            ),
            "uncertainty": (
                "If you are not confident in an answer, say so explicitly. "
                "Acknowledge your limitations rather than guessing."
            ),
        }

    def add_procedure(self, name: str, instruction: str):
        """Add or update a behavioral procedure."""
        self.procedures[name] = instruction

    def get_system_prompt_section(self) -> str:
        """Format all procedures as a system prompt section."""
        return "Behavioral guidelines:\n" + "\n".join(
            f"- {v}" for v in self.procedures.values()
        )
```

---

## Planning Strategies

```
PLANNING STRATEGIES COMPARISON
============================================================

STRATEGY 1: REACT (Reason + Act)
  Best for: Simple to medium tasks, tool-using, open-ended research
  Style: Interleaved thought + action + observation
  Limitation: Can get stuck in loops; no global plan

  User: "Find the CEO of Apple and their net worth"
  → Think → Search → Observe → Think → Search → Observe → Answer

STRATEGY 2: PLAN-AND-EXECUTE
  Best for: Complex multi-step tasks with clear sub-goals
  Style: Create full plan upfront, then execute each step
  Limitation: Plan may be wrong; hard to adapt to new information

  User: "Write a 5-page research report on climate change"
  Plan:
    1. Search for recent climate change statistics
    2. Find information on causes
    3. Find information on effects
    4. Find proposed solutions
    5. Write introduction
    6. Write body sections
    7. Write conclusion
    8. Format the report
  Execute: [run each step in sequence]

STRATEGY 3: TREE OF THOUGHTS
  Best for: Creative problems, math puzzles, strategy
  Style: Generate multiple reasoning paths, evaluate each, pick best
  Limitation: More expensive (many LLM calls); slower

  User: "Plan the best route from NYC to LA stopping at 3 cities"
  Tree:
    Root: NYC
    ├── Path A: NYC → Chicago → Denver → Phoenix → LA
    │   Score: 8/10 (good spread, scenic)
    ├── Path B: NYC → Nashville → Austin → Phoenix → LA
    │   Score: 7/10 (warmer route)
    └── Path C: NYC → DC → Atlanta → Houston → LA
        Score: 6/10 (longer southern route)
  Best: Path A (highest score)
```

```python
# ============================================================
# PLAN-AND-EXECUTE AGENT
# First creates a plan, then executes each step
# ============================================================

import json
from openai import OpenAI

client = OpenAI()


def create_plan(task: str, available_tools: list[str]) -> list[str]:
    """
    Ask the model to create a step-by-step plan for a task.

    Returns a list of strings, each describing one step.
    The planner is intentionally kept separate from the executor.
    """
    tools_str = ", ".join(available_tools)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a task planner. Given a task and available tools, "
                    "create a numbered list of clear, concrete steps to complete it. "
                    f"Available tools: {tools_str}\n\n"
                    "Return ONLY a JSON array of step strings. Example:\n"
                    '["Step 1: Search for X", "Step 2: Calculate Y from results"]'
                ),
            },
            {"role": "user", "content": f"Task: {task}"},
        ],
        temperature=0.2,   # Low temperature for structured planning
    )

    content = response.choices[0].message.content
    try:
        plan = json.loads(content)
        return plan if isinstance(plan, list) else [content]
    except json.JSONDecodeError:
        # Fallback: extract numbered lines manually
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        return lines or [content]


def execute_step(
    step: str,
    task: str,
    context: list[dict],
    tool_registry: dict,
) -> str:
    """
    Execute a single step from the plan.

    The executor knows what the overall task is, what step it's on,
    and what has been accomplished so far (context).
    """
    context_text = "\n".join(
        f"  Step {i+1} result: {c['result']}"
        for i, c in enumerate(context)
    )

    # Build tool list for the prompt
    tools_json = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Execute: {name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input for the tool"}
                    },
                    "required": ["input"],
                },
            },
        }
        for name in tool_registry
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are executing step {len(context)+1} of a multi-step plan.\n"
                    f"Overall task: {task}\n"
                    f"Previous results:\n{context_text}\n\n"
                    f"Current step: {step}\n\n"
                    "Complete this step using the available tools or your knowledge."
                ),
            },
            {"role": "user", "content": f"Execute: {step}"},
        ],
        tools=tools_json if tools_json else None,
        tool_choice="auto" if tools_json else None,
    )

    message = response.choices[0].message

    if response.choices[0].finish_reason == "tool_calls":
        results = []
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            tool = tool_registry.get(tc.function.name)
            if tool:
                result = tool(args.get("input", ""))
                results.append(result)
        return "; ".join(results)

    return message.content or "No result produced"


def run_plan_and_execute(task: str, tool_registry: dict) -> str:
    """
    Run the full Plan-and-Execute cycle.

    1. Create a plan (list of steps)
    2. Execute each step sequentially, collecting results
    3. Synthesize all results into a final answer
    """
    available_tools = list(tool_registry.keys())

    print(f"\nTask: {task}")
    print("Creating plan...")

    plan = create_plan(task, available_tools)
    print(f"Plan ({len(plan)} steps):")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    context = []

    # Execute each step
    for i, step in enumerate(plan, 1):
        print(f"\nExecuting step {i}: {step}")
        result = execute_step(step, task, context, tool_registry)
        context.append({"step": step, "result": result})
        print(f"  Result: {result[:100]}")

    # Synthesize all results into a final answer
    synthesis_messages = [
        {
            "role": "system",
            "content": "Synthesize the following step results into a clear final answer.",
        },
        {
            "role": "user",
            "content": (
                f"Original task: {task}\n\n"
                + "\n".join(
                    f"Step {i+1} ({c['step']}): {c['result']}"
                    for i, c in enumerate(context)
                )
            ),
        },
    ]

    synthesis = client.chat.completions.create(
        model="gpt-4o", messages=synthesis_messages
    )
    return synthesis.choices[0].message.content


# Run the agent
result = run_plan_and_execute(
    "Research the population of Tokyo and calculate how many years until"
    " it reaches 15 million at 0.5% annual growth",
    tool_registry={"search_web": search_web, "calculate": calculate},
)
print(f"\n\nFINAL ANSWER:\n{result}")
```

---

## Multi-Agent Systems: Orchestrator-Worker Pattern

```
MULTI-AGENT ORCHESTRATOR-WORKER ARCHITECTURE
============================================================

                    ┌──────────────────────┐
                    │    ORCHESTRATOR       │
                    │                      │
  USER ────────────>│  - Receives task     │
                    │  - Decomposes task   │
                    │  - Assigns subtasks  │
                    │  - Collects results  │
                    │  - Synthesizes final │
                    │    answer            │
                    └──────────┬───────────┘
                               │ assigns subtasks
              ┌────────────────┼───────────────────┐
              │                │                   │
              ▼                ▼                   ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ RESEARCH AGENT   │ │ ANALYSIS AGENT   │ │ WRITING AGENT    │
   │                  │ │                  │ │                  │
   │ - search_web()   │ │ - calculate()    │ │ - format_text()  │
   │ - read_url()     │ │ - run_code()     │ │ - check_grammar()│
   │                  │ │                  │ │                  │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
              │                │                   │
              └────────────────┴───────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │    ORCHESTRATOR       │
                    │    collects results  │
                    │    synthesizes       │
                    └──────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │    FINAL ANSWER      │
                    │    to USER           │
                    └──────────────────────┘

BENEFITS:
  - Specialization: each agent is optimized for its domain
  - Parallelism: independent subtasks run simultaneously
  - Modularity: easy to swap out or add agents
  - Scalability: add more workers for throughput

CHALLENGES:
  - Orchestration complexity
  - Result aggregation / conflict resolution
  - Cost (multiple model calls)
  - Debugging (harder to trace through multiple agents)
```

```python
# ============================================================
# MULTI-AGENT ORCHESTRATOR-WORKER IMPLEMENTATION
# ============================================================

import asyncio
import json
from openai import AsyncOpenAI

async_client = AsyncOpenAI()


class SpecializedAgent:
    """A worker agent with a specific role and tools."""

    def __init__(
        self,
        name: str,
        role_description: str,
        tools: list[dict],
        tool_functions: dict,
    ):
        self.name           = name
        self.role           = role_description
        self.tools          = tools
        self.tool_functions = tool_functions

    async def execute(self, subtask: str) -> str:
        """
        Execute a subtask assigned by the orchestrator.

        Returns:
            The result of completing the subtask
        """
        messages = [
            {"role": "system", "content": f"You are a {self.role}. {self.name}."},
            {"role": "user",   "content": subtask},
        ]

        # Simple single-round agent loop (ReAct-style but async)
        for _ in range(5):   # Max 5 tool call rounds
            response = await async_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
            )

            message       = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "stop":
                return message.content

            if finish_reason == "tool_calls":
                messages.append(message)
                for tc in message.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    if name in self.tool_functions:
                        result = self.tool_functions[name](**args)
                    else:
                        result = {"error": f"Unknown tool: {name}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })

        return "Could not complete subtask in allotted steps."


class Orchestrator:
    """
    Decomposes tasks and coordinates multiple specialized agents.
    """

    def __init__(self, agents: dict[str, SpecializedAgent]):
        """
        Args:
            agents: Dictionary mapping agent names to SpecializedAgent instances
        """
        self.agents = agents

    async def decompose_task(self, task: str) -> list[dict]:
        """
        Use the LLM to break a task into subtasks for specific agents.

        Returns:
            List of {"agent": agent_name, "subtask": description} dicts
        """
        agent_descriptions = "\n".join(
            f"  {name}: {agent.role}"
            for name, agent in self.agents.items()
        )

        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a task orchestrator. Decompose the given task "
                        "into specific subtasks for the available agents.\n\n"
                        f"Available agents:\n{agent_descriptions}\n\n"
                        "Return ONLY a JSON array:\n"
                        '[{"agent": "agent_name", "subtask": "description"}, ...]'
                    ),
                },
                {"role": "user", "content": f"Task: {task}"},
            ],
            temperature=0,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback: assign entire task to the first available agent
            first_agent = next(iter(self.agents))
            return [{"agent": first_agent, "subtask": task}]

    async def run(self, task: str) -> str:
        """
        Orchestrate the full multi-agent task execution.

        1. Decompose the task into subtasks
        2. Execute independent subtasks in parallel
        3. Synthesize results into a final answer
        """
        print(f"\nOrchestrating: {task}")

        # Step 1: Decompose
        subtasks = await self.decompose_task(task)
        print(f"Decomposed into {len(subtasks)} subtasks:")
        for st in subtasks:
            print(f"  [{st['agent']}] {st['subtask']}")

        # Step 2: Execute in parallel (concurrent subtasks)
        async def execute_subtask(subtask_info: dict) -> dict:
            agent_name = subtask_info["agent"]
            subtask    = subtask_info["subtask"]
            if agent_name not in self.agents:
                return {"agent": agent_name, "result": f"Unknown agent: {agent_name}"}
            result = await self.agents[agent_name].execute(subtask)
            return {"agent": agent_name, "subtask": subtask, "result": result}

        # asyncio.gather runs all subtasks concurrently
        results = await asyncio.gather(*[execute_subtask(st) for st in subtasks])

        # Step 3: Synthesize
        results_text = "\n".join(
            f"[{r['agent']}] {r['subtask']}:\n{r['result']}\n"
            for r in results
        )

        synthesis = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Synthesize the following agent results into a coherent, "
                        "comprehensive final answer for the user."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Original task: {task}\n\nAgent results:\n{results_text}",
                },
            ],
        )

        return synthesis.choices[0].message.content


# Build specialized agents
research_agent = SpecializedAgent(
    name="ResearchAgent",
    role_description="expert at finding information using web search",
    tools=[{
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]},
        }
    }],
    tool_functions={"search_web": search_web},
)

analysis_agent = SpecializedAgent(
    name="AnalysisAgent",
    role_description="expert at mathematical analysis and data interpretation",
    tools=[{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate math expression",
            "parameters": {"type": "object", "properties": {
                "expression": {"type": "string"}}, "required": ["expression"]},
        }
    }],
    tool_functions={"calculate": calculate},
)

writing_agent = SpecializedAgent(
    name="WritingAgent",
    role_description="expert at clear, structured writing and formatting",
    tools=[],    # No tools — uses LLM capabilities directly
    tool_functions={},
)

orchestrator = Orchestrator({
    "researcher": research_agent,
    "analyst":    analysis_agent,
    "writer":     writing_agent,
})


# Run the multi-agent system
async def main():
    result = await orchestrator.run(
        "Research Tokyo's population, calculate its growth over 5 years at 0.5%/year, "
        "and write a brief report about it"
    )
    print(f"\nFINAL ANSWER:\n{result}")


asyncio.run(main())
```

---

## LangChain AgentExecutor

```python
# ============================================================
# LANGCHAIN AGENTEXECUTOR — Production-grade agent framework
# pip install langchain langchain-openai langchain-community
# ============================================================

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# @tool decorator — LangChain's version of @function_tool
# Generates the JSON schema automatically from type hints and docstring
@tool
def search_information(query: str) -> str:
    """
    Search for information about any topic.
    Use this when you need current or factual information.
    """
    # Mock — replace with real search API
    return f"Mock search results for: {query}. Tokyo is the capital of Japan."


@tool
def perform_calculation(expression: str) -> str:
    """
    Perform mathematical calculations.
    Input should be a valid Python mathematical expression.
    Example: '2 + 2', '100 * 0.07', '(1.05) ** 10'
    """
    import ast, operator
    safe_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return safe_ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("Unsafe expression")
    try:
        result = _eval(ast.parse(expression, mode='eval').body)
        return str(round(result, 6))
    except Exception as e:
        return f"Calculation error: {e}"


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# LangChain uses a structured prompt with a MessagesPlaceholder for
# the agent's intermediate steps (tool calls and results)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Use the available tools "
        "to answer questions accurately.",
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),   # Required for agent
])

# create_openai_tools_agent constructs an agent that uses OpenAI's tool-calling API
agent = create_openai_tools_agent(
    llm=llm,
    tools=[search_information, perform_calculation],
    prompt=prompt,
)

# AgentExecutor wraps the agent with the execution loop, error handling,
# verbose logging, and iteration limits
executor = AgentExecutor(
    agent=agent,
    tools=[search_information, perform_calculation],
    verbose=True,          # Print each thought/action to stdout
    max_iterations=5,      # Safety: max tool-call rounds
    handle_parsing_errors=True,   # Recover from JSON parse errors
    return_intermediate_steps=True,   # Include steps in response
)

# Run the agent
result = executor.invoke({
    "input": "What is the capital of Japan and what is 15% of 240?"
})

print(f"\nFinal answer: {result['output']}")
print(f"Intermediate steps: {len(result['intermediate_steps'])}")
```

---

## Agent Safety

```python
# ============================================================
# AGENT SAFETY — PRODUCTION CONTROLS
# ============================================================

import time
from typing import Callable


class SafeAgentWrapper:
    """
    Wraps an agent with production safety controls:
    1. Maximum iteration limit (prevent infinite loops)
    2. Timeout limit (prevent hanging)
    3. Human confirmation for dangerous actions
    4. Budget limit (prevent runaway costs)
    4. Sandboxed tool execution
    """

    def __init__(
        self,
        agent_fn: Callable[[str], str],
        max_iterations: int = 10,
        timeout_seconds: float = 60.0,
        max_cost_usd: float = 1.00,          # $1.00 per run limit
        require_confirmation: bool = False,   # Human-in-the-loop mode
    ):
        self.agent_fn            = agent_fn
        self.max_iterations      = max_iterations
        self.timeout_seconds     = timeout_seconds
        self.max_cost_usd        = max_cost_usd
        self.require_confirmation = require_confirmation

        # Runtime state (reset per run)
        self._iterations   = 0
        self._start_time   = 0.0
        self._estimated_cost = 0.0

    def check_limits(self) -> tuple[bool, str]:
        """
        Check all safety limits. Returns (safe: bool, reason: str).
        Call this before each tool execution.
        """
        # Check iteration limit
        if self._iterations >= self.max_iterations:
            return False, f"Max iterations ({self.max_iterations}) reached"

        # Check timeout
        elapsed = time.time() - self._start_time
        if elapsed > self.timeout_seconds:
            return False, f"Timeout exceeded ({elapsed:.1f}s > {self.timeout_seconds}s)"

        # Check cost limit (approximate — count tokens in production)
        if self._estimated_cost > self.max_cost_usd:
            return False, f"Cost limit exceeded (${self._estimated_cost:.2f})"

        return True, "OK"

    def confirm_action(self, action: str, args: dict) -> bool:
        """
        Request human confirmation before executing an action.

        In production this would send a Slack notification, open
        a UI dialog, or call a webhook — not input() in the terminal.
        """
        if not self.require_confirmation:
            return True

        print(f"\n[CONFIRMATION REQUIRED]")
        print(f"  Action: {action}")
        print(f"  Args:   {args}")
        answer = input("  Allow this action? (yes/no): ").strip().lower()
        return answer in ("yes", "y")

    def run(self, task: str) -> dict:
        """
        Execute the agent with all safety controls active.

        Returns:
            {result: str, iterations: int, elapsed: float, safe: bool}
        """
        self._iterations     = 0
        self._start_time     = time.time()
        self._estimated_cost = 0.0

        safe, reason = self.check_limits()
        if not safe:
            return {"result": f"Blocked before start: {reason}", "safe": False}

        try:
            result = self.agent_fn(task)
            elapsed = time.time() - self._start_time

            return {
                "result":     result,
                "iterations": self._iterations,
                "elapsed":    round(elapsed, 2),
                "cost":       round(self._estimated_cost, 4),
                "safe":       True,
            }
        except Exception as e:
            return {
                "result":  f"Agent raised exception: {e}",
                "safe":    False,
                "elapsed": round(time.time() - self._start_time, 2),
            }
```

---

## Real Use Cases with Architecture

```
USE CASE 1: CUSTOMER SUPPORT AGENT
══════════════════════════════════
User: "I was charged twice for order #12345"

Agent architecture:
  ┌──────────────────────────────────────────┐
  │ Tools: lookup_order, issue_refund,        │
  │        check_payment, send_confirmation   │
  │                                           │
  │ Memory: customer history, past tickets    │
  │ Safety: require_confirmation for refunds  │
  │         max $500 refund without escalation│
  └──────────────────────────────────────────┘

Agent trace:
  1. Search(order_id=12345) → [order details]
  2. Check(payments, order=12345) → [2 payments found]
  3. [CONFIRM] Issue refund $49.99? → [user confirms]
  4. Refund(amount=49.99) → [refund issued]
  5. Send confirmation email → [email sent]

USE CASE 2: CODE REVIEW AGENT
══════════════════════════════
Input: Pull request diff

Agent architecture:
  ┌──────────────────────────────────────────┐
  │ Tools: read_file, run_tests, check_style, │
  │        search_docs, post_comment          │
  │                                           │
  │ Memory: codebase conventions, past PRs    │
  │ Safety: sandboxed code execution,         │
  │         no write access to main branch    │
  └──────────────────────────────────────────┘

USE CASE 3: RESEARCH AGENT
══════════════════════════
Input: "Write a report on the latest AI safety research"

Agent architecture:
  ┌──────────────────────────────────────────┐
  │ Tools: search_web, read_url, save_note,   │
  │        generate_outline, write_section    │
  │                                           │
  │ Memory: notes from visited pages,         │
  │         outline in progress               │
  │ Safety: no posting/publishing without     │
  │         human review                      │
  └──────────────────────────────────────────┘

USE CASE 4: DATA ANALYSIS AGENT
════════════════════════════════
Input: "Analyze sales data and identify trends"

Agent architecture:
  ┌──────────────────────────────────────────┐
  │ Tools: query_db, run_python, make_chart,  │
  │        calculate_statistics               │
  │                                           │
  │ Memory: data schema, previous queries     │
  │ Safety: read-only DB access,              │
  │         sandboxed Python execution        │
  └──────────────────────────────────────────┘
```

---

## Practice Questions

```
PRACTICE QUESTIONS — AI AGENTS
============================================================

CONCEPTUAL:
1.  What is the fundamental difference between a simple LLM API call
    and an AI agent? Give a concrete example of a task that requires
    an agent but cannot be done with a single LLM call.

2.  Describe the 4 core components of an agent (brain, memory, tools,
    planning). What happens if you remove each one? What breaks?

3.  Explain the ReAct pattern. What problem does the "Thought" step
    solve that a pure action-response loop does not? Trace through an
    example step by step.

4.  Compare Plan-and-Execute vs ReAct. When would you choose each?
    What is the biggest failure mode of Plan-and-Execute?

MEMORY:
5.  A user says "remember that I prefer formal language." Which type
    of memory should store this — working, episodic, semantic, or
    procedural? Explain your reasoning.

6.  Your agent's context window fills up after 30 messages. What
    three strategies can you use to handle this? What are the
    tradeoffs of each?

7.  Write a function that takes a conversation history (list of messages)
    and returns a semantic search index (embeddings) so that future
    agents can recall relevant past conversations.

MULTI-AGENT:
8.  You are building a multi-agent system for financial analysis.
    Design the agents (name, role, tools) and the orchestrator.
    Draw the architecture diagram in ASCII.

9.  In the orchestrator-worker pattern, what happens if one worker
    agent fails mid-task? Write the error handling logic for the
    orchestrator.

10. What is the difference between a "swarm" architecture and an
    "orchestrator-worker" architecture? When would you use swarms?

SAFETY:
11. An agent is browsing the web to answer a question and encounters a
    page with malicious instructions. What is this attack called?
    Write the defense in your system prompt.

12. Your ReAct agent keeps calling the same search tool in a loop
    with identical queries, never making progress. What is this called?
    Write three code-level mechanisms to detect and break this loop.

13. Design a "human-in-the-loop" confirmation system for an agent
    that can send emails. What actions require confirmation? What
    actions are safe to auto-approve? Write the decision logic.

IMPLEMENTATION:
14. Implement a simple Task Queue for agents: a data structure that
    stores pending subtasks, tracks which are in progress, and which
    are complete. Include methods: add_task, start_task, complete_task,
    get_pending.

15. Build a minimal agent eval framework: given a set of (task, expected_output)
    pairs, run the agent on each task, compare outputs using an LLM judge,
    and report pass/fail rates. Write the complete implementation.
```
