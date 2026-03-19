# 10 — Tools & Function Calling: Teaching AI to Take Actions

## Why LLMs Alone Have Limitations

A large language model is a text-in, text-out system. It predicts the next
token based on training data. This architecture has hard fundamental limits:

```
WHAT A PLAIN LLM CANNOT DO
============================================================
  Real-time data      |  Training cutoff is months/years old
  Execute code        |  It only generates text ABOUT code
  Fetch URLs          |  No network access at inference time
  Write files         |  No filesystem access
  Call external APIs  |  No HTTP client built in
  Run SQL queries     |  No database connection
  Get current time    |  Has no internal clock
  Do precise math     |  Arithmetic errors are common (3.11 > 3.9 bug)
============================================================

Example failure:
  User:  "What is the current price of AAPL stock?"
  LLM:   "As of my training cutoff in [date], AAPL was around $X..."
         ^^^^ This is stale, potentially wrong, and the user wanted NOW

Example failure 2:
  User:  "What is 1,234,567 * 9,876,543?"
  LLM:   "That is 12,193,254,449,481"  (may be wrong - LLMs hallucinate math)
  Calc:  12,193,263,481 (the real answer)
============================================================
```

Function calling (also called "tool use") solves this by giving the model a
structured way to request that the host application run code and return results.

---

## How Function Calling Works: The Full Flow

```
FUNCTION CALLING: STEP-BY-STEP ARCHITECTURE
============================================================

  USER              YOUR APPLICATION              LLM API
   |                      |                          |
   |  "Weather in Tokyo?" |                          |
   |--------------------->|                          |
   |                      |                          |
   |                      |   STEP 1: SEND REQUEST   |
   |                      |   ┌─────────────────┐   |
   |                      |   │ user message    │   |
   |                      |   │ tool definitions│   |
   |                      |   └─────────────────┘   |
   |                      |------------------------->|
   |                      |                          |
   |                      |   STEP 2: MODEL REASONS  |
   |                      |   "I need real-time      |
   |                      |    weather. I'll call    |
   |                      |    get_weather(Tokyo)"   |
   |                      |                          |
   |                      |   STEP 3: RETURN TOOL    |
   |                      |   CALL (not final answer)|
   |                      |   ┌─────────────────┐   |
   |                      |   │ tool_calls: [   │   |
   |                      |   │  {name: "get_   │   |
   |                      |   │   weather",     │   |
   |                      |   │   args: {city:  │   |
   |                      |   │   "Tokyo"}}]    │   |
   |                      |   └─────────────────┘   |
   |                      |<-------------------------|
   |                      |                          |
   |                      |   STEP 4: YOU EXECUTE    |
   |                      |   get_weather("Tokyo")   |
   |                      |   => {temp: 22, sky:     |
   |                      |       "Sunny"}           |
   |                      |                          |
   |                      |   STEP 5: SEND RESULTS   |
   |                      |   ┌─────────────────┐   |
   |                      |   │ role: "tool"    │   |
   |                      |   │ content: result │   |
   |                      |   └─────────────────┘   |
   |                      |------------------------->|
   |                      |                          |
   |                      |   STEP 6: FINAL ANSWER   |
   |                      |<-------------------------|
   |                      |                          |
   | "It is 22°C and      |                          |
   |  sunny in Tokyo!"    |                          |
   |<---------------------|                          |

CRITICAL INSIGHT:
  The LLM never directly calls anything.
  It produces structured JSON describing WHAT to call and WITH WHAT ARGS.
  YOUR application does all the actual execution.
  This means you have complete control over what actually runs.
```

---

## JSON Schema for Tool Definitions

When you define a tool, you describe its interface using JSON Schema.
The model reads this to understand when to use the tool and what arguments
to provide.

```
TOOL DEFINITION ANATOMY
============================================================

{
  "type": "function",         <- always "function" for OpenAI tools
  "function": {
    "name": "search_web",     <- what model uses to invoke this tool
    "description": "...",     <- MOST IMPORTANT: model reads this to
                                 decide WHEN to use this tool.
                                 Write clear, specific descriptions.
    "parameters": {           <- JSON Schema describing valid arguments
      "type": "object",       <- always "object" at the top level
      "properties": {
        ...property definitions...
      },
      "required": [...]       <- list of mandatory property names
    }
  }
}

ALL JSON SCHEMA PROPERTY TYPES
============================================================

  STRING — plain text value:
    "city": {
      "type": "string",
      "description": "The city name, e.g. 'Tokyo' or 'New York'"
    }

  NUMBER — any numeric value including decimals:
    "temperature": {
      "type": "number",
      "description": "Temperature value"
    }

  INTEGER — whole numbers only:
    "limit": {
      "type": "integer",
      "description": "Max results to return",
      "default": 10
    }

  BOOLEAN — true or false:
    "include_forecast": {
      "type": "boolean",
      "description": "Whether to include 7-day forecast"
    }

  ENUM — constrained string (only specific values allowed):
    "unit": {
      "type": "string",
      "enum": ["celsius", "fahrenheit", "kelvin"],
      "description": "Temperature unit to use in response"
    }

  ARRAY — list of values:
    "cities": {
      "type": "array",
      "items": { "type": "string" },
      "description": "List of city names to check"
    }

  OBJECT — nested structure:
    "location": {
      "type": "object",
      "properties": {
        "lat": {
          "type": "number",
          "description": "Latitude (-90 to 90)"
        },
        "lon": {
          "type": "number",
          "description": "Longitude (-180 to 180)"
        }
      },
      "required": ["lat", "lon"]
    }
```

---

## OpenAI Function Calling: Complete 6-Step Tutorial

```python
# ============================================================
# COMPLETE OPENAI FUNCTION CALLING TUTORIAL
# Covers all 6 steps: define -> call -> detect -> execute
#                     -> send result -> final response
# ============================================================

import json
import os
from openai import OpenAI

# Initialize the OpenAI client using an environment variable
# NEVER hardcode API keys — use environment variables or a secrets manager
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# STEP 1: DEFINE YOUR TOOLS
#
# Each tool has two parts:
#   a) The Python function that actually does the work
#   b) The JSON Schema that describes the function to the model
# ============================================================

def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """
    Retrieve current weather for a given city.

    In a real application this would call an external weather API
    (e.g. OpenWeatherMap). Here we use mock data for illustration.

    Args:
        location: City name such as "Tokyo" or "London"
        unit: "celsius" (default) or "fahrenheit"

    Returns:
        Dictionary with temperature, condition, and humidity
    """
    # Mock dataset — replace with real API call in production
    mock_data = {
        "Tokyo":    {"temp_c": 22, "condition": "Sunny",        "humidity": 65},
        "London":   {"temp_c": 14, "condition": "Cloudy",       "humidity": 80},
        "New York": {"temp_c": 18, "condition": "Partly cloudy","humidity": 70},
        "Sydney":   {"temp_c": 28, "condition": "Hot",          "humidity": 55},
    }

    # Look up the city; fall back to a default if unknown
    weather = mock_data.get(
        location,
        {"temp_c": 20, "condition": "Unknown", "humidity": 60}
    )

    temp = weather["temp_c"]

    # Convert to Fahrenheit if requested
    if unit == "fahrenheit":
        temp = (temp * 9 / 5) + 32   # Standard formula: F = (C * 9/5) + 32
        unit_symbol = "°F"
    else:
        unit_symbol = "°C"

    return {
        "location":    location,
        "temperature": f"{temp}{unit_symbol}",
        "condition":   weather["condition"],
        "humidity":    f"{weather['humidity']}%",
    }


# JSON Schema description — this is what you send to the API
# Think of it as documentation that the LLM reads at runtime
weather_tool_definition = {
    "type": "function",
    "function": {
        "name": "get_current_weather",    # Must match your Python function by convention
        "description": (
            "Get the current weather conditions for a specific city. "
            "Use this whenever the user asks about the weather, temperature, "
            "humidity, or climate in any location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "The city name to check weather for. "
                        "Examples: 'Tokyo', 'New York', 'London', 'Sydney'"
                    ),
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],   # Only these two values
                    "description": (
                        "Temperature unit for the response. "
                        "Use 'celsius' (default) or 'fahrenheit'."
                    ),
                },
            },
            "required": ["location"],   # Only location is mandatory; unit defaults to celsius
        },
    },
}


# ============================================================
# STEP 2: CALL THE API WITH TOOLS
#
# Pass your tool definitions alongside the user message.
# The model will either respond directly OR request a tool call.
# ============================================================

def run_weather_agent(user_message: str) -> str:
    """
    Execute the full 6-step function calling cycle for one user query.

    Returns:
        The final natural-language response from the model
    """
    print(f"\n[USER]: {user_message}")
    print("=" * 60)

    # Build the messages list — system prompt + user message
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful weather assistant. "
                "Use the get_current_weather tool whenever the user asks "
                "about weather in any city."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    # STEP 2: First API call — send tools alongside messages
    print("[STEP 2] Sending request with tool definitions...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[weather_tool_definition],  # All available tools
        tool_choice="auto",               # Let model decide when to use tools
    )

    response_message = response.choices[0].message


    # ============================================================
    # STEP 3: DETECT WHETHER THE MODEL WANTS A TOOL CALL
    #
    # finish_reason == "tool_calls"  -> model wants to call a tool
    # finish_reason == "stop"        -> model gave a final answer directly
    # ============================================================

    print(f"[STEP 3] finish_reason = {response.choices[0].finish_reason}")

    if response.choices[0].finish_reason != "tool_calls":
        # Model answered directly — no tool needed
        print("[INFO] Model replied without using any tool.")
        return response_message.content

    # Model wants to call tools — get the requested calls
    tool_calls = response_message.tool_calls
    print(f"[STEP 3] Model requested {len(tool_calls)} tool call(s):")
    for tc in tool_calls:
        print(f"  Tool: {tc.function.name}  Args: {tc.function.arguments}")


    # ============================================================
    # STEP 4: EXECUTE THE TOOLS
    #
    # Parse the JSON arguments the model generated, then call
    # the corresponding Python functions. This runs on YOUR server.
    # ============================================================

    print("\n[STEP 4] Executing tool calls...")

    # Append the assistant message (containing tool_calls) to history
    # so the model can see what it asked for in the next round
    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name

        # The model outputs arguments as a JSON string — parse it
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Could not parse tool arguments: {e}")
            function_result = {"error": "Invalid JSON in tool arguments"}
        else:
            # Dispatch to the right Python function
            # In production use a dict/registry instead of if/elif chains
            if function_name == "get_current_weather":
                function_result = get_current_weather(**function_args)
            else:
                function_result = {"error": f"Unknown function: {function_name}"}

        print(f"  Result: {function_result}")


        # ============================================================
        # STEP 5: SEND TOOL RESULTS BACK TO THE MODEL
        #
        # Add a message with role="tool" containing the result.
        # The tool_call_id MUST match the id from the tool_call object
        # so the model knows which result belongs to which request.
        # ============================================================

        messages.append(
            {
                "role": "tool",                          # Special role for tool results
                "tool_call_id": tool_call.id,            # Links result to specific request
                "content": json.dumps(function_result),  # Result must be a string
            }
        )


    # ============================================================
    # STEP 6: GET THE FINAL RESPONSE
    #
    # Send the full conversation history (messages now include the
    # tool results) back to the model to produce a human-readable answer.
    # ============================================================

    print("\n[STEP 6] Getting final response from model...")

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,   # Includes original query + tool results
    )

    final_answer = final_response.choices[0].message.content
    print(f"\n[ASSISTANT]: {final_answer}")
    return final_answer


# Test the complete flow
if __name__ == "__main__":
    # This should trigger the weather tool
    run_weather_agent("What's the weather like in Tokyo right now?")

    # This should answer directly (no tool needed)
    run_weather_agent("What is the capital of Japan?")

    # Test Fahrenheit conversion
    run_weather_agent("How hot is it in Sydney in Fahrenheit?")
```

---

## Parallel Tool Calls

When multiple independent tool calls are needed, OpenAI can request them
all in a single response. Your code must process every call and return all
results before asking for the final answer.

```python
# ============================================================
# PARALLEL TOOL CALLS
# The model requests multiple calls in one response when they
# are independent. More efficient than sequential round-trips.
# ============================================================

import json
from openai import OpenAI

client = OpenAI()


# Mock implementations — in production these call real APIs
def get_weather(city: str) -> dict:
    """Return weather for a city."""
    data = {
        "NYC":     {"temp": "18°C", "condition": "Cloudy"},
        "LA":      {"temp": "26°C", "condition": "Sunny"},
        "Chicago": {"temp": "12°C", "condition": "Windy"},
    }
    return data.get(city, {"temp": "?", "condition": "Unknown"})


def get_population(city: str) -> dict:
    """Return population statistics for a city."""
    data = {
        "NYC":     {"population": "8.3 million", "metro": "20 million"},
        "LA":      {"population": "3.9 million", "metro": "13 million"},
        "Chicago": {"population": "2.7 million", "metro": "9.5 million"},
    }
    return data.get(city, {"population": "unknown"})


# Define both tools so the model knows about both capabilities
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a US city abbreviation",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City abbreviation: NYC, LA, or Chicago"
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get population statistics for a US city abbreviation",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City abbreviation: NYC, LA, or Chicago"
                    }
                },
                "required": ["city"],
            },
        },
    },
]


def run_parallel_tool_example(user_message: str) -> str:
    """
    Demonstrates how to handle multiple simultaneous tool calls.

    When asked "Compare NYC and LA", the model might request:
      - get_weather("NYC")      simultaneously
      - get_weather("LA")       simultaneously
      - get_population("NYC")   simultaneously
      - get_population("LA")    simultaneously

    You process ALL of them before the next API call.
    """
    messages = [{"role": "user", "content": user_message}]

    # First call — model may request several tool calls at once
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    if response.choices[0].finish_reason != "tool_calls":
        return response_message.content   # No tools needed

    # Add the assistant message (with all requested tool_calls)
    messages.append(response_message)

    print(f"Model requested {len(response_message.tool_calls)} parallel calls:")

    # Process every tool call — in production these could run concurrently
    # using threading.ThreadPoolExecutor or asyncio.gather
    for tool_call in response_message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"  Executing: {name}({args})")

        # Dispatch to the correct function
        if name == "get_weather":
            result = get_weather(**args)
        elif name == "get_population":
            result = get_population(**args)
        else:
            result = {"error": f"Unknown tool: {name}"}

        # Each result needs its own tool message with matching tool_call_id
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,   # MUST match the call's id
                "content": json.dumps(result),
            }
        )

    # Final call — model synthesizes all results into one answer
    final = client.chat.completions.create(model="gpt-4o", messages=messages)
    return final.choices[0].message.content


answer = run_parallel_tool_example(
    "Compare the weather and population of NYC, LA, and Chicago"
)
print(f"\nFinal answer:\n{answer}")
```

---

## tool_choice Options

```python
# ============================================================
# tool_choice PARAMETER — ALL OPTIONS EXPLAINED
# Controls whether and how the model uses your tools
# ============================================================

# OPTION 1: "auto"  (default, recommended for most cases)
# The model decides whether to call a tool or reply directly.
# Use when you want the model to be smart about when tools are needed.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",   # Model picks: use tool OR reply directly
)

# OPTION 2: "none"
# Forces the model to reply directly, ignoring all tools.
# Use when you want a conversational response even if tools are defined
# (e.g., "explain what a weather API does" should not call the API).
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="none",   # NEVER use a tool — always reply in text
)

# OPTION 3: "required"
# Forces the model to call at least one tool before responding.
# Use for structured-output pipelines where you always want JSON data back.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="required",   # MUST call a tool — no direct text reply
)

# OPTION 4: Specific function
# Forces the model to call exactly the named function.
# Use in deterministic pipelines where the call pattern is fixed.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice={
        "type": "function",
        "function": {"name": "get_weather"},   # Must call this specific tool
    },
)
```

---

## Anthropic Tool Use: Different Format

```python
# ============================================================
# ANTHROPIC (CLAUDE) TOOL USE
# Conceptually identical to OpenAI but with a different API shape
# ============================================================

import anthropic
import json

client = anthropic.Anthropic()   # Reads ANTHROPIC_API_KEY from environment

# Anthropic tool definition format — note the differences from OpenAI:
#   - No "type": "function" wrapper
#   - "input_schema" instead of "parameters"
tools_anthropic = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Returns temperature, conditions, and humidity."
        ),
        "input_schema": {             # Called "input_schema", not "parameters"
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo' or 'London'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                },
            },
            "required": ["location"],
        },
    }
]


def run_anthropic_tool_use(user_message: str) -> str:
    """
    Complete tool use cycle with Claude.

    Key differences from OpenAI:
      - stop_reason is "tool_use" (not "tool_calls")
      - Tool results go in a "user" message (not a "tool" message)
      - input is already a dict (not a JSON string)
    """
    messages = [{"role": "user", "content": user_message}]

    # First API call — include tools
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=tools_anthropic,
        messages=messages,
    )

    # Anthropic uses stop_reason "tool_use" (OpenAI uses "tool_calls")
    if response.stop_reason != "tool_use":
        return response.content[0].text   # Direct text reply — no tool needed

    # Find all tool_use blocks in the response content
    # Claude can mix text blocks and tool_use blocks in one response
    tool_use_blocks = [
        block for block in response.content
        if block.type == "tool_use"
    ]

    tool_results = []

    for tool_use in tool_use_blocks:
        print(f"Claude calling: {tool_use.name}")
        print(f"With input: {tool_use.input}")   # Already a dict — no json.loads needed

        # Execute the tool
        if tool_use.name == "get_weather":
            result = get_current_weather(
                location=tool_use.input.get("location", ""),
                unit=tool_use.input.get("unit", "celsius"),
            )
        else:
            result = {"error": f"Unknown tool: {tool_use.name}"}

        # Anthropic tool results use "type": "tool_result" inside a user message
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,          # Must match the tool_use block id
                "content": json.dumps(result),
            }
        )

    # Add assistant response (with tool_use blocks) to history
    messages.append({"role": "assistant", "content": response.content})

    # Tool results are sent as a USER message containing tool_result objects
    # (OpenAI uses a separate "tool" role; Anthropic packs results into "user")
    messages.append({"role": "user", "content": tool_results})

    # Final API call — Claude now has the results and produces the answer
    final_response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=tools_anthropic,
        messages=messages,
    )

    return final_response.content[0].text


result = run_anthropic_tool_use("What's the weather in Paris?")
print(result)
```

---

## 5 Practical Tool Implementations

### Tool 1: Web Search

```python
# ============================================================
# TOOL 1: WEB SEARCH
# Gives the LLM access to current web content
# In production use: Serper API, Brave Search, or Bing API
# ============================================================

import requests


def search_web(query: str, num_results: int = 5) -> dict:
    """
    Search the web and return snippets from top results.

    Security considerations:
    - Validate and sanitize the query before sending it
    - Set request timeouts to prevent hanging
    - Use an API key stored in an environment variable

    Args:
        query: Search terms — must be 1 to 500 characters
        num_results: How many results to return (clamped to 1-10)

    Returns:
        Dictionary with a list of result objects (title, snippet, url)
    """
    # INPUT VALIDATION — reject bad inputs before hitting any API
    if not query or not isinstance(query, str):
        return {"error": "Query must be a non-empty string"}
    if len(query) > 500:
        return {"error": "Query too long (max 500 characters)"}

    # Clamp to valid range — never trust the model's parameter values
    num_results = max(1, min(num_results, 10))

    # In production replace this with a real search API:
    #   Serper:  POST https://google.serper.dev/search
    #   Brave:   GET  https://api.search.brave.com/res/v1/web/search
    #   Bing:    GET  https://api.bing.microsoft.com/v7.0/search

    # Here we use DuckDuckGo's unofficial JSON endpoint for demo only
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q":              query,
            "format":         "json",
            "no_html":        "1",   # Strip HTML tags from snippets
            "skip_disambig":  "1",   # Skip disambiguation pages
        }

        # Always set a timeout — never let network calls block indefinitely
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()   # Raise for HTTP 4xx/5xx

        data = response.json()
        results = []

        # Main abstract (shown when DuckDuckGo has a direct answer)
        if data.get("Abstract"):
            results.append(
                {
                    "title":   data.get("Heading", "Summary"),
                    "snippet": data["Abstract"],
                    "url":     data.get("AbstractURL", ""),
                }
            )

        # Related topic links
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and "Text" in topic:
                results.append(
                    {
                        "title":   topic["Text"][:80],   # Use first 80 chars as title
                        "snippet": topic["Text"],
                        "url":     topic.get("FirstURL", ""),
                    }
                )
            if len(results) >= num_results:
                break

        return {
            "query":       query,
            "num_results": len(results),
            "results":     results[:num_results],
        }

    except requests.Timeout:
        return {"error": "Search request timed out", "query": query}
    except requests.RequestException as e:
        return {"error": f"Search request failed: {str(e)}", "query": query}


search_tool_definition = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the internet for current information. "
            "Use this for recent news, real-time data, or anything that may "
            "have changed since the training cutoff. "
            "Do NOT use for stable well-known facts like capitals or formulas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Concise search query. "
                        "Example: 'Python 3.13 new features'"
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (1-10, default 5)",
                },
            },
            "required": ["query"],
        },
    },
}
```

### Tool 2: Safe Calculator

```python
# ============================================================
# TOOL 2: SAFE CALCULATOR
# LLMs make arithmetic errors — use a real calculator
# ============================================================

import ast
import operator
import math


def safe_calculate(expression: str) -> dict:
    """
    Evaluate a mathematical expression safely using AST parsing.

    WHY NOT eval()?
    eval("__import__('os').system('rm -rf /')") would destroy your server.
    Instead we parse the AST and allow ONLY math operations.

    Supported: +  -  *  /  **  %  //  sqrt()  log()  sin()  cos()  abs()  round()

    Args:
        expression: Math expression string, e.g. "(100 * 1.05) ** 10"

    Returns:
        Dictionary with "result" key or "error" key
    """
    # Map AST operator types to Python operator functions
    allowed_binary_ops = {
        ast.Add:      operator.add,        # +
        ast.Sub:      operator.sub,        # -
        ast.Mult:     operator.mul,        # *
        ast.Div:      operator.truediv,    # /
        ast.Pow:      operator.pow,        # **
        ast.Mod:      operator.mod,        # %
        ast.FloorDiv: operator.floordiv,   # //
    }
    allowed_unary_ops = {
        ast.USub: operator.neg,   # unary minus (-x)
        ast.UAdd: operator.pos,   # unary plus  (+x)
    }

    # Explicitly whitelisted functions and constants
    safe_names = {
        "abs":   abs,
        "round": round,
        "sqrt":  math.sqrt,
        "floor": math.floor,
        "ceil":  math.ceil,
        "log":   math.log,
        "log10": math.log10,
        "sin":   math.sin,
        "cos":   math.cos,
        "tan":   math.tan,
        "pi":    math.pi,
        "e":     math.e,
    }

    def eval_node(node):
        """Recursively evaluate a single AST node."""

        if isinstance(node, ast.Constant):
            # Numeric literal: 42, 3.14
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Non-numeric constant: {node.value!r}")

        elif isinstance(node, ast.Name):
            # Named constant or function: pi, sqrt, abs, ...
            if node.id in safe_names:
                return safe_names[node.id]
            raise ValueError(f"Undefined name: {node.id!r}")

        elif isinstance(node, ast.BinOp):
            # Binary operation: a + b
            op_type = type(node.op)
            if op_type not in allowed_binary_ops:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left  = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_binary_ops[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            # Unary operation: -x
            op_type = type(node.op)
            if op_type not in allowed_unary_ops:
                raise ValueError(f"Unsupported unary op: {op_type.__name__}")
            return allowed_unary_ops[op_type](eval_node(node.operand))

        elif isinstance(node, ast.Call):
            # Function call: sqrt(16), abs(-5)
            func = eval_node(node.func)
            if not callable(func):
                raise ValueError("Attempted to call a non-function")
            args = [eval_node(arg) for arg in node.args]
            return func(*args)

        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    # --- Validation ---
    expression = expression.strip()
    if len(expression) > 200:
        return {"error": "Expression too long (max 200 characters)"}

    try:
        # ast.parse in 'eval' mode returns an Expression node
        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree.body)

        # Round away floating-point noise (e.g. 0.30000000000000004 -> 0.3)
        if isinstance(result, float):
            result = round(result, 10)

        return {
            "expression": expression,
            "result":      result,
            "type":        type(result).__name__,
        }

    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except (ValueError, TypeError) as e:
        return {"error": str(e)}
    except SyntaxError:
        return {"error": "Invalid expression syntax"}


calculator_tool_definition = {
    "type": "function",
    "function": {
        "name": "safe_calculate",
        "description": (
            "Perform precise mathematical calculations. "
            "Use this for arithmetic, algebra, or any math — "
            "do not try to compute in your head. "
            "Supports +, -, *, /, **, %, //, sqrt, log, sin, cos, abs, round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate. "
                        "Examples: '2 + 2', '(100 * 1.05) ** 10', 'sqrt(144)'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}

# Quick smoke tests
print(safe_calculate("2 + 2"))                       # result: 4
print(safe_calculate("sqrt(144)"))                   # result: 12.0
print(safe_calculate("(100 * 1.05) ** 10"))          # compound interest
print(safe_calculate("__import__('os').getcwd()"))   # error — blocked
```

### Tool 3: Database Query

```python
# ============================================================
# TOOL 3: DATABASE QUERY — READ-ONLY WITH SAFETY CONTROLS
# ============================================================

import sqlite3
import re


# Only SELECT statements are permitted
READ_ONLY_REGEX = re.compile(r"^\s*SELECT\s", re.IGNORECASE)

# Keywords that should never appear even inside a SELECT
FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "INSERT", "UPDATE", "CREATE",
    "ALTER", "TRUNCATE", "EXEC", "EXECUTE", "--", "/*",
}


def query_database(sql: str, limit: int = 20) -> dict:
    """
    Execute a read-only SQL SELECT query.

    Security layers:
    1. Regex: Must start with SELECT
    2. Keyword scan: Blocks destructive/injection keywords
    3. Automatic LIMIT injection: Prevents full-table scans
    4. Parameterized result set: Data returned as plain dicts

    Args:
        sql: A SQL SELECT statement
        limit: Max rows to return (1-100, default 20)

    Returns:
        {"columns": [...], "rows": [...], "row_count": N}
    """
    # SECURITY CHECK 1: Must be a SELECT statement
    if not READ_ONLY_REGEX.match(sql):
        return {"error": "Only SELECT queries are allowed"}

    # SECURITY CHECK 2: Block forbidden keywords
    sql_upper = sql.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in sql_upper:
            return {"error": f"Forbidden keyword in query: {keyword}"}

    # Clamp limit to a safe range
    limit = max(1, min(limit, 100))

    # Inject LIMIT if the query does not already have one
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";") + f" LIMIT {limit}"

    try:
        # Connect to the SQLite database
        # In production use a connection pool and a PostgreSQL/MySQL driver
        conn = sqlite3.connect("products.db")
        conn.row_factory = sqlite3.Row   # Rows behave like dicts
        cursor = conn.cursor()

        cursor.execute(sql)

        # Extract column names from the cursor description
        columns = [desc[0] for desc in cursor.description]

        # Convert Row objects to plain dicts for JSON serialization
        rows = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {"columns": columns, "rows": rows, "row_count": len(rows), "sql": sql}

    except sqlite3.Error as e:
        return {"error": f"Database error: {str(e)}"}


db_tool_definition = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": (
            "Query the product catalog database. "
            "Available tables: products, orders, categories, inventory. "
            "Read-only — SELECT queries only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A valid SQL SELECT statement. "
                        "Example: \"SELECT name, price FROM products WHERE price < 50\""
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (1-100, default 20)",
                },
            },
            "required": ["sql"],
        },
    },
}
```

### Tool 4: Weather (Production-Ready)

```python
# ============================================================
# TOOL 4: PRODUCTION WEATHER — REAL API WITH CACHING
# ============================================================

import os
import time
import requests


# TTL cache — keyed by "city:unit", stores (timestamp, result)
_weather_cache: dict = {}
CACHE_TTL_SECONDS = 600   # 10 minutes — weather data does not change faster


def get_weather_production(
    location: str,
    unit: str = "metric",
    include_forecast: bool = False,
) -> dict:
    """
    Fetch real weather data from the OpenWeatherMap API.

    Requires OPENWEATHERMAP_API_KEY environment variable.
    Free tier at: https://openweathermap.org/api

    Args:
        location: City name or "city,country_code" e.g. "Tokyo,JP"
        unit: "metric" (Celsius) or "imperial" (Fahrenheit)
        include_forecast: Include 5-day daily forecast if True

    Returns:
        Weather dict, or {"error": "..."} on failure
    """
    # Never hardcode credentials — read from environment
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return {"error": "OPENWEATHERMAP_API_KEY not set"}

    # Input validation
    location = location.strip()
    if not location or len(location) > 100:
        return {"error": "Invalid location string"}
    if unit not in ("metric", "imperial"):
        return {"error": "unit must be 'metric' or 'imperial'"}

    # Check TTL cache to avoid redundant API calls
    cache_key = f"{location.lower()}:{unit}"
    if cache_key in _weather_cache:
        cached_ts, cached_data = _weather_cache[cache_key]
        if time.time() - cached_ts < CACHE_TTL_SECONDS:
            return {**cached_data, "cached": True}   # Mark as served from cache

    base = "https://api.openweathermap.org/data/2.5"

    try:
        resp = requests.get(
            f"{base}/weather",
            params={"q": location, "appid": api_key, "units": unit},
            timeout=5,   # Always timeout external calls
        )

        # Handle specific HTTP errors explicitly for clearer messages
        if resp.status_code == 404:
            return {"error": f"City not found: {location}"}
        if resp.status_code == 401:
            return {"error": "Invalid API key"}
        resp.raise_for_status()

        w = resp.json()
        unit_symbol = "°C" if unit == "metric" else "°F"
        speed_unit  = "m/s" if unit == "metric" else "mph"

        result = {
            "location":    f"{w['name']}, {w['sys']['country']}",
            "temperature": f"{w['main']['temp']:.1f}{unit_symbol}",
            "feels_like":  f"{w['main']['feels_like']:.1f}{unit_symbol}",
            "condition":   w["weather"][0]["description"],
            "humidity":    f"{w['main']['humidity']}%",
            "wind":        f"{w['wind']['speed']} {speed_unit}",
            "cached":      False,
        }

        if include_forecast:
            fr = requests.get(
                f"{base}/forecast",
                params={"q": location, "appid": api_key, "units": unit},
                timeout=5,
            )
            fr.raise_for_status()
            fd = fr.json()

            # The API returns 3-hour intervals; keep one entry per calendar date
            forecasts, seen = [], set()
            for item in fd["list"]:
                date = item["dt_txt"].split()[0]
                if date not in seen:
                    seen.add(date)
                    forecasts.append(
                        {
                            "date":      date,
                            "temp":      f"{item['main']['temp']:.1f}{unit_symbol}",
                            "condition": item["weather"][0]["description"],
                        }
                    )
                if len(forecasts) >= 5:
                    break

            result["forecast"] = forecasts

        # Store in TTL cache
        _weather_cache[cache_key] = (time.time(), result.copy())
        return result

    except requests.Timeout:
        return {"error": "Weather API timed out — try again"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
```

### Tool 5: Code Executor (with Safety Warning)

```python
# ============================================================
# TOOL 5: CODE EXECUTOR
#
# WARNING: Executing LLM-generated code is inherently dangerous.
# This implementation uses subprocess isolation which is minimal.
# For production consider:
#   - E2B Sandboxes  (https://e2b.dev)   — purpose-built for AI
#   - Docker containers with --no-new-privileges
#   - Firecracker microVMs
#   - AWS Lambda with strict IAM
# ============================================================

import subprocess
import tempfile
import os


EXECUTION_TIMEOUT = 10      # Kill the process after N seconds
MAX_OUTPUT_BYTES  = 4096    # Truncate stdout/stderr to this length

# Module names that can cause harm even inside a subprocess
# This is a defense-in-depth measure, NOT a complete sandbox
BLOCKED_IMPORT_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import requests",
    "import urllib",
    "import shutil",
    "__import__",
    "open(",
]


def execute_python_safely(code: str, timeout: int = 10) -> dict:
    """
    Execute Python code in a subprocess and return stdout/stderr.

    This is a MINIMAL safety implementation. Use a proper sandbox
    in production. See the warning above.

    Args:
        code: Python code to execute. Use print() to produce output.
        timeout: Max seconds before the process is killed (1-10)

    Returns:
        {"stdout": "...", "stderr": "...", "return_code": 0, "status": "success"}
    """
    # Basic content checks — NOT a complete defense, just defense-in-depth
    for pattern in BLOCKED_IMPORT_PATTERNS:
        if pattern in code:
            return {
                "error":  f"Disallowed pattern: {pattern!r}",
                "status": "blocked",
            }

    timeout = max(1, min(timeout, EXECUTION_TIMEOUT))

    tmp_path = None
    try:
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="codeexec_"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,   # Captures stdout and stderr separately
            text=True,             # Returns str instead of bytes
            timeout=timeout,       # Sends SIGKILL after timeout
        )

        # Truncate to prevent memory exhaustion from runaway output
        stdout = result.stdout[:MAX_OUTPUT_BYTES]
        stderr = result.stderr[:MAX_OUTPUT_BYTES]

        return {
            "stdout":      stdout,
            "stderr":      stderr,
            "return_code": result.returncode,
            "status":      "success" if result.returncode == 0 else "runtime_error",
            "truncated":   len(result.stdout) > MAX_OUTPUT_BYTES,
        }

    except subprocess.TimeoutExpired:
        return {"error": f"Timed out after {timeout}s", "status": "timeout"}
    except Exception as e:
        return {"error": str(e), "status": "exception"}
    finally:
        # Always remove the temporary file — even if an exception occurred
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


code_executor_tool_definition = {
    "type": "function",
    "function": {
        "name": "execute_python_safely",
        "description": (
            "Execute Python code and return its output. "
            "Useful for calculations, data transformations, or algorithm demos. "
            "Network and file system access are not available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. Use print() to produce output.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max execution time in seconds (1-10)",
                },
            },
            "required": ["code"],
        },
    },
}
```

---

## Tool Safety: Validation, Sanitization, Confirmation

```python
# ============================================================
# TOOL SAFETY FRAMEWORK
# Wraps any tool with validation, risk-based confirmation,
# rate limiting, and audit logging
# ============================================================

from enum import Enum
from typing import Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Four-tier risk classification for tool operations."""
    LOW      = "low"       # Read-only, zero side effects
    MEDIUM   = "medium"    # Modifies reversible state
    HIGH     = "high"      # Irreversible (send email, post to API)
    CRITICAL = "critical"  # Destructive (delete data, send money)


class ToolSafetyWrapper:
    """
    Decorator-based tool registry with safety controls.

    Features:
    - Risk-level annotation per tool
    - Optional per-tool validator function
    - Human-confirmation gate for HIGH/CRITICAL tools
    - Call-count rate limiting
    - Structured audit log
    """

    def __init__(
        self,
        require_confirmation_at: RiskLevel = RiskLevel.HIGH,
    ):
        self._tools: dict[str, dict] = {}
        self._threshold = require_confirmation_at
        self.audit_log: list[dict] = []

    def register(
        self,
        risk: RiskLevel,
        validate: Optional[Callable[[dict], Optional[str]]] = None,
    ):
        """
        Decorator to register a tool.

        Args:
            risk: Risk level classification for this tool
            validate: Optional function(args) -> error_string_or_None
        """
        def decorator(func: Callable) -> Callable:
            self._tools[func.__name__] = {
                "func":       func,
                "risk":       risk,
                "validate":   validate,
                "call_count": 0,
            }
            return func
        return decorator

    def _risk_index(self, level: RiskLevel) -> int:
        """Return numeric order for risk comparison."""
        return [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL].index(level)

    def execute(
        self,
        tool_name: str,
        args: dict,
        confirmed: bool = False,
    ) -> dict:
        """
        Execute a registered tool after safety checks.

        Args:
            tool_name: Registered function name
            args: Keyword arguments to pass to the function
            confirmed: True when the user has explicitly approved execution

        Returns:
            Tool result, or an error/confirmation-request dict
        """
        if tool_name not in self._tools:
            return {"error": f"Tool not registered: {tool_name}"}

        info = self._tools[tool_name]
        risk = info["risk"]

        # --- Validation ---
        if info["validate"]:
            error = info["validate"](args)
            if error:
                logger.warning("Validation blocked %s: %s", tool_name, error)
                return {"error": f"Validation error: {error}"}

        # --- Confirmation gate ---
        if (
            self._risk_index(risk) >= self._risk_index(self._threshold)
            and not confirmed
        ):
            return {
                "requires_confirmation": True,
                "tool":       tool_name,
                "args":       args,
                "risk_level": risk.value,
                "message":    (
                    f"This action has '{risk.value}' risk. "
                    "Please confirm before I proceed."
                ),
            }

        # --- Rate limiting (basic) ---
        info["call_count"] += 1
        if info["call_count"] > 500:
            return {"error": "Rate limit exceeded for this tool"}

        # --- Execute ---
        try:
            result = info["func"](**args)
            self.audit_log.append(
                {"tool": tool_name, "args": args, "status": "success", "risk": risk.value}
            )
            return result
        except Exception as exc:
            logger.exception("Tool %s raised: %s", tool_name, exc)
            self.audit_log.append(
                {"tool": tool_name, "args": args, "status": "error", "error": str(exc)}
            )
            return {"error": f"Execution error: {exc}"}


# ---- Demo usage ----

safety = ToolSafetyWrapper(require_confirmation_at=RiskLevel.HIGH)


@safety.register(risk=RiskLevel.LOW)
def read_file(path: str) -> dict:
    """Read a file — safe, no side effects."""
    return {"content": f"(contents of {path})"}


@safety.register(
    risk=RiskLevel.HIGH,
    validate=lambda args: (
        "suspicious" in args.get("to", "").lower() and "Suspicious recipient" or None
    ),
)
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email — high risk, requires confirmation."""
    return {"status": f"Email sent to {to}"}


# Low-risk: executes immediately
print(safety.execute("read_file", {"path": "/tmp/notes.txt"}))

# High-risk: returns confirmation request
print(safety.execute("send_email", {"to": "alice@example.com", "subject": "Hi", "body": "Test"}))

# High-risk with explicit confirmation: executes
print(safety.execute(
    "send_email",
    {"to": "alice@example.com", "subject": "Hi", "body": "Test"},
    confirmed=True,
))
```

---

## Building a Complete Agent Loop

```python
# ============================================================
# COMPLETE AGENT LOOP
# Production-ready implementation with tool dispatch and safety
# ============================================================

import json
from openai import OpenAI

client = OpenAI()


class SimpleAgent:
    """
    A minimal but production-quality AI agent.

    The loop:
      1. Send message + tools to the model
      2. If finish_reason == "tool_calls", execute all requested tools
      3. Add tool results to messages
      4. Repeat from step 1
      5. When finish_reason == "stop", return the final answer

    Safety:
      - max_iterations prevents runaway loops
      - Unknown tools return an error dict instead of raising
      - JSON parse errors are caught and returned as errors
    """

    def __init__(
        self,
        tools: list[dict],
        tool_functions: dict[str, Callable],
        system_prompt: str = "You are a helpful assistant.",
        max_iterations: int = 10,
        model: str = "gpt-4o",
    ):
        self.tools = tools
        self.tool_functions = tool_functions
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.model = model

    def run(self, user_message: str) -> str:
        """
        Process a user message through the full agent loop.

        Returns:
            Final text answer from the model
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_message},
        ]

        for iteration in range(1, self.max_iterations + 1):
            print(f"[Iteration {iteration}/{self.max_iterations}]")

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            message      = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            print(f"  finish_reason: {finish_reason}")

            # Model is done — return final answer
            if finish_reason == "stop":
                return message.content

            # Unexpected finish reason — return whatever content exists
            if finish_reason != "tool_calls":
                return message.content or f"[Stopped with reason: {finish_reason}]"

            # Append the assistant message (contains tool_calls metadata)
            messages.append(message)

            # Process every requested tool call
            for tc in message.tool_calls:
                name = tc.function.name
                print(f"  Calling tool: {name}")

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    result = {"error": f"Bad tool arguments (JSON): {e}"}
                else:
                    if name in self.tool_functions:
                        try:
                            result = self.tool_functions[name](**args)
                        except Exception as e:
                            result = {"error": f"Tool raised: {e}"}
                    else:
                        result = {"error": f"Tool not found: {name}"}

                messages.append(
                    {
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      json.dumps(result),
                    }
                )

        # Safety: exhausted max_iterations
        return (
            "I reached my step limit before finding a final answer. "
            "Please try rephrasing your question or breaking it into smaller parts."
        )


# Instantiate and run
agent = SimpleAgent(
    tools=[weather_tool_definition, calculator_tool_definition],
    tool_functions={
        "get_current_weather": get_current_weather,
        "safe_calculate":      safe_calculate,
    },
    system_prompt=(
        "You are a helpful assistant with a weather tool and a calculator. "
        "Use them whenever relevant."
    ),
    max_iterations=5,
)

answer = agent.run("What is 15% of Tokyo's current humidity?")
print(f"\nFinal answer: {answer}")
```

---

## OpenAI Agents SDK

```python
# ============================================================
# OPENAI AGENTS SDK — Higher-level agent framework
# pip install openai-agents
# ============================================================

from agents import Agent, Runner, function_tool
import asyncio


# @function_tool auto-generates the JSON schema from type hints + docstring
# No need to write the JSON schema manually
@function_tool
def get_stock_price(ticker: str) -> dict:
    """
    Get the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker like AAPL, GOOGL, or MSFT

    Returns:
        Price and daily change percentage
    """
    mock = {
        "AAPL": {"price": 185.50, "change": "+1.2%"},
        "GOOGL": {"price": 141.80, "change": "-0.5%"},
        "MSFT":  {"price": 415.00, "change": "+0.8%"},
    }
    data = mock.get(ticker.upper())
    if data:
        return {"ticker": ticker.upper(), **data}
    return {"error": f"Ticker {ticker!r} not found"}


@function_tool
def calculate_portfolio_value(holdings: list) -> dict:
    """
    Calculate the total value of a stock portfolio.

    Args:
        holdings: List of dicts, each with 'ticker' (str) and 'shares' (int)
                  Example: [{"ticker": "AAPL", "shares": 10}]

    Returns:
        Total value and per-holding breakdown
    """
    total = 0.0
    breakdown = []

    for h in holdings:
        ticker = h.get("ticker", "").upper()
        shares = h.get("shares", 0)

        # Inline call — tools can call other tools
        price_data = get_stock_price(ticker)

        if "error" in price_data:
            breakdown.append({"ticker": ticker, "error": price_data["error"]})
            continue

        price = price_data["price"]
        value = price * shares
        total += value
        breakdown.append({"ticker": ticker, "shares": shares, "price": price, "value": round(value, 2)})

    return {"total_value": round(total, 2), "holdings": breakdown}


# Declare the agent — tools are just the decorated functions
portfolio_agent = Agent(
    name="Portfolio Assistant",
    instructions=(
        "You help users track their stock portfolios. "
        "Use get_stock_price and calculate_portfolio_value as needed. "
        "Always explain the numbers clearly."
    ),
    tools=[get_stock_price, calculate_portfolio_value],
    model="gpt-4o",
)


async def main():
    """Async runner — handles the full agent loop internally."""
    result = await Runner.run(
        portfolio_agent,
        "What is my portfolio worth if I own 10 AAPL and 5 GOOGL shares?",
    )
    print(result.final_output)


# Async entry point
asyncio.run(main())

# Synchronous alternative (simpler for scripts)
result = Runner.run_sync(portfolio_agent, "What is the current price of MSFT?")
print(result.final_output)
```

---

## Practice Questions

```
PRACTICE QUESTIONS — TOOLS & FUNCTION CALLING
============================================================

CONCEPTUAL (understand the architecture):

1.  Why can a plain LLM not fetch real-time stock prices?
    What architectural property makes this impossible?

2.  Walk through the 6-step function calling flow. At which step does
    the model "execute" the function? (Hint: it never does.)

3.  What does finish_reason == "tool_calls" mean? What should your
    code do when it sees this vs "stop"?

4.  What is the purpose of tool_call_id? What would break if you
    sent all tool results without including matching IDs?

5.  Describe the three tool_choice values "auto", "none", and "required".
    Give a real-world use case where each one is the best choice.

6.  What are the two key structural differences between OpenAI's and
    Anthropic's tool-use APIs? (hint: stop_reason naming and result format)

JSON SCHEMA:

7.  Write a complete tool definition for a "send_notification" function
    that accepts: recipient (string), message (string), channel (enum:
    slack/email/sms), priority (enum: low/normal/high, default normal),
    and attachments (optional array of strings).

8.  What is the difference between a property in "required" vs having
    a "default" key? How does your Python function handle the missing case?

SAFETY:

9.  Why is Python's eval() dangerous for a calculator tool?
    What specific attack does the AST-based approach prevent?

10. You are building a "delete_file" tool. List five safety measures
    you would implement. Write the validation function for one of them.

11. Name three OS-level isolation techniques for code execution.
    What is the minimum you should use before deploying a code-executor
    tool to production?

12. What happens to an agent without a max_iterations limit if the model
    keeps calling a tool that always returns an error? Write the scenario.

IMPLEMENTATION:

13. Write a tool definition for a REST API caller that accepts a URL,
    HTTP method, optional headers (object), and optional body (string).
    What security checks must the caller implement before making the request?

14. An agent needs weather for 10 cities simultaneously. Explain how
    parallel tool calls work and write pseudocode using ThreadPoolExecutor
    to execute all 10 calls in parallel.

15. Design a "file manager" agent tool set with read_file, write_file,
    and list_directory. What path-traversal attacks must you prevent?
    Write the path-validation helper function.
```

---

## Summary Diagram

```
FUNCTION CALLING — THE COMPLETE PICTURE
============================================================

  DESIGN TIME                 RUNTIME
  ───────────────────         ────────────────────────────────────────
  You write:                  1. USER sends message
    Python functions    ──>   2. YOUR APP sends (message + tool schemas) to API
    JSON schema docs          3. MODEL reads schemas, decides to call a tool
                              4. MODEL returns tool_call JSON (NOT final answer)
                              5. YOUR APP parses JSON, calls Python function
                              6. YOUR APP sends tool result back to API
                              7. MODEL reads result, produces final answer
                              8. YOUR APP returns answer to USER

  FLOW DIAGRAM:
  ┌──────────┐      ┌────────────┐      ┌────────────┐
  │  USER    │─────>│  YOUR APP  │─────>│  LLM API   │
  │          │      │            │<─────│(gpt-4o etc)│
  │          │      │  Executes  │      └────────────┘
  │          │      │  tools     │           ^
  │          │<─────│  locally   │──────────>│
  └──────────┘      └────────────┘   (sends tool results back)

  KEY PRINCIPLES:
  ┌────────────────────────────────────────────────────────┐
  │ 1. The LLM is the BRAIN — decides what to call         │
  │ 2. YOUR CODE is the HANDS — actually runs functions    │
  │ 3. Always validate args before executing               │
  │ 4. Always set max_iterations to prevent loops          │
  │ 5. Log every tool execution for debugging              │
  │ 6. High-risk tools need confirmation gates             │
  └────────────────────────────────────────────────────────┘
```
