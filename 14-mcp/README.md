# 14 — Model Context Protocol (MCP): The USB Standard for AI

## The Integration Chaos Problem: N×M vs N+M

```
WITHOUT MCP: THE N×M INTEGRATION PROBLEM
============================================================

Every AI application needs custom code to connect to every tool.
If you have N AI apps and M tools, you need N×M integrations.

  3 AI apps × 5 tools = 15 custom integrations

  ┌──────────────┐          ┌──────────────┐
  │  Claude      │ ─────────│  GitHub      │
  │  Desktop     │ ─────────│  API         │
  │              │ ─────────│              │
  └──────────────┘    ╳     └──────────────┘
  ┌──────────────┐    ╳     ┌──────────────┐
  │  Cursor IDE  │ ─────────│  Filesystem  │
  │              │ ─────────│              │
  └──────────────┘    ╳     └──────────────┘
  ┌──────────────┐    ╳     ┌──────────────┐
  │  Your App    │ ─────────│  Database    │
  │              │ ─────────│              │
  └──────────────┘          └──────────────┘

  3 AI apps × 5 tools = 15 custom integrations
  Each one requires:
    - Authentication handling
    - Data format conversion
    - Error handling
    - Maintenance when APIs change

WITH MCP: THE N+M SOLUTION
============================================================

Each AI app implements MCP CLIENT once.
Each tool implements MCP SERVER once.
They speak a common protocol.

  ┌──────────────┐     MCP     ┌──────────────┐
  │  Claude      │◄───────────►│  GitHub      │
  │  Desktop     │     MCP     │  MCP Server  │
  │  (client)    │◄───────────►│              │
  └──────────────┘     MCP     └──────────────┘
  ┌──────────────┐◄───────────►┌──────────────┐
  │  Cursor IDE  │     MCP     │  Filesystem  │
  │  (client)    │◄───────────►│  MCP Server  │
  └──────────────┘     MCP     └──────────────┘
  ┌──────────────┐◄───────────►┌──────────────┐
  │  Your App    │             │  Database    │
  │  (client)    │             │  MCP Server  │
  └──────────────┘             └──────────────┘

  3 AI apps + 5 tools = 8 implementations (not 15!)
  Any client works with any server automatically.

WHY "USB STANDARD FOR AI":
  USB: Any USB device works with any USB port.
  MCP: Any MCP server works with any MCP client.
  No custom drivers. No custom integrations.
```

---

## MCP Architecture: Host, Client, Server

```
MCP COMPONENT ARCHITECTURE
============================================================

┌─────────────────────────────────────────────────────────┐
│                    HOST APPLICATION                       │
│           (Claude Desktop, Cursor, your app)             │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │                  MCP CLIENT                       │   │
│  │                                                   │   │
│  │  Responsibilities:                                │   │
│  │  - Manages connections to MCP servers             │   │
│  │  - Sends requests (list_tools, call_tool, etc.)   │   │
│  │  - Receives and caches server capabilities        │   │
│  │  - Injects tool/resource info into LLM context    │   │
│  │  - Routes tool_calls from LLM to correct server   │   │
│  └──────────────────────────────────────────────────┘   │
│                            │                             │
│              MCP Protocol  │  (JSON-RPC 2.0)            │
│                            │                             │
└────────────────────────────┼─────────────────────────────┘
                             │
         ┌───────────────────┼──────────────────┐
         │                   │                  │
         ▼                   ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  MCP SERVER A   │ │  MCP SERVER B   │ │  MCP SERVER C   │
│  (filesystem)   │ │  (github)       │ │  (your db)      │
│                 │ │                 │ │                 │
│ Resources:      │ │ Resources:      │ │ Resources:      │
│  - file://...   │ │  - repo://...   │ │  - table://...  │
│                 │ │                 │ │                 │
│ Tools:          │ │ Tools:          │ │ Tools:          │
│  - read_file    │ │  - search_repos │ │  - query_db     │
│  - write_file   │ │  - create_issue │ │  - insert_row   │
│  - list_dir     │ │  - list_prs     │ │                 │
│                 │ │                 │ │ Prompts:        │
│ Prompts:        │ │ Prompts:        │ │  - sql_helper   │
│  - edit_file    │ │  - pr_review    │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘

TRANSPORT LAYER:
  stdio     — Server runs as subprocess; stdin/stdout communication
              Used for: local tools, CLI-based servers
              Pro: Simple, no networking, secure
              Con: Single host, process overhead

  HTTP+SSE  — Server runs as HTTP service; SSE for streaming
              Used for: remote services, shared servers, production
              Pro: Remote access, multiple clients, scalable
              Con: Network overhead, auth complexity
```

---

## 3 Capabilities: Resources, Tools, Prompts

```
MCP CAPABILITIES IN DEPTH
============================================================

CAPABILITY 1: RESOURCES
════════════════════════
Resources are DATA that the AI can READ.
They are analogous to files or URLs — data sources, not actions.

URI scheme examples:
  file:///path/to/file.txt        — local file
  github://repo/owner/file.md     — GitHub file
  database://mydb/users           — database table
  memory://conversation-history   — in-memory data

Resources can be:
  - Static: content that doesn't change (a configuration file)
  - Dynamic: content that changes (live database records)
  - List-able: clients can browse available resources
  - Subscribe-able: clients can get notifications when content changes

Use resources when:
  - The AI needs to READ existing data
  - The data can be browsed or listed
  - The data is NOT something the AI creates or modifies

CAPABILITY 2: TOOLS
════════════════════
Tools are ACTIONS the AI can EXECUTE.
They are analogous to function calls — doing something, not reading.

Examples:
  search_web(query)       — side effect: network request
  write_file(path, data)  — side effect: file modification
  send_email(to, body)    — side effect: email sent
  run_query(sql)          — may modify database

Tools always have:
  - A name (snake_case convention)
  - A description (how the AI decides when to use it)
  - An inputSchema (JSON Schema for arguments)

Use tools when:
  - The AI needs to DO something
  - There are side effects
  - The result depends on inputs

CAPABILITY 3: PROMPTS
═════════════════════
Prompts are TEMPLATES that let users invoke common workflows.
They are reusable, parameterized prompts that appear in the UI.

Example: A "review code" prompt template:
  Name: review_code
  Description: Review code for bugs and suggest improvements
  Arguments: [{name: "language", description: "Programming language"}]
  Template: "Review this {language} code for bugs,
             performance issues, and style..."

When to use prompts:
  - Common, repeated workflows
  - Tasks that need careful prompt engineering
  - Exposing workflows to non-technical users in Claude Desktop
```

---

## Transport Layer: stdio vs HTTP+SSE

```
TRANSPORT COMPARISON
============================================================

STDIO TRANSPORT
══════════════
  ┌──────────────┐     stdin/stdout     ┌──────────────┐
  │  MCP Client  │◄────────────────────►│  MCP Server  │
  │              │                      │  (subprocess)│
  └──────────────┘                      └──────────────┘

  Communication:
    Client spawns server as subprocess
    Messages sent over stdin (client→server)
    Responses sent over stdout (server→client)
    stderr is for logs (not protocol messages)

  Message format:
    Newline-delimited JSON-RPC 2.0
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

  Best for:
    - Local tools (filesystem, git, local database)
    - Development and testing
    - Security-sensitive tools (no network exposure)
    - Claude Desktop configuration

HTTP+SSE TRANSPORT
═══════════════════
  ┌──────────────┐     HTTP POST     ┌──────────────┐
  │  MCP Client  │──────────────────►│  MCP Server  │
  │              │◄──────────────────│  (HTTP API)  │
  │              │  SSE (streaming)  │              │
  └──────────────┘                   └──────────────┘

  Communication:
    Client sends requests via HTTP POST
    Server sends responses and notifications via SSE
    SSE = Server-Sent Events (one-way streaming from server)

  Best for:
    - Remote/cloud-hosted tools
    - Shared servers (multiple clients connecting)
    - Production deployments
    - When tools need to push notifications to client

  Endpoints:
    GET  /sse         — Establish SSE connection
    POST /messages    — Send requests
```

---

## Building a Complete MCP Server in Python

```python
# ============================================================
# COMPLETE MCP SERVER — NOTES MANAGER
# A fully working example with all 3 capability types
# pip install mcp
# ============================================================

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)


# ---- Data storage ----
# In production, use a database. Here we use a JSON file.

NOTES_FILE = Path(os.environ.get("NOTES_FILE", "/tmp/mcp_notes.json"))


def load_notes() -> dict:
    """Load notes from the JSON file. Returns empty dict if file doesn't exist."""
    if NOTES_FILE.exists():
        with open(NOTES_FILE) as f:
            return json.load(f)
    return {}


def save_notes(notes: dict):
    """Persist notes to the JSON file."""
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)


# ---- Create the MCP server ----

server = Server("notes-manager")   # Server name (shown in client UIs)


# ============================================================
# CAPABILITY 1: RESOURCES
# Let clients list and read notes as resources
# ============================================================

@server.list_resources()
async def list_resources() -> ListResourcesResult:
    """
    Return a list of all available resources.

    This is called when the client connects or refreshes.
    Each resource has a URI, name, and optional description.
    """
    notes = load_notes()

    resources = [
        Resource(
            uri=f"note:///{note_id}",   # Custom URI scheme
            name=f"Note: {note['title']}",
            description=f"Created: {note['created_at']}. Content preview: {note['content'][:50]}...",
            mimeType="text/plain",      # MIME type of the resource content
        )
        for note_id, note in notes.items()
    ]

    # Always include a special "all notes" resource for overview
    resources.insert(0, Resource(
        uri="note:///all",
        name="All Notes",
        description="Index of all notes with titles and timestamps",
        mimeType="application/json",
    ))

    return ListResourcesResult(resources=resources)


@server.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    """
    Return the content of a specific resource by URI.

    This is called when the AI or user requests to read a resource.
    """
    notes = load_notes()

    # Handle the special "all notes" resource
    if uri == "note:///all":
        overview = {
            note_id: {
                "title":      note["title"],
                "created_at": note["created_at"],
                "preview":    note["content"][:100],
            }
            for note_id, note in notes.items()
        }
        return ReadResourceResult(
            contents=[
                TextContent(
                    type="text",
                    text=json.dumps(overview, indent=2),
                    uri=uri,
                )
            ]
        )

    # Handle individual note resources
    # URI format: note:///note_id
    note_id = uri.replace("note:///", "")

    if note_id not in notes:
        raise ValueError(f"Note not found: {note_id}")

    note = notes[note_id]
    content = (
        f"Title: {note['title']}\n"
        f"Created: {note['created_at']}\n"
        f"Updated: {note.get('updated_at', 'Never')}\n"
        f"Tags: {', '.join(note.get('tags', []))}\n\n"
        f"{note['content']}"
    )

    return ReadResourceResult(
        contents=[
            TextContent(type="text", text=content, uri=uri)
        ]
    )


# ============================================================
# CAPABILITY 2: TOOLS
# Let the AI create, update, delete, and search notes
# ============================================================

@server.list_tools()
async def list_tools() -> ListToolsResult:
    """
    Declare all tools this server provides.

    Called during initialization and when the client refreshes capabilities.
    """
    return ListToolsResult(
        tools=[
            Tool(
                name="create_note",
                description=(
                    "Create a new note with a title and content. "
                    "Returns the ID of the created note."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the note",
                        },
                        "content": {
                            "type": "string",
                            "description": "Main content of the note",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of tags for organization",
                        },
                    },
                    "required": ["title", "content"],
                },
            ),
            Tool(
                name="update_note",
                description="Update the title, content, or tags of an existing note.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_id": {
                            "type": "string",
                            "description": "ID of the note to update",
                        },
                        "title": {
                            "type": "string",
                            "description": "New title (omit to keep existing)",
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (omit to keep existing)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New tags (omit to keep existing)",
                        },
                    },
                    "required": ["note_id"],
                },
            ),
            Tool(
                name="delete_note",
                description="Permanently delete a note by its ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_id": {
                            "type": "string",
                            "description": "ID of the note to delete",
                        },
                    },
                    "required": ["note_id"],
                },
            ),
            Tool(
                name="search_notes",
                description=(
                    "Search notes by keyword. Searches titles, content, and tags. "
                    "Returns matching note IDs, titles, and relevance snippets."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keyword or phrase to search for",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]
    )


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """
    Execute a tool call.

    This is called whenever the LLM (or user) invokes a tool.
    Route to the correct handler based on tool name.
    """
    if name == "create_note":
        return await _create_note(arguments)
    elif name == "update_note":
        return await _update_note(arguments)
    elif name == "delete_note":
        return await _delete_note(arguments)
    elif name == "search_notes":
        return await _search_notes(arguments)
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )


async def _create_note(args: dict) -> CallToolResult:
    """Create a new note."""
    notes = load_notes()

    # Generate a unique ID using timestamp
    note_id = f"note_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    notes[note_id] = {
        "title":      args["title"],
        "content":    args["content"],
        "tags":       args.get("tags", []),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": None,
    }

    save_notes(notes)

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Note created successfully. ID: {note_id}",
            )
        ]
    )


async def _update_note(args: dict) -> CallToolResult:
    """Update an existing note."""
    notes = load_notes()
    note_id = args["note_id"]

    if note_id not in notes:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Note not found: {note_id}")],
            isError=True,
        )

    note = notes[note_id]

    # Only update fields that were provided
    if "title" in args:
        note["title"] = args["title"]
    if "content" in args:
        note["content"] = args["content"]
    if "tags" in args:
        note["tags"] = args["tags"]

    note["updated_at"] = datetime.utcnow().isoformat()

    save_notes(notes)

    return CallToolResult(
        content=[TextContent(type="text", text=f"Note {note_id} updated successfully.")]
    )


async def _delete_note(args: dict) -> CallToolResult:
    """Delete a note."""
    notes = load_notes()
    note_id = args["note_id"]

    if note_id not in notes:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Note not found: {note_id}")],
            isError=True,
        )

    title = notes[note_id]["title"]
    del notes[note_id]
    save_notes(notes)

    return CallToolResult(
        content=[TextContent(type="text", text=f"Deleted note '{title}' (ID: {note_id}).")]
    )


async def _search_notes(args: dict) -> CallToolResult:
    """Search notes by keyword."""
    notes = load_notes()
    query = args["query"].lower()

    matches = []
    for note_id, note in notes.items():
        # Simple keyword search — in production use full-text search or embeddings
        searchable = (
            note["title"].lower()
            + " "
            + note["content"].lower()
            + " "
            + " ".join(note.get("tags", []))
        )

        if query in searchable:
            # Find the first occurrence in content for a snippet
            idx     = note["content"].lower().find(query)
            snippet = (
                note["content"][max(0, idx - 30) : idx + 70]
                if idx >= 0
                else note["content"][:80]
            )

            matches.append({
                "id":      note_id,
                "title":   note["title"],
                "snippet": snippet,
            })

    if not matches:
        return CallToolResult(
            content=[TextContent(type="text", text=f"No notes found matching: {query}")]
        )

    result_text = f"Found {len(matches)} note(s) matching '{query}':\n\n"
    for m in matches:
        result_text += f"  [{m['id']}] {m['title']}\n    ...{m['snippet']}...\n\n"

    return CallToolResult(
        content=[TextContent(type="text", text=result_text)]
    )


# ============================================================
# CAPABILITY 3: PROMPTS
# Reusable templates for common workflows
# ============================================================

@server.list_prompts()
async def list_prompts() -> ListPromptsResult:
    """
    Declare available prompt templates.

    Prompts appear in Claude Desktop's UI as slash commands
    (e.g., /notes-manager:summarize_notes).
    """
    return ListPromptsResult(
        prompts=[
            Prompt(
                name="summarize_notes",
                description="Generate a summary of all notes grouped by topic",
                arguments=[
                    PromptArgument(
                        name="focus_area",
                        description="Optional topic to focus the summary on",
                        required=False,
                    )
                ],
            ),
            Prompt(
                name="create_from_idea",
                description="Turn a rough idea into a well-structured note",
                arguments=[
                    PromptArgument(
                        name="idea",
                        description="Your rough idea or notes",
                        required=True,
                    ),
                    PromptArgument(
                        name="category",
                        description="Category for the note (e.g., work, personal, ideas)",
                        required=False,
                    ),
                ],
            ),
        ]
    )


@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    """
    Return the resolved content of a prompt template.

    Arguments are substituted into the template here.
    """
    if name == "summarize_notes":
        focus = arguments.get("focus_area", "")
        focus_clause = f" Focus specifically on notes related to: {focus}." if focus else ""

        return GetPromptResult(
            description="Summarize all notes",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            "Please read all my notes (use the 'All Notes' resource) "
                            "and create a clear, organized summary grouped by topic."
                            f"{focus_clause} "
                            "Include key insights and action items you notice."
                        ),
                    ),
                )
            ],
        )

    elif name == "create_from_idea":
        idea     = arguments.get("idea", "")
        category = arguments.get("category", "general")

        return GetPromptResult(
            description="Create a structured note from a rough idea",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Take this rough idea and turn it into a well-structured note:\n\n"
                            f"{idea}\n\n"
                            f"Create a note with:\n"
                            f"- A clear, descriptive title\n"
                            f"- Well-organized content with sections if appropriate\n"
                            f"- Category/tags: {category}\n\n"
                            f"Use the create_note tool to save it."
                        ),
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ============================================================
# SERVER ENTRY POINT
# ============================================================

async def main():
    """Start the MCP server using stdio transport."""
    # stdio_server manages the stdin/stdout communication
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="notes-manager",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## Building an MCP Client

```python
# ============================================================
# MCP CLIENT — Connect to and use an MCP server
# ============================================================

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_mcp_client():
    """
    Connect to the notes-manager MCP server and demonstrate all capabilities.

    The client:
    1. Starts the server as a subprocess
    2. Discovers capabilities (tools, resources, prompts)
    3. Makes calls to demonstrate each capability
    """

    # Configure how to start the server process
    server_params = StdioServerParameters(
        command="python",               # The command to run
        args=["notes_server.py"],       # Arguments to the command
        env={"NOTES_FILE": "/tmp/demo_notes.json"},  # Environment variables
    )

    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # Step 1: Initialize — exchange capabilities
            await session.initialize()
            print("Connected to MCP server: notes-manager")


            # Step 2: List available tools
            tools_result = await session.list_tools()
            print(f"\nAvailable tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")


            # Step 3: Call a tool — create a note
            create_result = await session.call_tool(
                "create_note",
                {
                    "title":   "MCP Demo Note",
                    "content": "This note was created by the MCP client demo.",
                    "tags":    ["demo", "mcp"],
                }
            )
            print(f"\nCreate note result: {create_result.content[0].text}")


            # Step 4: List resources — see the note we just created
            resources_result = await session.list_resources()
            print(f"\nAvailable resources ({len(resources_result.resources)}):")
            for resource in resources_result.resources:
                print(f"  - {resource.uri}: {resource.name}")


            # Step 5: Read a resource — read the "all notes" overview
            read_result = await session.read_resource("note:///all")
            print(f"\nAll notes overview:")
            print(read_result.contents[0].text[:300])


            # Step 6: Call search tool
            search_result = await session.call_tool(
                "search_notes",
                {"query": "demo"}
            )
            print(f"\nSearch results: {search_result.content[0].text}")


            # Step 7: List prompts
            prompts_result = await session.list_prompts()
            print(f"\nAvailable prompts ({len(prompts_result.prompts)}):")
            for prompt in prompts_result.prompts:
                print(f"  - {prompt.name}: {prompt.description}")


            # Step 8: Get a prompt template
            prompt_result = await session.get_prompt(
                "create_from_idea",
                {"idea": "Build a reminder system", "category": "projects"}
            )
            print(f"\nPrompt template resolved:")
            print(prompt_result.messages[0].content.text[:200])


asyncio.run(run_mcp_client())
```

---

## Claude Desktop Configuration

```json
// ============================================================
// CLAUDE DESKTOP CONFIG — claude_desktop_config.json
// Add MCP servers so Claude Desktop can use them
// ============================================================

// LOCATION:
//   macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
//   Windows: %APPDATA%\Claude\claude_desktop_config.json
//   Linux:   ~/.config/Claude/claude_desktop_config.json

// AFTER EDITING: Restart Claude Desktop completely (quit, not just close window)

{
  "mcpServers": {

    // ---- LOCAL PYTHON SERVER (stdio) ----
    "notes-manager": {
      "command": "python",
      "args": ["/Users/yourname/notes_server.py"],
      "env": {
        "NOTES_FILE": "/Users/yourname/notes.json"
      }
    },

    // ---- OFFICIAL FILESYSTEM SERVER ----
    // npm install -g @modelcontextprotocol/server-filesystem
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/yourname/Documents",
        "/Users/yourname/Desktop"
      ]
    },

    // ---- OFFICIAL GITHUB SERVER ----
    // Requires a GitHub Personal Access Token
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    },

    // ---- OFFICIAL SQLITE SERVER ----
    // uvx is part of uv (pip install uv)
    "sqlite": {
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "/Users/yourname/data/mydb.sqlite"
      ]
    },

    // ---- OFFICIAL BRAVE SEARCH SERVER ----
    // Requires Brave Search API key: https://brave.com/search/api/
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "BSA_your_api_key_here"
      }
    },

    // ---- OFFICIAL POSTGRES SERVER ----
    // Connect to a PostgreSQL database
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://localhost/mydb"
      ]
    },

    // ---- PYTHON VIRTUALENV SERVER ----
    // Use a specific virtual environment
    "my-custom-server": {
      "command": "/Users/yourname/.venv/bin/python",
      "args": ["/Users/yourname/custom_server.py"],
      "env": {
        "API_KEY": "your_secret_key",
        "DATABASE_URL": "postgresql://localhost/prod"
      }
    }
  }
}
```

---

## All Official Servers with Configuration

```
OFFICIAL MCP SERVERS (by Anthropic)
============================================================

SERVER: filesystem
  Purpose: Read and write files on the local filesystem
  Repo:    @modelcontextprotocol/server-filesystem
  Tools:   read_file, write_file, list_directory, move_file,
           search_files, get_file_info, create_directory
  Config:  Pass directory paths as arguments to restrict access
  Use for: Document editing, code generation, file management

SERVER: github
  Purpose: Interact with GitHub repositories
  Repo:    @modelcontextprotocol/server-github
  Tools:   search_repositories, create_issue, list_pull_requests,
           get_file_contents, create_or_update_file, list_commits
  Config:  Requires GITHUB_PERSONAL_ACCESS_TOKEN env var
  Use for: Code review, issue management, repository exploration

SERVER: postgres
  Purpose: Query and modify PostgreSQL databases
  Repo:    @modelcontextprotocol/server-postgres
  Tools:   query (read-only by default)
  Config:  Pass connection string as argument
  Use for: Data analysis, schema exploration, report generation

SERVER: sqlite
  Purpose: Query SQLite database files
  Repo:    mcp-server-sqlite (Python)
  Tools:   read_query, write_query, create_table, list_tables,
           describe_table, append_insight
  Config:  --db-path argument for database file location
  Use for: Local data analysis, prototyping, small applications

SERVER: brave-search
  Purpose: Real-time web search via Brave Search API
  Repo:    @modelcontextprotocol/server-brave-search
  Tools:   brave_web_search, brave_local_search
  Config:  Requires BRAVE_API_KEY env var
  Use for: Current events, fact-checking, research

SERVER: puppeteer (browser automation)
  Purpose: Control a Chrome browser programmatically
  Repo:    @modelcontextprotocol/server-puppeteer
  Tools:   puppeteer_navigate, puppeteer_click,
           puppeteer_screenshot, puppeteer_fill
  Use for: Web scraping, UI testing, form filling

SERVER: memory (knowledge graph)
  Purpose: Persistent key-value memory across conversations
  Repo:    @modelcontextprotocol/server-memory
  Tools:   create_entities, create_relations,
           search_nodes, read_graph
  Use for: Personal assistant memory, knowledge accumulation

SERVER: google-drive
  Purpose: Access Google Drive files and folders
  Repo:    @modelcontextprotocol/server-gdrive
  Tools:   search, read_file (converts to text)
  Use for: Document retrieval, Google Workspace integration
```

---

## Production HTTP Server with FastAPI and SSE

```python
# ============================================================
# PRODUCTION MCP SERVER — HTTP + SSE TRANSPORT
# Use this for remote/shared deployments
# pip install mcp fastapi uvicorn
# ============================================================

import asyncio
import json
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mcp.server import Server
from mcp.types import (
    CallToolResult, ListToolsResult, TextContent, Tool
)


# ---- MCP Server logic ----

mcp_server = Server("remote-notes-manager")


@mcp_server.list_tools()
async def list_tools() -> ListToolsResult:
    """Same tool definitions as the stdio version."""
    return ListToolsResult(tools=[
        Tool(
            name="echo",
            description="Echo back the input message",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"}
                },
                "required": ["message"],
            },
        )
    ])


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name == "echo":
        return CallToolResult(
            content=[TextContent(type="text", text=f"Echo: {arguments['message']}")]
        )
    return CallToolResult(
        content=[TextContent(type="text", text=f"Unknown tool: {name}")],
        isError=True,
    )


# ---- SSE connection manager ----

class SSEConnectionManager:
    """
    Manages Server-Sent Events connections.

    Each MCP client gets a unique connection ID.
    Messages from the MCP server are sent to the right client via SSE.
    """

    def __init__(self):
        # Map connection_id -> asyncio.Queue for outgoing messages
        self.connections: dict[str, asyncio.Queue] = {}

    def create_connection(self) -> str:
        """Create a new SSE connection, return its ID."""
        conn_id = str(uuid.uuid4())
        self.connections[conn_id] = asyncio.Queue()
        return conn_id

    def close_connection(self, conn_id: str):
        """Remove a connection when the client disconnects."""
        self.connections.pop(conn_id, None)

    async def send(self, conn_id: str, data: str):
        """Queue a message for a specific connection."""
        if conn_id in self.connections:
            await self.connections[conn_id].put(data)

    async def stream(self, conn_id: str) -> AsyncGenerator[str, None]:
        """Async generator that yields SSE messages for a connection."""
        queue = self.connections.get(conn_id)
        if not queue:
            return

        try:
            while True:
                # Wait for the next message (with timeout to detect disconnects)
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {message}\n\n"   # SSE format: "data: ...\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # SSE heartbeat (comment line)
        except asyncio.CancelledError:
            pass   # Client disconnected


manager = SSEConnectionManager()


# ---- FastAPI application ----

app = FastAPI(title="Notes Manager MCP Server")


@app.get("/sse")
async def sse_endpoint(request: Request):
    """
    SSE endpoint — clients connect here to receive messages from the server.

    The client makes a GET request and keeps the connection open.
    The server sends events whenever it has something to send.
    """
    conn_id = manager.create_connection()

    async def event_generator():
        # Send the connection ID first so the client knows where to POST
        yield f"data: {json.dumps({'type': 'connection', 'id': conn_id})}\n\n"

        # Stream subsequent messages
        async for event in manager.stream(conn_id):
            # Check if client has disconnected
            if await request.is_disconnected():
                break
            yield event

        manager.close_connection(conn_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",   # Disable nginx buffering for SSE
        },
    )


@app.post("/messages")
async def messages_endpoint(request: Request):
    """
    Messages endpoint — clients POST JSON-RPC requests here.

    The response is sent back via SSE to the client's open connection.
    """
    # Get the connection ID from the query parameter or header
    conn_id = request.query_params.get("connection_id")
    if not conn_id or conn_id not in manager.connections:
        return {"error": "Invalid or missing connection_id"}

    # Parse the JSON-RPC request
    body = await request.json()

    # Process the request through the MCP server
    # (In production, use the actual MCP transport handler)
    method   = body.get("method", "")
    req_id   = body.get("id")

    # Route to appropriate handler
    if method == "tools/list":
        result = await mcp_server._tool_handlers["list"]()
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": [t.model_dump() for t in result.tools]},
        }
    elif method == "tools/call":
        params   = body.get("params", {})
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        result    = await mcp_server._tool_handlers["call"](tool_name, tool_args)
        response  = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [c.model_dump() for c in result.content]},
        }
    else:
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    # Send response via SSE to the client
    await manager.send(conn_id, json.dumps(response))

    return {"status": "queued"}


@app.get("/health")
async def health():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status":      "healthy",
        "server":      "notes-manager",
        "connections": len(manager.connections),
    }


# Start the server
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",   # Listen on all interfaces
        port=8000,
        log_level="info",
    )
```

---

## MCP vs Tool Calling Comparison

```
MCP vs FUNCTION CALLING: WHEN TO USE WHICH
============================================================

                    FUNCTION CALLING          MCP
                    ════════════════          ═══
Architecture:       In-process               Out-of-process
                    Tool code runs in         Tool code runs in
                    your app's process        a separate process/server

Coupling:           Tight — your app         Loose — any MCP client
                    owns the tool code        can use any MCP server

Reusability:        Low — code is tied        High — one server,
                    to one application        many clients

Shared state:       Easy — Python objects     Harder — must serialize
                                              all state to JSON

Latency:            Very low (in-process)     Low (subprocess) to
                                              Medium (HTTP)

Best for:           - Simple tools            - Tools shared across apps
                    - Tight integration       - Independent tool teams
                    - Single-app tools        - Remote/cloud tools
                    - Maximum control         - Pluggable ecosystem

Examples:           - "Get weather from       - "Let Claude Desktop,
                      MY weather API"           Cursor, AND my app
                    - "Query MY database"       all use the same
                    - "Run MY code sandbox"     GitHub server"

DECISION RULE:
  - Building a single app? Use function calling.
  - Building tools for others to use? Use MCP.
  - Want Claude Desktop to use your tools? You MUST use MCP.
  - Want programmatic control over execution? Use function calling.
```

---

## MCP Security

```python
# ============================================================
# MCP SECURITY CONSIDERATIONS
# ============================================================

# PRINCIPLE 1: LEAST PRIVILEGE
# Only expose the capabilities that are actually needed.
# If your server only needs to READ, don't expose write tools.

# Example: Read-only filesystem server
ALLOWED_PATHS = [
    "/Users/yourname/Documents",
    "/Users/yourname/Desktop",
]

def validate_path(path: str) -> bool:
    """Ensure path is within allowed directories."""
    from pathlib import Path
    requested = Path(path).resolve()
    for allowed in ALLOWED_PATHS:
        try:
            requested.relative_to(Path(allowed).resolve())
            return True   # Path is within an allowed directory
        except ValueError:
            continue
    return False          # Path is outside all allowed directories


# PRINCIPLE 2: INPUT VALIDATION
# Validate all inputs in call_tool before executing.
# The LLM can generate arbitrary argument values.

def validate_tool_input(tool_name: str, arguments: dict) -> str | None:
    """
    Validate tool arguments. Returns error string or None if valid.
    """
    if tool_name == "read_file":
        path = arguments.get("path", "")
        if not path:
            return "path is required"
        if ".." in path:
            return "Path traversal (.. ) not allowed"
        if not validate_path(path):
            return f"Path {path!r} is outside allowed directories"

    elif tool_name == "write_file":
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        if not path:
            return "path is required"
        if ".." in path:
            return "Path traversal not allowed"
        if not validate_path(path):
            return f"Path {path!r} is outside allowed directories"
        if len(content) > 10 * 1024 * 1024:   # 10MB limit
            return "Content too large (max 10MB)"

    return None   # No error


# PRINCIPLE 3: RATE LIMITING
# Prevent abuse by limiting how fast tools can be called.

import time
from collections import defaultdict

_call_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX    = 100  # max calls per window


def check_rate_limit(tool_name: str) -> bool:
    """Returns True if the call is allowed, False if rate limited."""
    now = time.time()
    calls = _call_counts[tool_name]

    # Remove calls outside the window
    _call_counts[tool_name] = [t for t in calls if now - t < RATE_LIMIT_WINDOW]

    if len(_call_counts[tool_name]) >= RATE_LIMIT_MAX:
        return False   # Rate limit exceeded

    _call_counts[tool_name].append(now)
    return True   # Allowed


# PRINCIPLE 4: AUDIT LOGGING
# Log every tool invocation for security audit trails.

import logging

security_logger = logging.getLogger("mcp.security")


def log_tool_call(tool_name: str, arguments: dict, result_status: str):
    """Log every tool call with relevant context."""
    security_logger.info(
        "tool_call",
        extra={
            "tool":      tool_name,
            "args_keys": list(arguments.keys()),   # Log keys but not values (may contain secrets)
            "status":    result_status,
            "timestamp": time.time(),
        }
    )


# PRINCIPLE 5: SANDBOXING
# For dangerous tools (code execution), run in an isolated environment.
# Use Docker, Firecracker, or E2B — not just subprocess.

# Bad: subprocess.run(["python", "-c", user_code])
# Better: docker run --rm --network=none --memory=256m python-sandbox python -c "..."
# Best: Use a purpose-built sandbox like E2B (https://e2b.dev)
```

---

## Practice Questions

```
PRACTICE QUESTIONS — MODEL CONTEXT PROTOCOL
============================================================

CONCEPTUAL:
1.  Explain the N×M vs N+M integration problem that MCP solves.
    If you have 5 AI applications and 10 tools, how many integrations
    do you need with and without MCP? Why is this important as the
    ecosystem grows?

2.  What are the three MCP capabilities? For each one, describe:
    (a) what it is
    (b) when to use it
    (c) one concrete example

3.  Compare stdio and HTTP+SSE transports. For each one, describe the
    communication mechanism, a use case where it excels, and a
    limitation.

4.  What is the JSON-RPC 2.0 protocol? Why does MCP use it instead
    of a custom protocol? What does the "2.0" mean?

5.  What is the difference between a Tool and a Resource in MCP?
    If you have a database, would you expose tables as Resources or
    build a query Tool? Explain your reasoning.

SERVER IMPLEMENTATION:
6.  Implement a minimal MCP server that exposes one tool: get_timestamp()
    that returns the current UTC time in ISO format. Include the full
    server setup, list_tools, and call_tool handlers.

7.  Your MCP server needs to handle both reading files (Resource) and
    writing files (Tool). Explain why these belong in different
    capability types. What would break if you implemented file reading
    as a Tool instead?

8.  Write the call_tool handler for a "send_slack_message" tool.
    Include: input validation, rate limiting (max 10 messages/minute),
    audit logging, and proper error handling.

9.  Your MCP server needs to read from a PostgreSQL database.
    Write the connection setup code, list_resources (list all tables),
    and read_resource (execute a SELECT on a specific table) handlers.
    Include connection pooling and SQL injection prevention.

10. Implement resource subscriptions: when a file changes on disk,
    the MCP server should notify connected clients. What MCP
    capability enables this? Write the server-side code.

CLIENT AND INTEGRATION:
11. Write the Claude Desktop config for three servers:
    (a) Your Python notes-manager server
    (b) The official filesystem server restricted to ~/Documents
    (c) The GitHub server with authentication
    What file do you edit and where is it on macOS?

12. You built an MCP server for your team's internal wiki. Multiple
    developers want to use it with Claude Desktop, Cursor, and a
    custom chat app. What transport would you choose and why?
    Write the server startup command.

13. When would you use MCP vs function calling for building a
    "search internal documentation" feature? Write both implementations
    and explain the tradeoffs.

SECURITY:
14. An attacker sends a tool call with path="/etc/passwd" to your
    filesystem MCP server. Write the complete input validation logic
    that blocks this attack. What other path-traversal patterns
    should you check for?

15. Your MCP server exposes a "run_sql" tool. List five SQL injection
    attack vectors, and show how to prevent each one. Then explain
    why you should use read-only database credentials for query tools.
```
