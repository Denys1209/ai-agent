# ai-agent

A fully local AI agent that answers questions about a code repository and can create files in it. It runs [Gemma 4](https://ai.google.dev/gemma) through `llama-cpp-python`, exposed over gRPC, with a Python agent loop on top. No cloud API is involved, so your code never leaves your machine.

## Features

- **Local model over gRPC:** Gemma 4 (GGUF) served by `server.py`, with both a single-response and a token-streaming RPC.
- **ReAct-style agent loop:** the model replies with JSON, either a tool call or a final answer. Invalid JSON is fed back so the model can retry, and a step limit prevents endless loops.
- **Tool registry:** add a tool by writing a Python function with a docstring and a `@tool` decorator. The system prompt updates itself.
- **Semantic search:** Chroma embeddings let the agent find code by meaning (for example, "where are users authenticated?"), alongside exact-text `git grep` search.
- **Safe by default:** every path is confined to the project folder, and all write operations ask for `[y/N]` confirmation.
- **Memory and tracing:** conversation memory across questions, automatic trimming of old tool output, and a `trace.jsonl` log of every step.

## Architecture

```
agents.py  --gRPC-->  server.py  -->  Gemma 4 (llama.cpp)
    |
    +--> tools.py  (file access, git, search, Chroma index)
```

| File | Purpose |
|---|---|
| `server.py` | Loads the model and serves it over gRPC. A lock serializes model calls, since llama.cpp is not thread-safe. |
| `chat.proto` | gRPC contract: `Request`, `Response`, `Message`, and two RPCs (`MakeRequest`, `MakeRequestStreamBackTokenByToken`). |
| `chat_pb2.py`, `chat_pb2_grpc.py` | Generated from `chat.proto`. |
| `agents.py` | The agent: system prompt, JSON parsing, the tool loop, memory, trimming, and the chat prompt. |
| `tools.py` | Tool registry, path sandbox, and the semantic index. |

The server is stateless: the client sends the full conversation history with every request.

## Tools

| Tool | Description |
|---|---|
| `list_files(path)` | List a directory. |
| `read_file(path)` | Read a text file. |
| `search_code(pattern)` | Exact-text search with `git grep`. |
| `semantic_search(query, n)` | Search by meaning using Chroma embeddings. |
| `git_log(n)` | Show recent commits. |
| `write_file(path, content)` | Write or overwrite a file (asks for confirmation). |
| `create_file(path, content)` | Create a new file, refusing to overwrite (asks for confirmation). |
| `create_folder(path)` | Create a folder (asks for confirmation). |

## Requirements

- Python 3.10+
- A GGUF build of Gemma 4 and its multimodal projector file, placed next to `server.py`:
  - `gemma-4-E4B-it-Q8_0.gguf`
  - `mmproj-F16.gguf`
- A GPU is recommended. The server offloads all layers with `n_gpu_layers='all'`.
- `git` on your PATH (used by `search_code` and `git_log`).

## Installation

```powershell
pip install llama-cpp-python grpcio grpcio-tools protobuf chromadb "numpy<2"
```

`numpy<2` is pinned because the `onnxruntime` build that Chroma uses for embeddings can fail with NumPy 2.x.

If you change `chat.proto`, regenerate the gRPC code:

```powershell
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. chat.proto
```

## Usage

**1. Start the model server** (terminal 1). Wait for "Server started, listening on 50051".

```powershell
python server.py
```

**2. Start the agent** (terminal 2), pointing it at the project folder you want to explore:

```powershell
python agents.py C:\path\to\your\repo
```

Options:

| Argument | Default | Meaning |
|---|---|---|
| `directory` | `.` | Project folder the agent may read and write. |
| `--server` | `localhost:50051` | gRPC server address. |
| `--max-steps` | `8` | Maximum tool calls per question. |

On first run the agent indexes the project, which can take a minute on large folders. Later runs re-embed only changed files. The index is stored in `~/.agent_index/`, so your repositories stay clean.

**3. Ask questions:**

```
You: what files are in this project?
You: where is the model loaded and configured?
You: create a folder called notes and add notes/todo.md with three tasks
```

## How the agent loop works

1. The system prompt lists the available tools and requires exactly one JSON object per reply.
2. The model answers with `{"tool": "read_file", "args": {"path": "main.py"}}` or `{"final": "..."}`.
3. `agents.py` runs the tool, truncates the output, and sends it back as a `TOOL RESULT:` message.
4. The loop repeats until the model gives a final answer or hits the step limit.

Tool errors are returned to the model as text instead of crashing, so it can recover and try something else.

## Debugging

Every task, model reply, and tool result is appended to `trace.jsonl`, one JSON object per line. Open it to see exactly what the model said and what each tool returned.

## Adding a tool

```python
@tool
def word_count(path: str) -> str:
    """Count the words in a text file."""
    return str(len(read_file(path).split()))
```

Nothing else is needed. `describe_tools()` reads the function signature and docstring and adds it to the prompt.

## Troubleshooting

| Problem | Fix |
|---|---|
| `_ARRAY_API not found` / NumPy import error | `pip install "numpy<2"` |
| Connection refused | The server is still loading the model. Wait for "Server started". |
| `search_code` or `git_log` returns nothing | The target folder must be a git repository. |
| `Gemma4Server` attribute error in `server.py` | Inherit from `chat_pb2_grpc.Gemma4ServerServicer`. |
| Context overflow on long sessions | Raise `n_ctx` in `server.py` or lower the trim budget in `agents.py`. |

## Limitations

- Gemma is a small model, so it can occasionally choose the wrong tool or produce malformed JSON. The retry logic reduces this but does not eliminate it.
- Model calls are serialized, so the server handles one request at a time.
- Write tools require interactive confirmation in the terminal.
