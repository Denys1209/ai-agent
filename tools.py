import inspect, subprocess
import hashlib, os
import chromadb
from pathlib import Path


ROOT = Path(".").resolve()
TOOLS = {}

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".idea", ".vscode"}
TEXT_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cs", ".go", ".rs", ".c", ".h",
            ".cpp", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css",
            ".proto", ".sh", ".ps1"}

MAX_FILE_BYTES = 200_000
CHUCK_LINES, OVERLAP = 40, 10

_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        key = hashlib.md5(str(ROOT).encode()).hexdigest()[:12]
        client = chromadb.PersistentClient(path=str(Path.home() / ".agent_index" / key))
        _collection = client.get_or_create_collection("code", metadata={"hnsw:space": "cosine"})
    return _collection

def _source_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in TEXT_EXT and p.stat().st_size <= MAX_FILE_BYTES:
                yield p

def _chunks(text):
    lines = text.splitlines()
    step = CHUCK_LINES - OVERLAP
    for start in range(0, max(len(lines), 1), step):
        piece = lines[start:start + CHUCK_LINES]
        if "".join(piece).strip():
            yield start + 1, start + len(piece), "\n".join(piece)
        if start + CHUCK_LINES >= len(lines):
            break
def index_file(p: Path):
    col = _get_collection()
    rel = p.relative_to(ROOT).as_posix()
    col.delete(where={"path": rel})

    if not p.exists():
        return

    text = p.read_text(encoding="utf-8", errors="replace")
    ids, docs, metas = [], [], []

    for s, e, chuck in _chunks(text):
        ids.append(f"{rel}:{s}")
        docs.append(f"File: {rel}\n{chuck}")
        metas.append({"path": rel, "start": s, "end": e, "mtime": p.stat().st_mtime})

    for i in range(0, len(ids), 100):
        col.upsert(ids=ids[i:i+100], documents=docs[i:i+100], metadatas=metas[i:i+100])

def build_index() -> str:
    col = _get_collection()
    existing = {m["path"]: m["mtime"] for m in col.get(include=["metadatas"])["metadatas"]}
    seen, changed = set(), 0
    for p in _source_files():
        rel = p.relative_to(ROOT).as_posix()
        seen.add(rel)
        if existing.get(rel) != p.stat().st_mtime:   # new or modified since last run
            index_file(p)
            changed += 1
    for rel in set(existing) - seen:                  # file was deleted
        col.delete(where={"path": rel})
    return f"index ready: {changed} files updated, {len(seen)} total"

def set_root(path: str):
    global ROOT, _collection
    ROOT = Path(path).resolve()
    _collection = None

def tool(fn):
    TOOLS[fn.__name__] = fn
    return fn

def safe_path(p: str) -> Path:
    full = (ROOT / p).resolve()
    if not full.is_relative_to(ROOT):
        return ValueError("path is outside the project folder")
    return full

@tool
def semantic_search(query: str, n: int = 5) -> str:
    """Use this when you need to find code or text by meaning use this tool over reading file when you can, e.g. 'where are users authenticated'. For exact names use search_code."""
    col = _get_collection()
    if col.count() == 0:
        return "index is empty"

    res = col.query(query_texts=[query], n_results=min(int(n), 10))
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        snippet = doc.split("\n", 1)[1][:600]
        out.append(f"{meta['path']} lines {meta['start']}-{meta['end']} (score {1 - dist:.2f})\n{snippet}")

    return "\n---\n".join(out)



@tool
def list_files(path: str = ".") -> str:
    """List files and folders in a directory."""
    names = []
    for entry in safe_path(path).iterdir():
        names.append(entry.name + ("/" if entry.is_dir() else ""))
    return "\n".join(sorted(names)) or "(empty directory)"

@tool
def read_files(path: str) -> str:
    """Read a text file."""
    return safe_path(path).read_text(encoding="utf-8", errors="replace")

@tool
def git_log(n: int = 5) -> str:
    """Show the last n commits."""
    return subprocess.run(["git", "log", f"-{n}", "--oneline"], cwd=ROOT, capture_output=True, text=True).stdout


@tool
def write_file(path: str, content: str) -> str:
    """Write a file. The user is asked to confirm first."""
    if input(f"Write {path}? [y/N]").lower() != "y":
        return "user declined"

    safe_path(path).write_text(content, encoding="utf-8")
    index_file(safe_path(path)) 
    return "written"

@tool
def create_folder(path: str) -> str:
    """Create a new folder (and any missing parent folders)."""
    full = safe_path(path)
    if full.exists():
        return f"error: '{path}' already exists"
    if input(f"Create folder {path}? [y/N] ").lower() != "y":
        return "user declined"
    full.mkdir(parents=True)
    return f"folder created: {path}"

@tool
def create_file(path: str, content: str = "") -> str:
    """Create a new file with optional content. Fails if the file already exists."""

    full = safe_path(path)

    if full.exists():
        return f"error: '{path}' already exists. Use write_file to overwrite it."

    if input(f"Create file '{path}' [y/N]").lower() != "y":
        return "user declined"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    index_file(safe_path(path)) 
    return f"file created: {path} ({len(content)} characters)"

def describe_tools() -> str:
    lines = []
    for name, fn in TOOLS.items():
        params = ", ".join(str(p) for p in inspect.signature(fn).parameters.values())
        lines.append(f"- {name}({params}): {fn.__doc__}")
    return "\n".join(lines)

def run_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"error: unknown tool '{name}'. Available: {list(TOOLS)}"

    try:
        result = str(TOOLS[name](**args))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
    return result[:4000] + ("\n...[truncated]" if len(result) > 4000 else "")