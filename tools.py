import inspect, subprocess
from pathlib import Path

ROOT = Path(".").resolve()
TOOLS = {}

def set_root(path: str):
    global ROOT
    ROOT = Path(path).resolve()

def tool(fn):
    TOOLS[fn.__name__] = fn
    return fn

def safe_path(p: str) -> Path:
    full = (ROOT / p).resolve()
    if not full.is_relative_to(ROOT):
        return ValueError("path is outside the project folder")
    return full

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