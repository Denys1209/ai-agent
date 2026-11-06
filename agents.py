import argparse, json, re, sys
import time
import grpc
import chat_pb2, chat_pb2_grpc
import tools
from tools import describe_tools, run_tool

parser = argparse.ArgumentParser(description="Local coding agent")
parser.add_argument("directory", nargs="?", default=".",
                    help="project folder the agent may read (default: current folder)")
parser.add_argument("--server", default="localhost:50051", help="gRPC server address")
parser.add_argument("--max-steps", type=int, default=8)
args = parser.parse_args()

tools.set_root(args.directory)
print("Indexing project...")
print(tools.build_index())

if not tools.ROOT.is_dir():
    sys.exit(f"Not a directory: {tools.ROOT}")

SYSTEM = f"""You are a coding assistant working inside a git repository.
Project folder: {tools.ROOT}
You can use these tools:
{describe_tools()}

Reply with ONE JSON object and nothing else.
To use a tool:  {{"tool": "read_file", "args": {{"path": "main.py"}}}}
To answer:      {{"final": "your answer to the user"}}
Use one tool per reply. After a tool call you will receive its result.
Paths are relative to the project folder."""

stub = chat_pb2_grpc.Gemma4ServerStub(grpc.insecure_channel(args.server))


def msg(role, text):
    return chat_pb2.Message(role=role, text=text)

def ask(messages):
    req = chat_pb2.Request(history=chat_pb2.History(messages=messages[:-1]),
                           userMessage=messages[-1])
    return stub.MakeRequest(req).answer

def parse(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else None

def log(event):
    with open("trace.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), **event}) + "\n")

def trim(messages, budget_chars=40000):
    total = sum(len(m.text) for m in messages)
    for m in messages[1:]:
        if (total <= budget_chars):
            break
        if m.text.startswith("TOOL RESULT:") and len(m.text) > 300:
            total -= len(m.text) - 300
            m.text = m.text[:300] + "\n...[old result trimmed]"

messages = [msg(chat_pb2.SYSTEM, SYSTEM)]

def run_agent(task, max_steps):
    messages.append(msg(chat_pb2.USER, task))
    log({"type":"task", "text":task})
    for step in range(max_steps):
        trim(messages)
        reply = ask(messages)
        log({"type":"model", "step": step, "text":reply})
        messages.append(msg(chat_pb2.ASSISTANT, reply))
        try:
            action = parse(reply)
            assert action
        except Exception:
            messages.append(msg(chat_pb2.USER, "Invalid format. Reply with one JSON object."))
            continue
        if "final" in action:
            return action["final"]
        result = run_tool(action.get("tool", ""), action.get("args", {}))
        log({"type": "tool", "step": step, "call": action, "result":result})
        print(f"[step {step}] {action} -> {result[:150]!r}")
        messages.append(msg(chat_pb2.USER, f"TOOL RESULT:\n{result}"))
    return "Stopped: step limit reached."

if __name__ == "__main__":
    print(f"Agent working in: {tools.ROOT}")
    while True:
        try:
            task = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        print(run_agent(task, args.max_steps))