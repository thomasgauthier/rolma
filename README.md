# ROLMA

DSPy-based question answering system inspired by [ROMA](https://github.com/sentient-agi/ROMA) (Recursive Open Meta-Agents).

This repo contains two versions:

| Version | Executor | How it works | Requires Deno? |
|---------|----------|--------------|----------------|
| `app.py` (ReAct) | ReAct + terminal tool | Uses shell `grep` to search KB files | No |
| `app_rlm.py` (RLM) | RLM | Loads KB as dict, uses `llm_query_batched` | Yes |

## What is RLM?

RLM (Recursive Language Model) comes from:
- **Paper**: [Recursive Language Models (arxiv.org/abs/2512.24601v1)](https://arxiv.org/abs/2512.24601v1)
- **GitHub**: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)

RLM is a general plug-and-play inference library for Recursive Language Models, supporting various sandboxes. The LLM uses a REPL to programmatically explore large contexts via `llm_query`/`llm_query_batched` instead of stuffing everything into the prompt.

This repo contains two versions:

| Version | Executor | How it works | Requires Deno? |
|---------|----------|--------------|----------------|
| `app.py` (ReAct) | ReAct + terminal tool | Uses shell `grep` to search KB files | No |
| `app_rlm.py` (RLM) | RLM | Loads KB as dict, uses `llm_query_batched` | Yes |

## Quick Start

```bash
# ReAct version (default)
./run.sh

# RLM version
./run_rlm.sh
```

## What's the difference?

### ReAct (`app.py`)
- Uses **DSPy ReAct** with a `terminal` tool
- The LLM executes shell commands (`grep`, `cat`, etc.) to search the KB
- Simpler setup, no Deno required
- Good for smaller knowledge bases

### RLM (`app_rlm.py`)
- Uses **DSPy RLM** (Retrieve-LLM-Maybe)
- KB loaded as nested Python dict via `dir_to_dict()`
- LLM uses `llm_query_batched` within a REPL to chunk and process the KB
- Better for large contexts (1M+ tokens)
- Requires Deno for the Pyodide sandbox

## Architecture

Both versions share the same core:

```
Query → Atomizer (atomic vs planned?)
       ├── Atomic → Executor (KB retrieval + LLM)
       └── Planned → Planner → Subtasks → Executor → Aggregator
```

- **Atomizer**: Classifies tasks as atomic or requiring planning
- **Planner**: Decomposes complex goals into subtasks with dependencies
- **Executor**: Retrieves info from KB and generates answer
- **Aggregator**: Synthesizes subtask results

## Running the ReAct Version

The ReAct version (`app.py`) uses a **terminal tool** (not sandboxed). The LLM can execute shell commands like `grep`, `cat`, `ls`, etc. on the host system.

Mount only `knowledge_base/` to limit filesystem access:

```bash
docker build -t rolma .
docker run -it -v "$(pwd)/knowledge_base:/app/knowledge_base" rolma
```

The RLM version (`app_rlm.py`) uses a Pyodide/WASM sandbox instead.

## Setup

1. **For RLM version only**: Install Deno (required for Pyodide/WASM sandbox):
   ```bash
   curl -fsSL https://deno.land/install.sh | sh
   ```

2. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

## Telemetry (Optional)

This app uses OpenTelemetry (OTLP) for tracing. To enable telemetry exports, set these environment variables:

```bash
# Required: OTLP endpoint (e.g., Langfuse, Grafana Tempo, Jaeger, etc.)
export OTEL_EXPORTER_OTLP_ENDPOINT=https://your-otlp-endpoint.com

# Optional: headers (e.g., for authentication)
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic your_auth"
```

If not set, telemetry is disabled entirely.

## Run

Both scripts run inside Docker and expect a `knowledge_base/` folder in the current directory.

```bash
./run.sh        # ReAct version
./run_rlm.sh    # RLM version
```

Type `quit` to exit.

## Testing without a Knowledge Base

If you don't have a knowledge base on hand:
```bash
uv run --with pyyaml,datasets python - <<'EOF'
import yaml, json, os
from datasets import load_dataset

class Folded(str): pass
yaml.add_representer(Folded, lambda d, s: d.represent_scalar('tag:yaml.org,2002:str', s, style='>'))

def fix_body(data):
    if isinstance(data, dict):
        return {k: (Folded(v) if k == 'body' and isinstance(v, str) else fix_body(v)) for k, v in data.items()}
    if isinstance(data, list):
        return [fix_body(i) for i in data]
    return data

os.makedirs("knowledge_base", exist_ok=True)
ds = load_dataset('notesbymuneeb/epstein-emails')

for row in ds["train"]:
    data = fix_body(json.loads(row["messages"]))
    
    with open(f"knowledge_base/{row['thread_id']}.yaml", "w") as f:
        yaml.dump(data, f, width=260, default_flow_style=False, allow_unicode=True)
EOF
```
