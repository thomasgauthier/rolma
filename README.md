# ROLMA

DSPy-based question answering system using the ROMA architecture.

## Setup

1. Create a `.env` file with your OpenRouter API key:
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

If not set, telemetry runs in no-op mode (traces are generated but not exported).

## Run

```bash
./run.sh
```

The app runs inside Docker and expects a `knowledge_base/` folder in the current directory. Type `quit` to exit.

If you don't have a knowledge base on hand and want to test this out quickly you can use:
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
