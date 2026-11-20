#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/hg_reasoning_agent
mkdir -p "$ROOT"/{envs,configs/env,runs/p0/setup,reports/p0}
export UV_NO_CACHE=1
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_INSTALLS=2
uv venv --allow-existing --python /venv/main/bin/python --system-site-packages "$ROOT/envs/main"
cat > "$ROOT/configs/env/main.requirements.in" <<'EOF'
gliner==0.2.29
transformers==4.57.6
accelerate
pyarrow
pytest
fastapi
uvicorn
httpx
jsonschema
sentencepiece==0.2.1
safetensors
protobuf==5.29.6
requests
psutil
EOF
"$ROOT/envs/main/bin/python" -m pip install --no-cache-dir -r "$ROOT/configs/env/main.requirements.in"
"$ROOT/envs/main/bin/python" -m pip freeze --all > "$ROOT/configs/env/main.lock"
"$ROOT/envs/main/bin/python" - <<'PY'
import sys, torch, transformers, gliner, pyarrow
import json
from pathlib import Path
record = {'environment': 'main', 'python': sys.version, 'torch': torch.__version__,
          'transformers': transformers.__version__, 'pyarrow': pyarrow.__version__,
          'base_prefix': sys.base_prefix, 'system_site_packages': True,
          'note': 'Shares the preinstalled base PyTorch stack without modifying it; project packages are isolated in envs/main.'}
Path('/workspace/hg_reasoning_agent/reports/p0/main_environment.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
PY
uv venv --python /venv/main/bin/python "$ROOT/envs/serve"
printf '%s\n' 'vllm==0.17.1' > "$ROOT/configs/env/serve.requirements.in"
uv pip install --python "$ROOT/envs/serve/bin/python" --only-binary=:all: -r "$ROOT/configs/env/serve.requirements.in"
uv pip freeze --python "$ROOT/envs/serve/bin/python" > "$ROOT/configs/env/serve.lock"
"$ROOT/envs/serve/bin/python" - <<'PY'
import sys, torch, transformers, vllm
import json
from pathlib import Path
record = {'environment':'serve', 'python': sys.version, 'torch':torch.__version__,
          'transformers':transformers.__version__, 'vllm':vllm.__version__,
          'install':'released binary wheel only; not built from external/vllm'}
Path('/workspace/hg_reasoning_agent/reports/p0/serve_environment.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
PY
df -h "$ROOT"
