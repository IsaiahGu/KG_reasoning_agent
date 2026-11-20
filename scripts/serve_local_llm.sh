#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/hg_reasoning_agent
export HF_HOME="$ROOT/preparation/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
if [[ -f "$ROOT/configs/serve_runtime.env" ]]; then
  source "$ROOT/configs/serve_runtime.env"
fi
profile_args=()
if [[ "${HYPERBIND_VLLM_GRAPHS:-0}" != "1" ]]; then
  profile_args+=(--enforce-eager)
fi
# Compact JSON (no whitespace between JSON tokens) is the setting of every chain run since 2026-11-17. The HippoRAG 2
# baseline was validated on 2026-11-16 with the server default and is run with it: HYPERBIND_STRUCTURED_DEFAULT=1, which
# only scripts/with_services.py --llm default-json sets.
if [[ "${HYPERBIND_STRUCTURED_DEFAULT:-0}" != "1" ]]; then
  profile_args+=(--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}')
fi
cd "$ROOT"
exec "$ROOT/envs/serve/bin/vllm" serve "$ROOT/models/Qwen/Qwen3.5-9B" \
  --served-model-name hyperbind-local \
  --host 127.0.0.1 --port 18080 \
  --language-model-only --dtype bfloat16 \
  --gpu-memory-utilization "${HYPERBIND_GPU_MEMORY_UTILIZATION:-0.52}" --max-model-len 16384 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-num-seqs "${HYPERBIND_MAX_NUM_SEQS:-4}" \
  --max-num-batched-tokens "${HYPERBIND_MAX_BATCHED_TOKENS:-2048}" \
  "${profile_args[@]}"
