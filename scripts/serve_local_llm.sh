#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
if [[ -f "$ROOT/configs/serve_runtime.env" ]]; then
  source "$ROOT/configs/serve_runtime.env"
fi
profile_args=()
if [[ "${SUPPORTCHAIN_VLLM_GRAPHS:-0}" != "1" ]]; then
  profile_args+=(--enforce-eager)
fi
# Compact JSON (no whitespace between JSON tokens) is the setting of every chain run. The HippoRAG 2 baseline runs with
# the server default instead: start the server with SUPPORTCHAIN_STRUCTURED_DEFAULT=1 for its extraction and answering.
if [[ "${SUPPORTCHAIN_STRUCTURED_DEFAULT:-0}" != "1" ]]; then
  profile_args+=(--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}')
fi
cd "$ROOT"
exec "${VLLM:-$ROOT/envs/serve/bin/vllm}" serve "$ROOT/models/Qwen/Qwen3.5-9B" \
  --served-model-name supportchain-local \
  --host 127.0.0.1 --port 18080 \
  --language-model-only --dtype bfloat16 \
  --gpu-memory-utilization "${SUPPORTCHAIN_GPU_MEMORY_UTILIZATION:-0.52}" --max-model-len 16384 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-num-seqs "${SUPPORTCHAIN_MAX_NUM_SEQS:-4}" \
  --max-num-batched-tokens "${SUPPORTCHAIN_MAX_BATCHED_TOKENS:-2048}" \
  "${profile_args[@]}"
