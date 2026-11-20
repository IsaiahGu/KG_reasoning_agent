#!/usr/bin/env bash
# One-off tidy-up of 2026-11-18 (user instruction: keep only the necessary code and the trained models).
# Everything removed from the tree is first packed into archive/legacy_2026-11-18.tar.gz (code, tests, configs, notes,
# set-up payloads) and, on the server, archive/legacy_runs_2026-11-18.tar.gz (small run folders of earlier phases), so
# that nothing is lost. Restore one path with:  tar -xzf archive/legacy_2026-11-18.tar.gz <path>
# Run from the project root. Usage: scripts/tidy_2026_11_18.sh [--runs] [--hypergraph]
#   --runs        also pack the small run folders of earlier phases
#   --hypergraph  also pack the hypergraph module (removed from the method on 2026-11-18, configs/final.json) into
#                 archive/hypergraph_2026-11-18.tar.gz; its large indexes are deleted separately, not packed
set -euo pipefail
RUNS=0; HYPERGRAPH=0
for flag in "$@"; do
  case $flag in --runs) RUNS=1 ;; --hypergraph) HYPERGRAPH=1 ;; *) echo "unknown flag $flag" >&2; exit 2 ;; esac
done
[[ -f configs/frozen_v2.json && -d src/hyperbind ]] || { echo 'run from the project root' >&2; exit 2; }
if [[ -f runs/queue/state.json ]] && grep -q '"running": {' runs/queue/state.json; then
  echo 'the queue is running: tidy up after it has stopped' >&2; exit 3
fi
mkdir -p archive

pack() {  # pack <tarball> <path>...: pack the paths that exist, check the tarball, then remove them
  local tarball=$1; shift
  local list; list=$(mktemp)
  for path in "$@"; do [[ -e $path ]] && echo "$path" >> "$list"; done
  if [[ ! -s $list ]]; then echo "nothing to pack for $tarball"; rm -f "$list"; return; fi
  [[ -e $tarball ]] && { echo "$tarball exists already: not overwritten" >&2; rm -f "$list"; exit 4; }
  tar -czf "$tarball" -T "$list"
  local packed; packed=$(tar -tzf "$tarball" | grep -vc '/$' || true)
  local present; present=$(xargs -a "$list" -d '\n' -I{} find {} -type f | wc -l)
  if [[ $packed -ne $present ]]; then echo "check failed for $tarball: $packed files packed, $present present" >&2; exit 5; fi
  xargs -a "$list" -d '\n' rm -rf
  echo "$tarball: $packed files from $(wc -l < "$list") paths"
  rm -f "$list"
}

modules=(baseline_hipporag baseline_ircot check_json diagnostics gates hipporag_backend hipporag_fp16_store hipporag_preflight
         memory openie_recovery p0_phase_validation p1_retrieval p2_experiments qwen_embedding r3_concurrency reasoning_agent
         reranker server_master smoke_models stage_status truncated_json)
scripts=(apply_llm_order_fix.py compare_chain.py make_queue_v1.py make_reference_scores.py prepare_qwen_dev_embeddings.py
         queue_report.py run_p0_continuation.py run_p1_retrieval.py run_p1_stage.py run_p2_stage.py run_r3_concurrency_stage.py
         run_r3_throughput_profiles.py smoke_gpu_monitor.py smoke_models.py supervised_job.py)
tests=(test_baselines test_gates test_hipporag_baseline test_hipporag_preflight test_json_check test_master_scope test_memory
       test_openie_recovery test_p1_stage test_p2_experiments test_p2_stage test_reasoning_agent test_recovery test_server_master
       test_truncated_json)
configs=(AUTOMATIC_RUN.txt automatic_chain_manifest.json baseline_B2.json embed_runtime.env embed_service.supervisor.conf
         embed_service.supervisor.sh experiment_job.supervisor.conf experiment_master.supervisor.conf job_runtime.json job_scope.json
         p0_baselines.json p0_services.conf p2.json smoke.json smoke_compatibility_sources.json)
notes=(P0_REPORT.md P4_L2_REPORT.md PREPARATION.md REVIEW_FOR_CODEX_P0.md REVIEW_FOR_CODEX_P2.md REVIEW_FOR_CODEX_PREPARATION.md)
probes=(probe/compare.py probe/comparison_dev_A.txt probe/probe_loop.py probe/select_pilot.py probe/select_pilot2.py probe/sota_check.py
        probe/statement_merge_pilot.py probe/2wikimultihopqa probe/hotpotqa probe/musique probe/import_closure.py probe/job_state.py
        probe/stop_after_nq.py)
keep_preparation='download_data.py|prepare_models.py|prepare_auxiliary_models.py|install_root_envs.sh|offline_env.sh|verify_data.py|verify_preparation.py|record_environment.py|hf_cache|xet_cache|p0_complete.tgz|p0_complete.tgz.sha256|p0_splits_snapshot.tgz'

rm -rf .pytest_cache  # caches are rebuilt on demand, they are not packed
find src scripts tests preparation -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
[[ -f archive/test_hypergraph.original.py ]] || cp tests/test_hypergraph.py archive/test_hypergraph.original.py

legacy=("${notes[@]}" "${probes[@]}" runs/chain/partial_compare.py runs/chain/finish_fit.sh)
for name in "${modules[@]}"; do legacy+=("src/hyperbind/$name.py"); done
for name in "${scripts[@]}"; do legacy+=("scripts/$name"); done
for name in "${tests[@]}"; do legacy+=("tests/$name.py"); done
for name in "${configs[@]}"; do legacy+=("configs/$name"); done
if [[ -d preparation ]]; then
  while IFS= read -r name; do legacy+=("preparation/$name"); done < <(ls -A preparation | grep -vxE "$keep_preparation|.*\.partial")
fi
pack archive/legacy_2026-11-18.tar.gz "${legacy[@]}"

# The hypergraph tests that exercised the archived reranker, Qwen embedding and rung-plan modules go with those modules.
if grep -q 'hyperbind.p1_retrieval' tests/test_hypergraph.py 2>/dev/null; then
  sed -i '158,229d;133,142d;20,21d' tests/test_hypergraph.py
  echo 'tests/test_hypergraph.py: removed the tests of the archived modules'
fi

if [[ $HYPERGRAPH == 1 ]]; then
  pack archive/hypergraph_2026-11-18.tar.gz src/hyperbind/hyper_retriever.py src/hyperbind/retrieval_operator.py \
    src/hyperbind/graph_index.py src/hyperbind/hypergraph.py scripts/build_graph_index.py scripts/build_test_graph_index.py \
    tests/test_hypergraph.py archive/test_hypergraph.original.py configs/p1_index.json configs/p1_operator.json
fi

if [[ $RUNS == 1 ]]; then
  pack archive/legacy_runs_2026-11-18.tar.gz runs/history runs/recovery runs/hotpot_cache_recovery runs/p2 runs/p2_auto runs/r3 \
    runs/r3_throughput runs/r3_concurrency runs/r1_hotpot runs/progression runs/p1_auto runs/p1_stage runs/p0_3_to_p0_5 \
    runs/b2_json_mode_v2_validation runs/p0_compact_json_recheck runs/server_master runs/supervisor_job.log runs/p0_dev/diagnostics \
    runs/p0_dev/B3 runs/p0_dev/json_check runs/p0_dev/openie_benchmark runs/p0_dev/tests runs/p0_dev/logs runs/p1/retrieval \
    runs/p1/rerank_cache runs/p1/preparation
fi
echo done
