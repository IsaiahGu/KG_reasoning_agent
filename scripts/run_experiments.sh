#!/usr/bin/env bash
# Every experiment of the paper, in order. Usage: scripts/run_experiments.sh <stage>
#   stages: prepare | fit | dev | test | baselines | variants | evaluate
# Each chain run is resumable: scripts/run_chain.py skips questions that are already in its records.jsonl.
#
# Services. The chain needs the reader on 127.0.0.1:18080 (scripts/serve_local_llm.sh) and the embedding service on
# 127.0.0.1:18081 (python -m supportchain.embedding_service). The offline encoding in `prepare` needs the GPU without the
# reader. The HippoRAG 2 baseline runs with the reader started as SUPPORTCHAIN_STRUCTURED_DEFAULT=1 scripts/serve_local_llm.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-envs/main/bin/python}
EMBED_PY=${EMBED_PY:-envs/embed/bin/python}
export PYTHONPATH=src
MULTIHOP="musique 2wikimultihopqa hotpotqa"
ALL="$MULTIHOP nq_rear popqa"
ONE="--candidates 1"                      # the method reads one quoted answer per hop
FIT="--offset 0 --limit 1500"             # train-S positions 0-1500 fit the reliability model and fill the memories
chain() { $PY -B scripts/run_chain.py --workers 12 "$@"; }
memory() { echo "configs/plan_memory/$1_round$2.json"; }
source_of() { case "$1" in nq_rear|popqa) echo hotpotqa ;; *) echo "$1" ;; esac; }  # NQ and PopQA reuse HotpotQA's

prepare() {
  # Splits of the training sets (dev-A, dev-B, train-S; no overlap with the test subsets), then NV-Embed-v2 vectors.
  $PY -B -m supportchain.splits --root .
  $EMBED_PY -m supportchain.prepare_dev_embeddings
  for d in $MULTIHOP; do $EMBED_PY -m supportchain.prepare_split_embeddings --dataset "$d" --split train-S; done
  # Test vectors through the embedding service: the three multi-hop subsets, then NQ and PopQA.
  $EMBED_PY -m supportchain.calibrate_dense
  for d in nq_rear popqa; do $PY -B scripts/prepare_test_embeddings.py --dataset "$d"; done
}

fit() {
  # Reliability model (Section 3.3) and the two rounds of plan memory (Section 3.5), from train-S answers only.
  for d in $MULTIHOP; do
    chain --dataset "$d" --split train-S --tag single_fit $FIT $ONE
    chain --dataset "$d" --split train-S --tag a0_fit $FIT --mode dense
  done
  chain --dataset musique --split train-S --tag cand3_fit_reader $FIT     # MuSiQue also fits on 3-candidate chains
  $PY -B scripts/fit_support_model.py --dataset musique --tags single_fit cand3_fit_reader --out configs/support/musique_chain_v1.json
  for d in 2wikimultihopqa hotpotqa; do
    $PY -B scripts/fit_support_model.py --dataset "$d" --tags single_fit --out "configs/support/${d}_chain_v1.json"
  done
  for d in $MULTIHOP; do
    $PY -B scripts/build_plan_memory.py --dataset "$d" --admit answer --tags single_fit --out "$(memory "$d" 1)"
    # Round 2: the training questions are planned again with the round-1 memory (each skips its own plan).
    chain --dataset "$d" --split train-S --tag selfmem_answer_fit $FIT $ONE --plan knn --plan-cases "$(memory "$d" 1)" --plan-loo
    $PY -B scripts/build_plan_memory.py --dataset "$d" --admit answer --tags selfmem_answer_fit single_fit --out "$(memory "$d" 2)"
  done
}

dev() {
  # Development set dev-A (Appendix A): the executions of the full method, the reader-picks control and the baselines.
  for d in $MULTIHOP; do
    chain --dataset "$d" --split dev-A --tag a0 --mode dense
    chain --dataset "$d" --split dev-A --tag single $ONE
    chain --dataset "$d" --split dev-A --tag selfmem_answer $ONE --plan knn --plan-cases "$(memory "$d" 1)"
    chain --dataset "$d" --split dev-A --tag selfmem2_answer $ONE --plan knn --plan-cases "$(memory "$d" 2)"
    $PY -B scripts/run_llm_selector.py --dataset "$d" --split dev-A
  done
  $PY -B -m supportchain.baseline_ircot --dataset all
  for d in $MULTIHOP; do
    for phase in openie encode-index build run; do
      $EMBED_PY -u -m supportchain.baseline_hipporag --split dev-A --workers 16 --phase "$phase" --dataset "$d"
    done
  done
}

test_runs() {
  # The executions of the full method, then the resampled executions (Section 6.2 and Table 2).
  for d in $ALL; do
    s=$(source_of "$d")
    plans="--plan knn --plan-dataset $s"
    chain --dataset "$d" --split test --tag a0 --mode dense
    chain --dataset "$d" --split test --tag single $ONE
    chain --dataset "$d" --split test --tag selfmem_answer $ONE $plans --plan-cases "$(memory "$s" 1)"
    chain --dataset "$d" --split test --tag selfmem2_answer $ONE $plans --plan-cases "$(memory "$s" 2)"
    for k in 1 2; do chain --dataset "$d" --split test --tag "sc07_$k" $ONE --temperature 0.7 --seed "$k"; done
    $PY -B scripts/run_llm_selector.py --dataset "$d" --split test        # control: the reader picks the answer
  done
  for d in $MULTIHOP; do
    chain --dataset "$d" --split test --tag sc07_3 $ONE --temperature 0.7 --seed 3
    for k in 1 2; do chain --dataset "$d" --split test --tag "sc03_$k" $ONE --temperature 0.3 --seed "$k"; done
  done
}

baselines() {
  # Same reader and embedder: IRCoT on the multi-hop sets, HippoRAG 2 from its official code on all five.
  # Start the reader with SUPPORTCHAIN_STRUCTURED_DEFAULT=1 for HippoRAG 2. The encode-index phase needs the GPU
  # without the reader; the build phase runs on CPU.
  for d in $MULTIHOP; do $PY -B scripts/run_ircot_test.py --dataset "$d"; done
  for d in $ALL; do
    for phase in openie encode-index build run; do
      $EMBED_PY -u -m supportchain.baseline_hipporag --split test --workers 16 --phase "$phase" --dataset "$d"
    done
    $PY -B scripts/score_hipporag_test.py --dataset "$d"
  done
}

variants() {
  # Chain design (Table 5): one change to the single execution at a time.
  for d in $MULTIHOP; do
    chain --dataset "$d" --split test --tag ctx_notes $ONE --reader-context notes
    chain --dataset "$d" --split test --tag ctx_passages $ONE --reader-context passages
    chain --dataset "$d" --split test --tag khop4 $ONE --k-hop 4
    chain --dataset "$d" --split test --tag cand3
  done
  # MuSiQue only: plan examples from gold decompositions, and the learned hop scorer with backtracking. The scorer is
  # fitted on 3-candidate train-S chains (tags cand3_fit, cand3_val) by scripts/fit_binding_scorer.py.
  chain --dataset musique --split test --tag knn_single $ONE --plan knn
  chain --dataset musique --split test --tag cand3_s1500_bt --selector scorer --scorer configs/scorers/musique_l2_1500.json --backtrack
}

evaluate() {
  $PY -B scripts/evaluate.py --split test --refit
  $PY -B scripts/evaluate.py --split dev-A --datasets $MULTIHOP
}

case "${1:-}" in
  prepare) prepare ;;
  fit) fit ;;
  dev) dev ;;
  test) test_runs ;;
  baselines) baselines ;;
  variants) variants ;;
  evaluate) evaluate ;;
  *) echo "usage: $0 prepare|fit|dev|test|baselines|variants|evaluate" >&2; exit 2 ;;
esac
