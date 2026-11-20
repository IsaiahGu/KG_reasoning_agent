#!/usr/bin/env bash
# After the experiments: remove model weights, the serving environments, rebuildable vector caches and unused data.
# The list was confirmed by the user on 2026-11-19. Everything else stays: code, configs, paper/, every run's records,
# the fitted reliability models, hop scorer and plan memories, the LLM call cache, embeddings, test subsets and splits,
# external/HippoRAG (answer scoring and the reader template load code from it) and envs/main.
#
#   cleanup_server.sh list      print every path with its size; changes nothing
#   cleanup_server.sh stage     move the paths into _to_delete/ (same disk, instant, undone by "restore")
#   cleanup_server.sh restore   move them back
#   cleanup_server.sh purge     delete _to_delete/ for good
#
# "stage" refuses to run while the queue runner or a model service is still running.
set -euo pipefail
cd "$(dirname "$0")/.."
TRASH=_to_delete

paths() {
    for path in models envs/serve envs/embed data/wikitopics data/raw data/infinitebench; do
        [ -e "$path" ] && echo "$path"
    done
    for path in external/*; do
        [ "$path" = external/HippoRAG ] || echo "$path"
    done
    # HippoRAG 2 baseline: the embedding cache and the vector stores of every test run (rebuilt from the embedder only)
    find runs/b2_test -name 'nv_cache.fp16.sqlite*' 2>/dev/null
    find runs/b2_test -path '*_embeddings/vdb_*' -type f 2>/dev/null
    return 0
}

case "${1:-list}" in
list)
    paths | while read -r path; do du -sh "$path"; done
    paths | tr '\n' '\0' | du -ch --files0-from=- | tail -1
    ;;
stage)
    if supervisorctl status hyperbind_queue 2>/dev/null | grep -q RUNNING; then
        echo "the queue runner is still running; nothing moved" >&2
        exit 1
    fi
    if pgrep -f 'vllm|embedding_service|baseline_hipporag' >/dev/null; then
        echo "a model service or a baseline run is still running; nothing moved" >&2
        exit 1
    fi
    mkdir -p "$TRASH"
    paths | while read -r path; do
        mkdir -p "$TRASH/$(dirname "$path")"
        mv "$path" "$TRASH/$path"
        echo "$path" >> "$TRASH/MOVED.txt"
    done
    du -sh "$TRASH"
    ;;
restore)
    while read -r path; do
        [ -e "$TRASH/$path" ] && mkdir -p "$(dirname "$path")" && mv "$TRASH/$path" "$path"
    done < "$TRASH/MOVED.txt"
    rm -f "$TRASH/MOVED.txt"
    find "$TRASH" -type d -empty -delete
    ;;
purge)
    [ -d "$TRASH" ] || { echo "nothing staged" >&2; exit 1; }
    rm -rf "$TRASH"
    df -h . | tail -1
    ;;
*)
    echo "usage: $0 list|stage|restore|purge" >&2
    exit 2
    ;;
esac
