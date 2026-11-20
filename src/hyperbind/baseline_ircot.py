"""B3: our dense IRCoT-style loop with the downloaded official reasoning prompts."""
from __future__ import annotations

import argparse
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np

from .baseline_dense import (DATASETS, READER_GENERATION, DenseRetriever, dataset_run, official_messages,
                             read_answer, retrieval_scores)
from .llm import LocalLLMClient, client_for_run, file_sha256, save_json


@lru_cache(maxsize=4)
def _official_reason_step(root: Path):
    path = root / "external/HippoRAG/src/hipporag/utils/qa_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"merge_elements_with_same_first_line", "reason_step"}]
    if len(functions) != 2:
        raise ValueError("Upstream IRCoT reasoning helpers changed")
    namespace = {"PromptTemplateManager": object, "BaseLLM": object}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["reason_step"]


def reason_step(root: Path, client: LocalLLMClient, example, passages, thoughts, step: int):
    recorded = []
    class Prompts:
        def render(self, name, **kwargs):
            return official_messages(root, name, kwargs["prompt_user"])
    class LLM:
        def infer(self, messages):
            result = client.complete(messages, generation_overrides={"max_tokens": 512}, metadata={"phase": "P0.3", "system": "B3",
                                     "dataset": example.dataset, "question_id": example.id, "step": step})
            recorded.append(result)
            return result.content, result.usage, result.cache_hit
    thought = _official_reason_step(root)(example.dataset, Prompts(), example.question, passages, thoughts, LLM())
    return thought, recorded[0]


def predict_ircot(root: Path, example, client: LocalLLMClient, retriever, *, max_steps: int = 4) -> dict:
    if not 1 <= max_steps <= 4:
        raise ValueError("B3 is capped at four retrieval rounds")
    first = retriever.search(example.question, 5, question_id=example.id)
    merged = {document.index: document for document in first}
    thoughts, trace = [], []
    retrieval_calls = 1
    for step in range(1, max_steps):
        ranked = sorted(merged.values(), key=lambda document: document.score, reverse=True)
        thought, result = reason_step(root, client, example, [document.content for document in ranked[:5]], thoughts, step)
        thoughts.append(thought)
        event = {"step": step, "thought": thought, "llm_cache_key": result.cache_key,
                 "usage": dict(result.usage), "terminal_answer_marker": "So the answer is:" in thought}
        if event["terminal_answer_marker"]:
            trace.append(event)
            break
        following = retriever.search(thought, 5)
        retrieval_calls += 1
        event["retrieved_indices"] = [document.index for document in following]
        for document in following:
            if document.index not in merged or document.score > merged[document.index].score:
                merged[document.index] = document
        trace.append(event)
    final = sorted(merged.values(), key=lambda document: document.score, reverse=True)[:5]
    passages = [document.content for document in final]
    return {**read_answer(root, client, example, passages, system="B3"),
            "top5_indices": [document.index for document in final], "top5_scores": [document.score for document in final],
            "thoughts": thoughts, "trace": trace, "retrieval_calls": retrieval_calls,
            **retrieval_scores(example, passages)}


def run_b3(root: Path, dataset: str, *, base_url="http://127.0.0.1:18080/v1", model=None,
           workers=4, limit=None, output_root: Path | None = None, generation: dict | None = None,
           max_steps=4) -> dict:
    output = (output_root or root / "runs/p0_dev/B3") / dataset
    client = client_for_run(root, output, base_url=base_url, model=model, generation={**READER_GENERATION, **(generation or {})})
    retriever = DenseRetriever(root, dataset)
    return dataset_run(root, dataset, output, client, retriever,
                       lambda example: predict_ircot(root, example, client, retriever, max_steps=max_steps),
                       system="B3", workers=workers, limit=limit,
                       config={"max_retrieval_steps": max_steps, "per_step_top_k": 5,
                               "merge": "max minmax-normalized dense score per document",
                               "implementation": "dense IRCoT-style loop, not HippoRAG retrieval",
                               "reasoning_helper": "official qa_utils.reason_step AST, unmodified",
                               "reasoning_helper_sha256": file_sha256(root / "external/HippoRAG/src/hipporag/utils/qa_utils.py"),
                               "reasoning_template_sha256": file_sha256(root / "external/HippoRAG/src/hipporag/prompts/templates" / ("ircot_" + dataset + ".py")),
                               "loop_implementation_sha256": file_sha256(Path(__file__)),
                               "reader": "official rag_qa_musique", "thought_max_tokens": 512, "thinking": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/hg_reasoning_agent"))
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    results = [run_b3(args.root, dataset, base_url=args.base_url, model=args.model, workers=args.workers,
                      limit=args.limit, output_root=args.output_root, max_steps=args.max_steps)
               for dataset in (DATASETS if args.dataset == "all" else [args.dataset])]
    save_json(args.root / "reports/p0_b3.json", {"datasets": results,
              "macro_f1_percent": float(np.mean([row["f1_percent"] for row in results]))})


if __name__ == "__main__":
    main()
