"""B1 on dev-A: pinned NV dense top five and the upstream QA reader template."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
from string import Template
import threading
import time

import numpy as np

from .calibrate_dense import fetch
from .data import QAExample, dataset_name, load_corpus, load_samples
from .hipporag_upstream import module as upstream_module, score_normalizer, query_instruction
from .llm import LocalLLMClient, client_for_run, digest, file_sha256, save_json, summarize_calls

DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")
READER_GENERATION = {"temperature": 0, "top_p": 1.0, "seed": 42, "max_tokens": 2048}


def dev_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    folder = root / "data/splits/seed42" / dataset_name(dataset)
    return folder / "dev-A.json", folder / "dev_corpus.json"


def dev_examples(root: Path, dataset: str, limit: int | None = None) -> list[QAExample]:
    path, _ = dev_paths(root, dataset)
    examples = load_samples(path, dataset)
    assert len(examples) == 500, (dataset, len(examples))
    if dataset_name(dataset) == "hotpotqa":
        assert all(example.raw.get("level") == "hard" for example in examples), "R1 hard-only split is required"
    return examples if limit is None else examples[:limit]


@dataclass(frozen=True)
class RetrievedDocument:
    index: int
    score: float
    content: str


class DenseRetriever:
    """CPU ranking over offline passage vectors; new queries use the NV HTTP service."""

    def __init__(self, root: Path, dataset: str, *, endpoint="http://127.0.0.1:18081"):
        self.root, self.dataset, self.endpoint = root, dataset_name(dataset), endpoint.rstrip("/")
        query_path, corpus_path = dev_paths(root, dataset)
        self.examples = dev_examples(root, dataset)
        self.documents = [document.content for document in load_corpus(corpus_path)]
        folder = root / "runs/p0_dev/embeddings" / self.dataset
        self.metadata = json.loads((folder / "metadata.json").read_text())
        if self.metadata["status"] != "complete":
            raise ValueError("Offline dev embeddings are not complete")
        if self.metadata["dataset"] != self.dataset or self.metadata["dtype"] != "float16" or self.metadata["dimension"] != 4096:
            raise ValueError("Dense artifact dataset, dtype or dimension mismatch")
        if self.metadata["settings"]["provider"] != "HippoRAG NVEmbedV2EmbeddingModel":
            raise ValueError("Dense artifact provider is not the pinned official NV wrapper")
        inventory = json.loads((root / "reports/model_inventory.json").read_text())
        expected_revision = next(row["revision"] for row in inventory if row["repo_id"] == "nvidia/NV-Embed-v2")
        if self.metadata["model_revision"] != expected_revision:
            raise ValueError("Dense artifact model revision mismatch")
        assert self.metadata["settings"]["max_length"] == 2048
        assert self.metadata["settings"]["normalize"] is True
        assert self.metadata["settings"]["passage_instruction"] == ""
        assert self.metadata["settings"]["dimension"] == 4096
        assert self.metadata["settings"]["model_dtype"] == "torch.float16"
        assert self.metadata["settings"]["query_instruction"] == "Instruct: " + query_instruction(root) + "\nQuery: "
        sources = self.metadata["sources"]
        if (root / sources["corpus"]["path"]).resolve() != corpus_path.resolve() or (root / sources["queries"]["path"]).resolve() != query_path.resolve():
            raise ValueError("Dense artifacts must point to the current dev-A and dev corpus paths")
        assert sources["corpus"]["sha256"] == file_sha256(corpus_path), "Stale dev corpus vectors"
        assert sources["queries"]["sha256"] == file_sha256(query_path), "Stale dev query vectors"
        expected_passage_ids = [hashlib.sha256(content.encode("utf-8")).hexdigest() for content in self.documents]
        if self.metadata["passage_ids"] != expected_passage_ids:
            raise ValueError("Dense artifact passage order mismatch")
        if self.metadata["query_ids"] != [example.id for example in self.examples]:
            raise ValueError("Dense artifact query order mismatch")
        if self.metadata["passages_file"] != "passages.f16.npy" or self.metadata["queries_file"] != "queries_dev-A.f16.npy":
            raise ValueError("Unexpected dense artifact array filenames")
        for filename in ("passages.f16.npy", "queries_dev-A.f16.npy"):
            if file_sha256(folder / filename) != self.metadata["files_sha256"][filename]:
                raise ValueError("Dense artifact array hash mismatch: " + filename)
        self.vectors = np.load(folder / "passages.f16.npy", mmap_mode="r")
        self.query_vectors = np.load(folder / "queries_dev-A.f16.npy", mmap_mode="r")
        assert self.vectors.dtype == self.query_vectors.dtype == np.float16
        assert self.vectors.shape == (len(self.documents), 4096)
        assert self.query_vectors.shape == (len(self.examples), 4096)
        self._ids = {example.id: index for index, example in enumerate(self.examples)}
        self._float_vectors = np.asarray(self.vectors, dtype=np.float32)
        self._normalizer = score_normalizer(root)
        self._service_checked = False
        self._service_lock = threading.Lock()
        self.fingerprint = {"metadata_sha256": file_sha256(folder / "metadata.json"), "sources": sources}

    def search(self, query: str, k: int = 5, *, question_id: str | None = None) -> list[RetrievedDocument]:
        if not 1 <= k <= len(self.documents):
            raise ValueError("Invalid dense retrieval depth")
        position = self._ids.get(question_id)
        if position is not None and self.examples[position].question == query:
            vector = self.query_vectors[position]
        else:
            import base64
            with self._service_lock:
                if not self._service_checked:
                    health = fetch(self.endpoint + "/health")
                    assert health["status"] == "ready"
                    for setting in ("max_length", "normalize", "query_instruction", "passage_instruction", "dimension", "model_dtype"):
                        assert health["settings"][setting] == self.metadata["settings"][setting], "NV online/offline settings mismatch"
                    self._service_checked = True
            response = fetch(self.endpoint + "/encode", {"texts": [query], "kind": "query"})
            assert response["dtype"] == "float16" and response["shape"] == [1, 4096]
            vector = np.frombuffer(base64.b64decode(response["data"]), dtype="<f2").reshape(1, 4096)[0]
        assert np.isfinite(vector).all()
        scores = self._normalizer(np.dot(self._float_vectors, np.asarray(vector, dtype=np.float32)))
        indices = np.argsort(scores)[::-1][:k]
        return [RetrievedDocument(int(i), float(scores[i]), self.documents[i]) for i in indices]


def official_messages(root: Path, template_name: str, prompt_user: str) -> list[dict]:
    template = upstream_module(root, "prompts.templates." + template_name).prompt_template
    return [{"role": message["role"], "content": Template(message["content"]).substitute(prompt_user=prompt_user)}
            for message in template]


def reader_messages(root: Path, question: str, passages: list[str]) -> list[dict]:
    # Exactly the upstream HippoRAG.qa prompt construction; only passages differ by system.
    prompt = "".join(f"Wikipedia Title: {passage}\n\n" for passage in passages)
    prompt += "Question: " + question + "\nThought: "
    return official_messages(root, "rag_qa_musique", prompt)


def extract_answer(content: str) -> tuple[str, bool]:
    # Upstream HippoRAG.qa uses split()[1], with the raw output as its fallback.
    try:
        return content.split("Answer:")[1].strip(), True
    except (IndexError, AttributeError):
        return content, False


def read_answer(root: Path, client: LocalLLMClient, example: QAExample, passages: list[str], *, system: str) -> dict:
    result = client.complete(reader_messages(root, example.question, passages),
                             metadata={"phase": "P0.3" if system in {"B1", "B3"} else "P0.4",
                                       "system": system, "dataset": example.dataset, "question_id": example.id})
    answer, parsed = extract_answer(result.content)
    return {"answer": answer, "answer_marker_found": parsed, "response": result.content,
            "finish_reason": result.finish_reason, "llm_cache_key": result.cache_key,
            "usage": dict(result.usage), "cache_hit": result.cache_hit}


def score_answers(root: Path, examples: list[QAExample], predictions: list[str]) -> tuple[dict, list[dict]]:
    metrics = upstream_module(root, "evaluation.qa_eval")
    gold = [list(example.answers) for example in examples]
    em, per_em = metrics.QAExactMatch().calculate_metric_scores(gold, predictions)
    f1, per_f1 = metrics.QAF1Score().calculate_metric_scores(gold, predictions)
    return {"em": float(em["ExactMatch"]), "f1": float(f1["F1"])}, [
        {"em": float(e["ExactMatch"]), "f1": float(f["F1"])} for e, f in zip(per_em, per_f1)]


def retrieval_scores(example: QAExample, passages: list[str]) -> dict:
    gold = set(example.gold_documents)
    return {"recall_at_5": len(gold.intersection(passages[:5])) / len(gold) if gold else 0,
            "all_supporting_at_5": bool(gold) and gold.issubset(set(passages[:5]))}


def dataset_run(root: Path, dataset: str, output: Path, client: LocalLLMClient, retriever: DenseRetriever,
                predictor, *, system: str, workers: int = 4, limit: int | None = None, config: dict | None = None) -> dict:
    examples = dev_examples(root, dataset, limit)
    started = time.monotonic()
    identity = {"system": system, "dataset": dataset, "split": "dev-A", "question_ids": [x.id for x in examples],
                "retrieval": retriever.fingerprint, "model": client.model_fingerprint,
                "generation": client.generation, "config": config or {},
                "source_sha256": {name: file_sha256(root / name) for name in (
                    "src/hyperbind/baseline_dense.py", "src/hyperbind/llm.py",
                    "external/HippoRAG/src/hipporag/evaluation/qa_eval.py")},
                "reader_template_sha256": file_sha256(root / "external/HippoRAG/src/hipporag/prompts/templates/rag_qa_musique.py")}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise ValueError("Run directory contains a different experiment identity")
    save_json(manifest, {"fingerprint": digest(identity), "identity": identity})

    def one(example):
        try:
            record = predictor(example)
            return {"id": example.id, "question": example.question, "status": "success", **record}
        except Exception as error:
            return {"id": example.id, "question": example.question, "status": "error", "answer": "",
                    "error_type": type(error).__name__, "error": str(error)}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one, examples):
            records.append(row)
            if len(records) % 25 == 0 or len(records) == len(examples):
                save_json(output / "progress.json", {"completed": len(records), "total": len(examples),
                                                     "errors": sum(x["status"] == "error" for x in records)})
                print(json.dumps({"system": system, "dataset": dataset, "completed": len(records), "total": len(examples)}), flush=True)
    metrics, per_question = score_answers(root, examples, [row["answer"] for row in records])
    for row, scores in zip(records, per_question):
        row.update(scores)
    save_json(output / "per_question.json", records)
    result = {"system": system, "dataset": dataset, "split": "dev-A", "questions": len(examples),
              "em_percent": metrics["em"] * 100, "f1_percent": metrics["f1"] * 100,
              "errors": sum(row["status"] == "error" for row in records),
              "answer_marker_missing": sum(row.get("answer_marker_found") is False for row in records),
              "truncated_answers": sum(row.get("finish_reason") == "length" for row in records),
              "recall_at_5_percent": 100 * sum(row.get("recall_at_5", 0) for row in records) / len(records),
              "all_supporting_at_5_percent": 100 * sum(row.get("all_supporting_at_5", False) for row in records) / len(records),
              "seconds": time.monotonic() - started, "fingerprint": digest(identity),
              "llm_calls_cumulative": summarize_calls(client.log_path),
              "metric_source": "external/HippoRAG/src/hipporag/evaluation/qa_eval.py",
              "identity": identity}
    save_json(output / "summary.json", result)
    return result


def run_b1(root: Path, dataset: str, *, base_url="http://127.0.0.1:18080/v1", model=None,
           workers=4, limit=None, output_root: Path | None = None, generation: dict | None = None) -> dict:
    output = (output_root or root / "runs/p0_dev/B1") / dataset
    client = client_for_run(root, output, base_url=base_url, model=model, generation={**READER_GENERATION, **(generation or {})})
    retriever = DenseRetriever(root, dataset)
    def predict(example):
        documents = retriever.search(example.question, 5, question_id=example.id)
        passages = [document.content for document in documents]
        return {**read_answer(root, client, example, passages, system="B1"),
                "top5_indices": [document.index for document in documents],
                "top5_scores": [document.score for document in documents],
                **retrieval_scores(example, passages)}
    return dataset_run(root, dataset, output, client, retriever, predict, system="B1", workers=workers, limit=limit,
                       config={"top_k": 5, "reader": "upstream rag_qa_musique", "thinking": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/hg_reasoning_agent"))
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    results = [run_b1(args.root, name, base_url=args.base_url, model=args.model, workers=args.workers,
                      limit=args.limit, output_root=args.output_root)
               for name in (DATASETS if args.dataset == "all" else [args.dataset])]
    save_json(args.root / "reports/p0_b1.json", {"datasets": results,
              "macro_f1_percent": float(np.mean([row["f1_percent"] for row in results]))})


if __name__ == "__main__":
    main()
