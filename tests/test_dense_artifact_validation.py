import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from supportchain import baseline_dense
from supportchain.baseline_dense import DenseRetriever
from supportchain.llm import file_sha256


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    dataset = "musique"
    source = tmp_path / "data/splits/seed42" / dataset
    source.mkdir(parents=True)
    documents = [{"title": "A", "text": "First text."}, {"title": "B", "text": "Second text."}]
    (source / "dev_corpus.json").write_text(json.dumps(documents))
    (source / "dev-A.json").write_text(json.dumps([{"id": str(i), "question": "Q"} for i in range(500)]))
    examples = [SimpleNamespace(id=str(i), question="Q") for i in range(500)]
    monkeypatch.setattr(baseline_dense, "dev_examples", lambda *args, **kwargs: examples)
    monkeypatch.setattr(baseline_dense, "query_instruction", lambda root: "Expected query instruction.")
    monkeypatch.setattr(baseline_dense, "score_normalizer", lambda root: lambda values: values)
    folder = tmp_path / "runs/dev/embeddings" / dataset
    folder.mkdir(parents=True)
    passage_vectors = np.zeros((2, 4096), dtype=np.float16)
    passage_vectors[0, 0], passage_vectors[1, 1] = 1, 1
    query_vectors = np.zeros((500, 4096), dtype=np.float16)
    query_vectors[:, 0] = 1
    np.save(folder / "passages.f16.npy", passage_vectors)
    np.save(folder / "queries_dev-A.f16.npy", query_vectors)
    metadata = {
        "dataset": dataset, "status": "complete", "dtype": "float16", "dimension": 4096, "model_revision": "r1",
        "settings": {"provider": "HippoRAG NVEmbedV2EmbeddingModel", "max_length": 2048, "normalize": True,
                     "dimension": 4096, "model_dtype": "torch.float16",
                     "passage_instruction": "", "query_instruction": "Instruct: Expected query instruction.\nQuery: "},
        "sources": {key: {"path": str((source / filename).relative_to(tmp_path)),
                          "sha256": file_sha256(source / filename)} for key, filename in
                    [("corpus", "dev_corpus.json"), ("queries", "dev-A.json")]},
        "passage_ids": [hashlib.sha256((row["title"] + "\n" + row["text"]).encode()).hexdigest() for row in documents],
        "query_ids": [str(i) for i in range(500)], "passages_file": "passages.f16.npy", "queries_file": "queries_dev-A.f16.npy",
        "files_sha256": {name: file_sha256(folder / name) for name in ["passages.f16.npy", "queries_dev-A.f16.npy"]},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.json").write_text(json.dumps([{"repo_id": "nvidia/NV-Embed-v2", "revision": "r1"}]))
    return tmp_path, folder, metadata


def test_validated_artifacts_load_and_known_query_needs_no_http(artifacts):
    root, _, _ = artifacts
    retriever = DenseRetriever(root, "musique")
    assert retriever.search("Q", 1, question_id="0")[0].content == "A\nFirst text."


@pytest.mark.parametrize("name", ["passages.f16.npy", "queries_dev-A.f16.npy"])
def test_same_shape_array_corruption_is_rejected(artifacts, name):
    root, folder, _ = artifacts
    vector = np.load(folder / name)
    vector[0, 0] = 0.5
    np.save(folder / name, vector)
    with pytest.raises(ValueError, match="array hash mismatch"):
        DenseRetriever(root, "musique")


@pytest.mark.parametrize("kind", ["passage_ids", "query_ids"])
def test_array_order_identity_is_checked(artifacts, kind):
    root, folder, metadata = artifacts
    metadata[kind] = list(reversed(metadata[kind]))
    (folder / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="order mismatch"):
        DenseRetriever(root, "musique")


@pytest.mark.parametrize("field,value,match", [("status", "encoding", "not complete"),
                                               ("model_revision", "r2", "revision mismatch"),
                                               ("dataset", "hotpotqa", "dataset"),
                                               ("dimension", 1024, "dimension")])
def test_incomplete_or_wrong_artifact_identity_is_rejected(artifacts, field, value, match):
    root, folder, metadata = artifacts
    metadata[field] = value
    (folder / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        DenseRetriever(root, "musique")
