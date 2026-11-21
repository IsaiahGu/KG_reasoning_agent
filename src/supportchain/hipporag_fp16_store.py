"""Only persist upstream embedding-store rows as FP16 blobs; math stays FP32."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sqlite3
from unittest.mock import patch

import numpy as np
from hipporag.embedding_store import EmbeddingStore, _validate_embeddings


class FP16EmbeddingStore(EmbeddingStore):
    def _load_data(self):
        path = Path(self.filename + '.fp16.sqlite')
        if Path(self.filename).exists():
            raise ValueError('Use a fresh B2 index directory for FP16 persistence')
        rows = []
        if path.exists():
            with sqlite3.connect(path) as connection:
                rows = connection.execute('SELECT hash_id, content, vector FROM vectors ORDER BY position').fetchall()
        self.hash_ids = [row[0] for row in rows]
        self.texts = [row[1] for row in rows]
        self.embeddings = [np.frombuffer(row[2], dtype='<f2').astype(np.float32) for row in rows]
        if len(set(self.hash_ids)) != len(self.hash_ids):
            raise ValueError('Duplicate upstream embedding IDs')
        self.hash_id_to_idx = {key: index for index, key in enumerate(self.hash_ids)}
        self.hash_id_to_row = {key: {'hash_id': key, 'content': text} for key, text in zip(self.hash_ids, self.texts)}
        self.hash_id_to_text = dict(zip(self.hash_ids, self.texts))
        self.text_to_hash_id = dict(zip(self.texts, self.hash_ids))

    def _save_data(self, hash_ids=None, texts=None, embeddings=None):
        hash_ids = list(self.hash_ids if hash_ids is None else hash_ids)
        texts = list(self.texts if texts is None else texts)
        embeddings = list(self.embeddings if embeddings is None else embeddings)
        if len(hash_ids) != len(texts):
            raise ValueError('Inconsistent embedding-store columns')
        if embeddings:
            vectors = _validate_embeddings(embeddings, len(hash_ids)).astype('<f2')
        else:
            vectors = []
        path = Path(self.filename + '.fp16.sqlite')
        with sqlite3.connect(path) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS vectors (position INTEGER PRIMARY KEY, hash_id TEXT UNIQUE, content TEXT, vector BLOB)')
            connection.execute('DELETE FROM vectors')
            connection.executemany('INSERT INTO vectors VALUES (?,?,?,?)',
                [(index, key, text, vector.tobytes()) for index, (key, text, vector) in enumerate(zip(hash_ids, texts, vectors))])
        self._load_data()


@contextmanager
def persistence_adapter():
    """Inject storage only while constructing the unmodified upstream class."""
    upstream = importlib.import_module('hipporag.HippoRAG')

    def factory(embedding_model, db_path, batch_size, namespace, global_config=None):
        return FP16EmbeddingStore(embedding_model, db_path, batch_size, namespace)

    with patch.object(upstream, 'get_embedding_store', factory):
        yield
