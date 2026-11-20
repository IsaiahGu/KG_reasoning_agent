"""Thin operational adapter around HippoRAG's downloaded NVEmbedV2 wrapper."""
from pathlib import Path
import json
import os
import subprocess
import threading
import time

from .hipporag_upstream import module, query_instruction


class MemoryMonitor:
    def __init__(self):
        self.peak_mib = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, check=True, timeout=5)
                self.peak_mib = max(self.peak_mib, int(result.stdout.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop_event.wait(.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=6)


def configure(root: Path):
    os.environ['HF_HOME'] = str(root / 'preparation/hf_cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class NVEncoder:
    def __init__(self, root: Path, batch_size: int = 4, max_length: int = 2048):
        configure(root)
        import torch
        self.root = root
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = query_instruction(root)
        instruction_builder = module(root, 'prompts.linking').get_query_instruction
        self.allowed_instructions = {instruction_builder(name) for name in
                                     ('ner_to_node', 'query_to_node', 'query_to_fact',
                                      'query_to_sentence', 'query_to_passage')}
        config_class = module(root, 'utils.config_utils').BaseConfig
        config = config_class(embedding_model_name=str(root / 'models/nvidia/NV-Embed-v2'),
                              embedding_batch_size=batch_size, embedding_max_seq_len=max_length,
                              embedding_return_as_normalized=True, embedding_model_dtype='auto')
        wrapper = module(root, 'embedding_model.NVEmbedV2').NVEmbedV2EmbeddingModel
        self.wrapper = wrapper(global_config=config)
        self.wrapper.embedding_model.eval()
        self.dimension = self.wrapper.embedding_dim
        self.model_dtype = str(next(self.wrapper.embedding_model.parameters()).dtype)
        self.settings = {'provider': 'HippoRAG NVEmbedV2EmbeddingModel', 'max_length': max_length,
                         'batch_size': batch_size, 'normalize': True, 'model_dtype': self.model_dtype,
                         'query_instruction': 'Instruct: ' + self.instruction + '\nQuery: ',
                         'passage_instruction': '', 'dimension': self.dimension}

    def encode(self, texts: list[str], kind: str, instruction: str | None = None):
        import numpy as np
        import torch
        if kind not in ('query', 'passage'):
            raise ValueError('kind must be query or passage')
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError('texts must be a nonempty list of strings')
        selected_instruction = self.instruction if kind == 'query' else ''
        if instruction is not None:
            if kind == 'passage' and instruction != '':
                raise ValueError('passage embeddings require an empty instruction')
            if kind == 'query' and instruction not in self.allowed_instructions:
                raise ValueError('query instruction must be one of the upstream allowed values')
            selected_instruction = instruction
        with torch.inference_mode():
            vectors = self.wrapper.batch_encode(texts, instruction=selected_instruction,
                                                 batch_size=self.batch_size, max_length=self.max_length)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
            raise RuntimeError('Invalid embeddings')
        return vectors


def smoke(root: Path, batch_size: int = 2):
    configure(root)
    import numpy as np
    import torch
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with MemoryMonitor() as monitor:
        encoder = NVEncoder(root, batch_size=batch_size)
        vectors = encoder.encode(['The Eiffel Tower is in Paris.', 'Paris is the capital of France.'], 'passage')
        torch.cuda.synchronize()
        norms = np.linalg.norm(vectors, axis=1)
        assert vectors.shape == (2, 4096)
        assert np.allclose(norms, 1, atol=1e-3)
        report = {'status': 'passed', 'model': 'nvidia/NV-Embed-v2', 'shape': list(vectors.shape),
                  'norms': norms.tolist(), 'cosine_similarity': float(vectors[0] @ vectors[1]),
                  'settings': encoder.settings, 'seconds': time.time() - started,
                  'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    report['nvidia_smi_peak_mib'] = monitor.peak_mib
    path = root / 'runs/p0/smoke/nvembed.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/hg_reasoning_agent'))
    parser.add_argument('--smoke', action='store_true', required=True)
    args = parser.parse_args()
    smoke(args.root)
