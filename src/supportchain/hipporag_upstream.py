"""Import the downloaded upstream implementation without importing its LLM stack.

No upstream source is copied or patched. A namespace package points directly at
the pinned checkout; the small gold function is compiled from its original AST.
"""
import ast
import importlib
from pathlib import Path
import sys
import types


def module(root: Path, name: str):
    source = root / 'external/HippoRAG/src/hipporag'
    namespace = '_supportchain_hipporag_upstream'
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(source)]
        sys.modules[namespace] = package
    return importlib.import_module(namespace + '.' + name)


def gold_function(root: Path):
    path = root / 'external/HippoRAG/main.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_gold_docs')
    program = ast.Module(body=[function], type_ignores=[])
    namespace = {'List': list}
    exec(compile(program, str(path), 'exec'), namespace)
    return namespace['get_gold_docs']


def score_normalizer(root: Path):
    import numpy as np
    path = root / 'external/HippoRAG/src/hipporag/utils/misc_utils.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'min_max_normalize')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['min_max_normalize']


def query_instruction(root: Path):
    return module(root, 'prompts.linking').get_query_instruction('query_to_passage')


def recall_metric(root: Path):
    return module(root, 'evaluation.retrieval_eval').RetrievalRecall()
