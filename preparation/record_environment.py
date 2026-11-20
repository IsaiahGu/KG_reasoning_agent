"""Record runtime provenance without importing or loading any model."""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=['main', 'embed', 'serve'], required=True)
args = parser.parse_args()
root = Path('/workspace/hg_reasoning_agent')
packages = ['torch', 'transformers', 'gliner', 'pyarrow', 'vllm', 'numpy', 'sentencepiece', 'protobuf', 'huggingface-hub', 'hipporag']
versions = {}
for package in packages:
    try:
        distribution = md.distribution(package)
        versions[package] = {'version': distribution.version, 'site_packages': str(distribution.locate_file(''))}
    except md.PackageNotFoundError:
        pass
lock = root / 'configs/env' / (args.name + '.lock')
report = {'name': args.name, 'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'executable': sys.executable, 'python': sys.version,
          'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'packages': versions,
          'lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest()}
destination = root / 'reports/p0' / (args.name + '_environment.json')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
