#!/usr/bin/env python
"""Run one command with its own model services: start them, wait until they answer, run, stop them.

For queue jobs that need the model server in another setting than the queue's (job field services: off, so that the
queue has stopped its own services first). The only such case is the HippoRAG 2 baseline: its JSON-object requests were
validated on 2026-11-16 with the server's default structured-output setting, while the chain agent needs the compact
setting introduced on 2026-11-17, under which those requests come back malformed (see DEVIATIONS.md, 2026-11-18).

Usage: with_services.py [--llm compact|default-json] [--embed] -- COMMAND [ARG ...]
  --llm default-json   start scripts/serve_local_llm.sh without its structured-outputs setting
  --embed              also start the NV-Embed-v2 embedding service on port 18081
The exit code is the command's. Logs: runs/queue/logs/with_services.{vllm,embed}.log.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LLM, EMBED = 'http://127.0.0.1:18080/v1/models', 'http://127.0.0.1:18081/health'


def answers(url, ready=lambda payload: True):
    try:
        with urlopen(url, timeout=10) as response:
            return ready(json.load(response))
    except Exception:
        return False


def gpu_used_mib():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0])
    except Exception:
        return 0


def stop(processes):
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 90
    while time.time() < deadline and (any(process.poll() is None for process in processes) or gpu_used_mib() > 3000):
        time.sleep(3)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', default='compact', choices=['compact', 'default-json'])
    parser.add_argument('--embed', action='store_true')
    parser.add_argument('--start-minutes', type=float, default=20)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ['--'] else arguments.command
    if not command:
        parser.error('no command given')
    if answers(LLM) or answers(EMBED):
        raise SystemExit('model services are already running: this job needs them stopped (queue field services: off)')
    logs = ROOT / 'runs/queue/logs'
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        environment = dict(os.environ, HYPERBIND_STRUCTURED_DEFAULT='1' if arguments.llm == 'default-json' else '0')
        with (logs / 'with_services.vllm.log').open('ab') as stream:
            processes.append(subprocess.Popen(['bash', str(ROOT / 'scripts/serve_local_llm.sh')], cwd=ROOT, env=environment, stdout=stream,
                                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True))
        if arguments.embed:
            environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), TOKENIZERS_PARALLELISM='false',
                               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (logs / 'with_services.embed.log').open('ab') as stream:
                processes.append(subprocess.Popen(
                    [str(ROOT / 'envs/embed/bin/python'), '-u', '-m', 'hyperbind.embedding_service', '--port', '18081',
                     '--batch-size', '1', '--provider', 'nv'], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True))
        deadline = time.time() + arguments.start_minutes * 60
        while not (answers(LLM, lambda payload: bool(payload.get('data')))
                   and (not arguments.embed or answers(EMBED, lambda payload: payload.get('status') == 'ready'))):
            if any(process.poll() is not None for process in processes):
                raise SystemExit('a model service exited during start-up (see runs/queue/logs/with_services.*.log)')
            if time.time() > deadline:
                raise SystemExit('model services did not answer within %.0f minutes' % arguments.start_minutes)
            time.sleep(5)
        print(json.dumps({'services': 'ready', 'llm': arguments.llm, 'embed': arguments.embed}), flush=True)
        code = subprocess.run(command, cwd=ROOT).returncode
    finally:
        stop(processes)
        print(json.dumps({'services': 'stopped', 'gpu_used_mib': gpu_used_mib()}), flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
