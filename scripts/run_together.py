#!/usr/bin/env python
"""Run several commands at the same time; succeed only if all of them do.

For queue jobs whose parts wait on single LLM requests rather than on the GPU: the answering phase of the HippoRAG 2
baseline is a sequential loop (one request per question, twice), so the three datasets can share the model server.

Usage: run_together.py --logs DIR --done FILE NAME=COMMAND[@@RESULT] [NAME=COMMAND[@@RESULT] ...]
Each COMMAND is split like a shell line and writes to DIR/NAME.log. A part whose RESULT file already exists is skipped,
so a retry only repeats what is missing. Once a minute the log sizes are printed, which is what the queue runner watches
to tell work from a stall. FILE is written when every part has ended with exit code 0.
"""
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--done', type=Path, required=True)
    parser.add_argument('parts', nargs='+')
    arguments = parser.parse_args()
    logs = ROOT / arguments.logs
    logs.mkdir(parents=True, exist_ok=True)
    running, codes = {}, {}
    for part in arguments.parts:
        name, _, rest = part.partition('=')
        command, _, result = rest.partition('@@')
        if result and (ROOT / result).exists() and (ROOT / result).stat().st_size:
            codes[name] = 0
            print(json.dumps({'part': name, 'skipped': 'result exists'}), flush=True)
            continue
        stream = (logs / (name + '.log')).open('ab')
        running[name] = (subprocess.Popen(shlex.split(command), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT), stream)
        print(json.dumps({'part': name, 'started': command}), flush=True)
    reported = 0.0
    while running:
        for name in list(running):
            process, stream = running[name]
            if process.poll() is not None:
                stream.close()
                codes[name] = process.returncode
                del running[name]
                print(json.dumps({'part': name, 'exit': process.returncode}), flush=True)
        if running and time.time() - reported >= 60:
            reported = time.time()
            print(json.dumps({'running': {name: (logs / (name + '.log')).stat().st_size for name in running}}), flush=True)
        time.sleep(5)
    failed = sorted(name for name, code in codes.items() if code)
    if failed:
        raise SystemExit('failed parts: ' + ', '.join(failed))
    (ROOT / arguments.done).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / arguments.done).write_text(json.dumps({'parts': sorted(codes), 'finished': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}) + '\n')


if __name__ == '__main__':
    main()
