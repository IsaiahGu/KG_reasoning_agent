#!/usr/bin/env python3
"""Print the queue state of the named jobs. Exit code 0 when the last one is done or any of them failed or is blocked,
1 otherwise; so `until scripts/queue_state.py a b c; do sleep 60; done` waits for a group of jobs."""
import json
import sys
from pathlib import Path

state = json.loads((Path(__file__).resolve().parents[1] / 'runs/queue/state.json').read_text())['jobs']
names = sys.argv[1:]
values = [state.get(name, {}).get('status') for name in names]
print(dict(zip(names, values)))
raise SystemExit(0 if values[-1] == 'done' or any(value in ('failed', 'blocked') for value in values) else 1)
