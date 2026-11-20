"""Run exactly P0.2 once, with managed embedding service and GPU telemetry."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.request import urlopen


def run(root: Path):
    output = root / 'runs/p0/logs'
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    samples = []

    def monitor():
        with (output / 'calibration_gpu.jsonl').open('w') as log:
            while not stop.is_set():
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'], capture_output=True,
                                            text=True, check=True, timeout=5)
                    memory, utilization = map(int, result.stdout.strip().splitlines()[0].split(','))
                    sample = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              'memory_mib': memory, 'gpu_utilization_percent': utilization}
                    samples.append(sample)
                    log.write(json.dumps(sample) + '\n')
                    log.flush()
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                stop.wait(.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    status = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'scope': 'one fixed-setting P0.2 run; no generation or API calls', 'status': 'running'}
    exit_code = 1
    try:
        subprocess.run(['supervisorctl', 'start', 'hyperbind_embed'], check=True)
        for attempt in range(120):
            try:
                with urlopen('http://127.0.0.1:18081/health', timeout=3) as response:
                    if json.load(response)['status'] == 'ready':
                        break
            except Exception:
                if attempt == 119:
                    raise RuntimeError('Embedding service did not become ready')
                time.sleep(2)
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(root / 'src')
        # Bound CPU BLAS threads for single-query scoring; this changes no settings.
        environment['OPENBLAS_NUM_THREADS'] = '8'
        environment['OMP_NUM_THREADS'] = '8'
        environment['MKL_NUM_THREADS'] = '8'
        with (output / 'calibration.log').open('w') as log:
            result = subprocess.run([str(root / 'envs/main/bin/python'), '-u', '-m', 'hyperbind.calibrate_dense',
                                     '--root', str(root), '--batch-size', '8'], env=environment,
                                    cwd=root, stdout=log, stderr=subprocess.STDOUT)
        exit_code = result.returncode
        status['status'] = 'completed' if exit_code == 0 else 'failed'
        status['exit_code'] = exit_code
    except Exception as error:
        status['status'] = 'failed'
        status['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        subprocess.run(['supervisorctl', 'stop', 'hyperbind_embed'], check=False)
        stop.set()
        thread.join(timeout=7)
        service_log = Path('/var/log/portal/hyperbind_embed.log')
        if service_log.exists():
            shutil.copyfile(service_log, output / 'embedding_service.log')
        status.update({'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       'gpu_samples': len(samples),
                       'peak_gpu_memory_mib': max((sample['memory_mib'] for sample in samples), default=None),
                       'embedding_service_stopped': True})
        (root / 'reports/p0_calibration_resources.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status), flush=True)
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/hg_reasoning_agent'))
    args = parser.parse_args()
    raise SystemExit(run(args.root))
