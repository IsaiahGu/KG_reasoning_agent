#!/usr/bin/env python
"""Fault-tolerant sequential experiment queue for one GPU.

One job at a time. The runner
  - starts the model services when a job needs them, checks them while it runs and restarts them when they die;
  - skips jobs whose result is already complete, so it can be stopped and started again at any time;
  - retries a failed, stalled or timed-out job (a rerun resumes inside the job), then moves on instead of stopping;
  - re-reads the queue file between jobs, so jobs can be appended while it runs;
  - writes runs/queue/STATUS.txt (progress and ETA), state.json and queue.log.
Stop it with `touch runs/queue/STOP` (after the current job) or `touch runs/queue/STOP_NOW` (at once).
Remove that file before starting it again. Standard library only.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def stamp(seconds=None):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(seconds))


def minutes_text(minutes):
    minutes = max(0, int(round(minutes)))
    return '%d 小时 %d 分钟' % (minutes // 60, minutes % 60) if minutes >= 60 else '%d 分钟' % minutes


class Services:
    """The local vLLM server and the NV embedding service: start, health checks, stop."""

    def __init__(self, root, log_dir, llm_url='http://127.0.0.1:18080/v1', embed_url='http://127.0.0.1:18081', start_minutes=12):
        self.root, self.log_dir, self.llm_url, self.embed_url, self.start_minutes = root, log_dir, llm_url, embed_url, start_minutes
        self.patterns = {'llm': 'vllm serve', 'embed': 'hyperbind.embedding_service'}
        # A model server that was killed hard leaves its engine process behind, still holding GPU memory.
        self.leftovers = {'llm': ['VLLM::EngineCore'], 'embed': []}

    @staticmethod
    def _json(url, payload=None, timeout=8):
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(url, data=data, headers={'Content-Type': 'application/json'}), timeout=timeout) as response:
            return json.load(response)

    def llm_ready(self):
        try:
            return bool(self._json(self.llm_url + '/models')['data'])
        except Exception:
            return False

    def embed_ready(self):
        try:
            return self._json(self.embed_url + '/health')['status'] == 'ready'
        except Exception:
            return False

    def ready(self):
        return self.llm_ready() and self.embed_ready()

    def llm_answers(self, timeout=240):
        """A one-token completion: tells a hung engine from a healthy one."""
        try:
            model = self._json(self.llm_url + '/models')['data'][0]['id']
            reply = self._json(self.llm_url + '/chat/completions', {
                'model': model, 'messages': [{'role': 'user', 'content': 'ping'}], 'max_tokens': 1, 'temperature': 0,
                'chat_template_kwargs': {'enable_thinking': False}}, timeout=timeout)
            return bool(reply['choices'])
        except Exception:
            return False

    @staticmethod
    def _pgrep(pattern):
        result = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
        return [int(line) for line in result.stdout.split() if line.strip().isdigit() and int(line) != os.getpid()]

    def pids(self, name):
        """The service's main process (the one that is started and answers HTTP)."""
        return self._pgrep(self.patterns[name])

    def all_pids(self, name):
        found = set(self.pids(name))
        for pattern in self.leftovers[name]:
            found.update(self._pgrep(pattern))
        return sorted(found)

    def gpu_processes(self):
        """(pid, name, ours) for every process holding GPU memory. Ours: a vLLM engine or a binary of this project's environments."""
        try:
            output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=20).stdout
        except Exception:
            return []
        found = []
        for line in output.splitlines():
            pid, _, name = line.partition(',')
            if not pid.strip().isdigit():
                continue
            try:
                with open('/proc/%s/cmdline' % pid.strip(), 'rb') as stream:
                    command = stream.read().replace(b'\0', b' ').decode('utf-8', 'replace')
            except OSError:
                command = ''
            # Only the two services count as ours; any other GPU job (Codex's, an offline encoding) is never touched.
            ours = 'VLLM' in name or 'vllm serve' in command or 'hyperbind.embedding_service' in command
            found.append((int(pid), name.strip(), ours))
        return found

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _kill_all(self, select):
        """SIGTERM, then SIGKILL, every pid returned by select(); returns the pids that were signalled."""
        signalled = set()
        for sign, wait in ((signal.SIGTERM, 25), (signal.SIGKILL, 10)):
            running = [pid for pid in select() if self._alive(pid)]
            if not running:
                break
            for pid in running:
                signalled.add(pid)
                try:
                    os.kill(pid, sign)
                except OSError:
                    pass
            deadline = time.time() + wait
            while time.time() < deadline and any(self._alive(pid) for pid in running):
                time.sleep(1)
        return sorted(signalled)

    def clear_leftovers(self, log=None):
        """Remove engine processes whose server is gone; otherwise a new server cannot get GPU memory."""
        removed = []
        for name in self.patterns:
            if not self.pids(name):
                removed += self._kill_all(lambda name=name: self.all_pids(name))
        if removed and log:
            log('services: removed leftover processes %s' % removed)
        return removed

    @staticmethod
    def gpu_used_mib():
        try:
            output = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=20).stdout
            return int(output.strip().splitlines()[0])
        except Exception:
            return None

    def start_missing(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        started = []
        if not self.pids('llm'):
            with (self.log_dir / 'vllm.log').open('ab') as stream:
                subprocess.Popen(['bash', str(self.root / 'scripts/serve_local_llm.sh')], cwd=self.root, stdout=stream,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            started.append('llm')
        if not self.pids('embed'):
            environment = dict(os.environ, PYTHONPATH=str(self.root / 'src'), TOKENIZERS_PARALLELISM='false',
                               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (self.log_dir / 'embed.log').open('ab') as stream:
                subprocess.Popen([str(self.root / 'envs/embed/bin/python'), '-u', '-m', 'hyperbind.embedding_service',
                                  '--port', '18081', '--batch-size', '1', '--provider', 'nv'], cwd=self.root, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            started.append('embed')
        return started

    def stop(self, log=None):
        """Stop both services, including engine leftovers and any of this project's processes still on the GPU."""
        self._kill_all(lambda: [pid for name in self.patterns for pid in self.all_pids(name)]
                       + [pid for pid, _, ours in self.gpu_processes() if ours])
        deadline = time.time() + 60
        while time.time() < deadline and (self.gpu_used_mib() or 0) > 3000:
            time.sleep(3)
        foreign = [(pid, name) for pid, name, ours in self.gpu_processes() if not ours]
        if foreign and log:
            log('services: GPU memory is also held by processes that are not ours and were left alone: %s' % foreign)

    def ensure(self, should_stop, log):
        """Return True once both services answer. Keeps trying (with a growing pause) until told to stop."""
        attempt = 0
        while not should_stop():
            if self.ready():
                return True
            attempt += 1
            self.clear_leftovers(log)
            started = self.start_missing()
            log('services: attempt %d, started %s, waiting for readiness' % (attempt, started or 'nothing new'))
            deadline, restarts = time.time() + self.start_minutes * 60, 0
            while time.time() < deadline and not should_stop():
                if self.ready():
                    if self.llm_answers():
                        log('services: ready')
                        return True
                    log('services: model server lists the model but does not answer')
                    break
                self.clear_leftovers(log)
                restarted = self.start_missing()  # a service that died during start-up is started again
                if restarted:
                    restarts += 1
                    if restarts <= 3 or restarts % 20 == 0:
                        log('services: %s exited during start-up and was started again (%d times so far; see logs/vllm.log, logs/embed.log)'
                            % (restarted, restarts))
                time.sleep(10)
            if should_stop():
                break
            log('services: not ready after attempt %d (GPU memory used: %s MiB); stopping both and trying again'
                % (attempt, self.gpu_used_mib()))
            self.stop(log)
            pause = min(600, 60 * attempt)
            deadline = time.time() + pause
            while time.time() < deadline and not should_stop():
                time.sleep(5)
        return False


class Runner:
    def __init__(self, root, queue_path, state_dir, services, poll=20, backoffs=(30, 120, 300), min_free_gib=6):
        self.root, self.queue_path, self.state_dir, self.services = Path(root), Path(queue_path), Path(state_dir), services
        self.poll, self.backoffs, self.min_free_gib = poll, backoffs, min_free_gib
        self.down_grace = max(45, 2 * poll)  # seconds a service may be unreachable before the job is interrupted
        self.stop_signal = False
        self.events = []
        for name in ('logs', 'done'):
            (self.state_dir / name).mkdir(parents=True, exist_ok=True)

    # ---- small helpers -------------------------------------------------------------------------------------------
    def log(self, text):
        line = '%s  %s' % (stamp(), text)
        self.events = (self.events + [line])[-8:]
        with (self.state_dir / 'queue.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        print(line, flush=True)

    def should_stop(self):
        return self.stop_signal or (self.state_dir / 'STOP_NOW').exists()

    def pause(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline and not self.should_stop():
            time.sleep(min(self.poll, 5, max(0.01, deadline - time.time())))

    def load_queue(self):
        """Re-read on every loop so that jobs can be appended while the queue runs. A broken file keeps the last good queue."""
        try:
            value = json.loads(self.queue_path.read_text(encoding='utf-8'))
            defaults = {'services': 'on', 'check': 'marker', 'max_attempts': 4, 'stall_minutes': 20, 'timeout_minutes': 240,
                        'est_minutes': 10, 'deps': []}
            defaults.update(value.get('defaults', {}))
            jobs = [dict(defaults, **job) for job in value['jobs']]
            names = [job['name'] for job in jobs]
            if len(set(names)) != len(names) or any(not isinstance(job.get('cmd'), list) or not job['cmd'] for job in jobs):
                raise ValueError('duplicate job names or a job without a command list')
            self._last_jobs = jobs
        except (OSError, ValueError, KeyError, TypeError) as error:
            if getattr(self, '_last_jobs', None) is None:
                raise
            self.log('queue file could not be read (%s: %s); keeping the previous queue' % (type(error).__name__, error))
        return self._last_jobs

    def load_state(self):
        path = self.state_dir / 'state.json'
        return json.loads(path.read_text()) if path.exists() else {'jobs': {}, 'passes': 1, 'created': stamp()}

    def save_state(self, state):
        state['updated'] = stamp()
        temporary = self.state_dir / 'state.json.tmp'
        temporary.write_text(json.dumps(state, indent=1, ensure_ascii=False))
        temporary.replace(self.state_dir / 'state.json')

    def check(self, job):
        """Is the job's result complete? 'chain': run_chain summary without errors; 'file': a non-empty file."""
        kind = job['check']
        if kind == 'marker':
            return True
        path = self.root / job['path']
        if not path.exists() or path.stat().st_size == 0:
            return False
        if kind == 'file':
            return True
        if kind == 'chain':
            try:
                summary = json.loads(path.read_text())
            except ValueError:
                return False
            return summary.get('errors', 1) == 0 and summary.get('requested', 0) > 0 and summary.get('scored') == summary.get('requested')
        raise ValueError('Unknown check: ' + kind)

    def marker(self, job):
        return self.state_dir / 'done' / (job['name'] + '.json')

    def is_done(self, job):
        return self.marker(job).exists() or (job['check'] != 'marker' and self.check(job))

    def progress_count(self, job):
        path = self.root / job['progress'] if job.get('progress') else None
        if path is None or not path.exists():
            return None
        with path.open('rb') as stream:
            return sum(chunk.count(b'\n') for chunk in iter(lambda: stream.read(1 << 20), b''))

    def signature(self, job, log_path):
        parts = [log_path.stat().st_size if log_path.exists() else 0]
        if job.get('progress') and (self.root / job['progress']).exists():
            parts.append((self.root / job['progress']).stat().st_size)
        return tuple(parts)

    # ---- status file --------------------------------------------------------------------------------------------
    def write_status(self, jobs, state, current=None, started=None, attempt=None, headline=None):
        done = [job for job in jobs if state['jobs'].get(job['name'], {}).get('status') == 'done']
        failed = [job for job in jobs if state['jobs'].get(job['name'], {}).get('status') == 'failed']
        blocked = [job for job in jobs if state['jobs'].get(job['name'], {}).get('status') == 'blocked']
        waiting = [job for job in jobs if job not in done and job not in failed and job not in blocked and job is not current]
        remaining = sum(job['est_minutes'] for job in waiting)
        lines = ['队列状态  UTC %s  （北京时间 %s）' % (stamp(), stamp(time.time() + 8 * 3600)[5:16])]
        if headline:
            lines.append(headline)
        if current is not None:
            elapsed = (time.time() - started) / 60
            count, total = self.progress_count(current), current.get('questions')
            left = max(0.5, current['est_minutes'] - elapsed)
            detail = ''
            if count is not None and total:
                first = getattr(self, '_first_count', None)
                if first is None or first[0] != current['name']:
                    self._first_count = first = (current['name'], count, time.time())
                gained, spent = count - first[1], (time.time() - first[2]) / 60
                if gained >= 10 and spent > 0.5:
                    left = max(0.2, (total - count) / (gained / spent))
                detail = '，%d / %d 题' % (min(count, total), total)
            remaining += left
            lines.append('正在运行：%s（第 %d 次尝试，已运行 %s%s）' % (current['name'], attempt, minutes_text(elapsed), detail))
        lines.append('已完成 %d / %d，失败 %d，受阻 %d' % (len(done), len(jobs), len(failed), len(blocked)))
        if current is not None or waiting:
            lines.append('预计剩余：约 %s（大约北京时间 %s 结束）' % (
                minutes_text(remaining), stamp(time.time() + remaining * 60 + 8 * 3600)[5:16]))
        if waiting:
            lines.append('接下来：' + '、'.join(job['name'] for job in waiting[:6]) + (' …' if len(waiting) > 6 else ''))
        for job in failed:
            lines.append('失败：%s —— %s' % (job['name'], state['jobs'][job['name']].get('note', '')[:200]))
        for job in blocked:
            lines.append('受阻（依赖的任务失败）：%s' % job['name'])
        lines.append('最近事件：')
        lines.extend('  ' + event for event in self.events[-6:])
        lines.append('停止：touch runs/queue/STOP（跑完当前任务后停）或 touch runs/queue/STOP_NOW（立刻停）；再次启动前删除该文件。')
        temporary = self.state_dir / 'STATUS.txt.tmp'
        temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        temporary.replace(self.state_dir / 'STATUS.txt')

    # ---- one attempt of one job -------------------------------------------------------------------------------
    def kill(self, process):
        for sign, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
            try:
                os.killpg(process.pid, sign)
            except ProcessLookupError:
                return
            deadline = time.time() + wait
            while time.time() < deadline and process.poll() is None:
                time.sleep(0.2)
            if process.poll() is not None:
                return

    def run_once(self, job, attempt, jobs, state):
        mode = job['services']
        if mode == 'on' and not self.services.ensure(self.should_stop, self.log):
            return 'stopped', 0.0
        if mode == 'off':
            self.services.stop(self.log)
        log_path = self.state_dir / 'logs' / ('%s.attempt%d.log' % (job['name'], attempt))
        self._first_count = None
        environment = dict(os.environ, PYTHONPATH=str(self.root / 'src'), OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4',
                           TOKENIZERS_PARALLELISM='false')
        environment.update(job.get('env', {}))
        started = time.time()
        with log_path.open('ab') as stream:
            process = subprocess.Popen(job['cmd'], cwd=self.root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        # Remembered on disk: if this runner dies, the next one ends this job process before starting the job again.
        state['running'] = {'name': job['name'], 'pid': process.pid, 'cmd': job['cmd']}
        self.save_state(state)
        signature, changed, down_since = None, started, None
        outcome = None
        while process.poll() is None:
            time.sleep(self.poll)
            current = self.signature(job, log_path)
            if current != signature:
                signature, changed = current, time.time()
            self.write_status(jobs, state, job, started, attempt)
            if self.should_stop():
                outcome = 'stopped'
            elif time.time() - started > job['timeout_minutes'] * 60:
                outcome = 'timeout'
            elif time.time() - changed > job['stall_minutes'] * 60:
                outcome = 'stalled'
            elif mode == 'on' and not self.services.ready():
                down_since = down_since or time.time()
                if time.time() - down_since > self.down_grace:
                    outcome = 'service_down'
            else:
                down_since = None
            if outcome:
                self.kill(process)
                break
        seconds = time.time() - started
        state.pop('running', None)
        self.save_state(state)
        if outcome is None:
            outcome = 'done' if process.returncode == 0 and self.check(job) else 'failed'
        if outcome == 'done':
            self.marker(job).write_text(json.dumps({'finished': stamp(), 'seconds': round(seconds), 'attempt': attempt}))
        return outcome, seconds

    @staticmethod
    def tail(path, characters=600):
        if not path.exists():
            return ''
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - characters))
            return stream.read().decode('utf-8', 'replace').strip().replace('\n', ' | ')

    # ---- the queue ----------------------------------------------------------------------------------------------
    def next_job(self, jobs, state):
        status = {job['name']: state['jobs'].get(job['name'], {}).get('status') for job in jobs}
        for job in jobs:
            name = job['name']
            if status[name] in ('done', 'failed', 'blocked'):
                continue
            if self.is_done(job):
                state['jobs'].setdefault(name, {})['status'] = 'done'
                status[name] = 'done'
                self.log('%s: result already complete, skipped' % name)
                continue
            needs = [dep for dep in job['deps'] if status.get(dep) != 'done']
            if any(status.get(dep) in ('failed', 'blocked') or dep not in status for dep in needs):
                state['jobs'].setdefault(name, {})['status'] = 'blocked'
                status[name] = 'blocked'
                self.log('%s: blocked, a job it depends on failed or does not exist (%s)' % (name, needs))
                continue
            if needs:
                continue  # a dependency comes later in the file; look again after it has run
            return job
        return None

    def reap_previous(self):
        """After a runner crash its job may still be running in its own session; two copies must never write one result."""
        state = self.load_state()
        previous = state.pop('running', None)
        if previous:
            try:
                with open('/proc/%d/cmdline' % previous['pid'], 'rb') as stream:
                    command = [part.decode('utf-8', 'replace') for part in stream.read().split(b'\0') if part]
            except OSError:
                command = []
            if command and command[-len(previous['cmd']) + 1:] == previous['cmd'][1:]:
                self.log('%s: a job process from the previous runner is still alive (pid %d); ending it' % (previous['name'], previous['pid']))
                def gone():
                    try:
                        with open('/proc/%d/stat' % previous['pid']) as stream:
                            return stream.read().rsplit(')', 1)[1].split()[0] == 'Z'
                    except (OSError, IndexError):
                        return True

                for sign, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
                    try:
                        os.killpg(previous['pid'], sign)
                    except OSError:
                        break
                    deadline = time.time() + wait
                    while time.time() < deadline and not gone():
                        time.sleep(0.2)
                    if gone():
                        break
            self.save_state(state)

    def run(self):
        self.log('queue runner started (pid %d), queue file %s' % (os.getpid(), self.queue_path))
        self.reap_previous()
        while not self.should_stop():
            jobs, state = self.load_queue(), self.load_state()
            if (self.state_dir / 'STOP').exists():
                self.log('STOP file present: stopping before the next job')
                break
            free = shutil.disk_usage(self.root).free / 1024 ** 3
            if free < self.min_free_gib:
                self.log('paused: only %.1f GiB free disk (needs %.0f); checking again in 10 minutes' % (free, self.min_free_gib))
                self.write_status(jobs, state, headline='暂停：硬盘剩余 %.1f GiB，低于 %.0f GiB。腾出空间后自动继续。' % (free, self.min_free_gib))
                self.pause(600)
                continue
            job = self.next_job(jobs, state)
            self.save_state(state)
            if job is None:
                failed = [name for name, row in state['jobs'].items() if row.get('status') in ('failed', 'blocked')
                          and any(item['name'] == name for item in jobs)]
                if failed and state.get('passes', 1) < 2:
                    state['passes'] = 2
                    for name in failed:
                        state['jobs'][name] = {'status': None, 'attempts': 0, 'second_pass': True}
                    self.save_state(state)
                    self.log('first pass finished; retrying %d failed or blocked job(s) once more: %s' % (len(failed), failed))
                    continue
                self.log('queue finished: %d job(s) failed or blocked' % len(failed) if failed else 'queue finished: every job is done')
                self.write_status(jobs, state, headline='队列已全部跑完。' if not failed else '队列已结束，但有 %d 个任务没有完成（见下）。' % len(failed))
                self.services.stop(self.log)
                return 0
            row = state['jobs'].setdefault(job['name'], {})
            attempt = row.get('attempts', 0) + row.get('service_restarts', 0) + 1
            self.log('%s: attempt %d starts (estimate %s min)' % (job['name'], attempt, job['est_minutes']))
            outcome, seconds = self.run_once(job, attempt, jobs, state)
            log_path = self.state_dir / 'logs' / ('%s.attempt%d.log' % (job['name'], attempt))
            if outcome == 'stopped':
                self.log('%s: interrupted by a stop request after %.1f min' % (job['name'], seconds / 60))
                break
            if outcome == 'done':
                row.update(status='done', seconds=round(seconds), finished=stamp())
                self.log('%s: done in %.1f min' % (job['name'], seconds / 60))
                self.save_state(state)
                continue
            self.log('%s: %s after %.1f min; log tail: %s' % (job['name'], outcome, seconds / 60, self.tail(log_path, 300)))
            if job['services'] == 'on' and (outcome in ('service_down', 'stalled') or not self.services.llm_answers(90)):
                self.log('services look unhealthy after %s: stopping them, the next attempt starts them again' % outcome)
                self.services.stop(self.log)
            if outcome == 'service_down' and row.get('service_restarts', 0) < 6:
                row['service_restarts'] = row.get('service_restarts', 0) + 1  # not the job's fault: does not use up an attempt
            else:
                row['attempts'] = row.get('attempts', 0) + 1
            if row.get('attempts', 0) >= job['max_attempts']:
                row.update(status='failed', note='%s; %s' % (outcome, self.tail(log_path)))
                self.log('%s: gave up after %d attempts; the queue continues with the next job' % (job['name'], row['attempts']))
                self.save_state(state)
                continue
            self.save_state(state)
            self.pause(self.backoffs[min(max(row.get('attempts', 1) - 1, 0), len(self.backoffs) - 1)])
        jobs, state = self.load_queue(), self.load_state()
        self.write_status(jobs, state, headline='已按停止请求停下。删除 runs/queue/STOP 或 STOP_NOW 后重新启动会从断点继续。')
        self.log('queue runner stopped on request; model services are left as they are')
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--queue', type=Path, default=ROOT / 'configs/queue_v1.json')
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'runs/queue')
    arguments = parser.parse_args()
    arguments.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (arguments.state_dir / 'lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('another queue runner holds the lock; nothing to do')
        return 0
    runner = Runner(arguments.root, arguments.queue, arguments.state_dir, Services(arguments.root, arguments.state_dir / 'logs'))
    if any((arguments.state_dir / name).exists() for name in ('STOP', 'STOP_NOW')):
        runner.log('a STOP file exists in %s: remove it to let the queue run' % arguments.state_dir)
        return 0
    signal.signal(signal.SIGTERM, lambda *_: setattr(runner, 'stop_signal', True))
    signal.signal(signal.SIGINT, lambda *_: setattr(runner, 'stop_signal', True))
    try:
        return runner.run()
    except Exception:  # never die silently: record, wait a minute, let the supervisor start a fresh runner
        runner.log('runner crashed:\n' + traceback.format_exc())
        time.sleep(60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
