"""The queue runner's failure handling, tested with shell jobs and a fake service manager (no GPU, no model)."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_queue  # noqa: E402


class FakeServices:
    def __init__(self, down_after=None):
        self.ensured = self.stopped = self.calls = 0
        self.down_after = down_after

    def ensure(self, should_stop, log):
        self.ensured, self.calls = self.ensured + 1, 0
        return True

    def stop(self, log=None):
        self.stopped += 1

    def ready(self):
        self.calls += 1
        return self.down_after is None or self.calls <= self.down_after

    def llm_answers(self, timeout=0):
        return True


def runner(tmp_path, jobs, services=None, **defaults):
    queue = tmp_path / 'queue.json'
    queue.write_text(json.dumps({'defaults': dict({'services': 'any', 'stall_minutes': 5, 'timeout_minutes': 5}, **defaults),
                                 'jobs': jobs}))
    return run_queue.Runner(tmp_path, queue, tmp_path / 'state', services or FakeServices(), poll=0.05, backoffs=(0, 0, 0),
                            min_free_gib=0)


def state(tmp_path):
    return json.loads((tmp_path / 'state/state.json').read_text())['jobs']


def lines(path):
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_finished_job_is_not_run_again(tmp_path):
    jobs = [{'name': 'a', 'cmd': ['bash', '-c', 'echo run >> count.txt; echo ok > out.txt'], 'check': 'file', 'path': 'out.txt'}]
    assert runner(tmp_path, jobs).run() == 0 and state(tmp_path)['a']['status'] == 'done'
    assert runner(tmp_path, jobs).run() == 0 and lines(tmp_path / 'count.txt') == 1
    assert '队列已全部跑完' in (tmp_path / 'state/STATUS.txt').read_text()


def test_failing_job_is_retried_until_it_succeeds(tmp_path):
    command = 'n=$(cat n.txt 2>/dev/null || echo 0); echo $((n+1)) > n.txt; [ "$n" -ge 2 ]'
    assert runner(tmp_path, [{'name': 'flaky', 'cmd': ['bash', '-c', command]}]).run() == 0
    row = state(tmp_path)['flaky']
    assert row['status'] == 'done' and row['attempts'] == 2 and (tmp_path / 'n.txt').read_text().strip() == '3'


def test_a_job_that_keeps_failing_does_not_stop_the_queue(tmp_path):
    jobs = [{'name': 'bad', 'cmd': ['bash', '-c', 'echo x >> bad.txt; exit 1'], 'max_attempts': 2},
            {'name': 'child', 'cmd': ['bash', '-c', 'echo x >> child.txt'], 'deps': ['bad']},
            {'name': 'good', 'cmd': ['bash', '-c', 'echo x >> good.txt']}]
    assert runner(tmp_path, jobs).run() == 0
    rows = state(tmp_path)
    assert rows['bad']['status'] == 'failed' and rows['child']['status'] == 'blocked' and rows['good']['status'] == 'done'
    assert lines(tmp_path / 'bad.txt') == 4  # two attempts in the first pass and two in the single retry pass
    assert lines(tmp_path / 'good.txt') == 1 and not (tmp_path / 'child.txt').exists()
    assert '失败：bad' in (tmp_path / 'state/STATUS.txt').read_text()


def test_stalled_job_is_killed_and_the_queue_moves_on(tmp_path):
    jobs = [{'name': 'hang', 'cmd': ['bash', '-c', 'echo x >> hang.txt; sleep 60'], 'stall_minutes': 0.005, 'max_attempts': 1},
            {'name': 'after', 'cmd': ['bash', '-c', 'echo x >> after.txt']}]
    began = time.time()
    assert runner(tmp_path, jobs).run() == 0
    assert time.time() - began < 30 and state(tmp_path)['hang']['status'] == 'failed' and lines(tmp_path / 'after.txt') == 1
    assert 'stalled' in state(tmp_path)['hang']['note']


def test_service_outage_interrupts_the_job_and_does_not_use_up_an_attempt(tmp_path):
    services = FakeServices(down_after=2)
    command = '[ -f flag ] || { touch flag; sleep 60; }'
    instance = runner(tmp_path, [{'name': 'needs_model', 'cmd': ['bash', '-c', command], 'services': 'on'}], services)
    instance.down_grace = 0.1
    assert instance.run() == 0
    row = state(tmp_path)['needs_model']
    assert row['status'] == 'done' and row['service_restarts'] == 1 and row.get('attempts', 0) == 0
    assert services.ensured == 2 and services.stopped >= 2  # once after the outage, once when the queue finished


def test_job_left_running_by_a_dead_runner_is_ended_before_the_job_starts_again(tmp_path):
    import subprocess
    command = ['bash', '-c', 'echo x >> twice.txt; sleep 120']
    orphan = subprocess.Popen(command, cwd=tmp_path, start_new_session=True)
    instance = runner(tmp_path, [{'name': 'j', 'cmd': ['bash', '-c', 'echo done > j.txt'], 'check': 'file', 'path': 'j.txt'}])
    instance.save_state({'jobs': {}, 'passes': 1, 'running': {'name': 'j', 'pid': orphan.pid, 'cmd': command}})
    assert instance.run() == 0 and orphan.wait(timeout=30) != 0  # ended by the new runner, not by itself
    assert 'running' not in json.loads((tmp_path / 'state/state.json').read_text())


def test_chain_check_requires_a_complete_summary(tmp_path):
    instance = runner(tmp_path, [{'name': 'c', 'cmd': ['true'], 'check': 'chain', 'path': 'summary.json'}])
    job = instance.load_queue()[0]
    assert not instance.check(job)
    (tmp_path / 'summary.json').write_text(json.dumps({'errors': 2, 'requested': 500, 'scored': 498}))
    assert not instance.check(job)
    (tmp_path / 'summary.json').write_text(json.dumps({'errors': 0, 'requested': 500, 'scored': 500}))
    assert instance.check(job) and instance.is_done(job)


def test_stop_file_and_broken_queue_file(tmp_path):
    instance = runner(tmp_path, [{'name': 'a', 'cmd': ['bash', '-c', 'echo x >> a.txt']}])
    (tmp_path / 'state/STOP').write_text('')
    assert instance.run() == 0 and not (tmp_path / 'a.txt').exists()
    (tmp_path / 'state/STOP').unlink()
    (tmp_path / 'queue.json').write_text('{ not json')
    assert [job['name'] for job in instance.load_queue()] == ['a']  # the last good queue is kept
    (tmp_path / 'queue.json').write_text(json.dumps({'jobs': [{'name': 'a', 'cmd': ['bash', '-c', 'echo x >> a.txt']},
                                                              {'name': 'b', 'cmd': ['bash', '-c', 'echo x >> b.txt']}]}))
    assert instance.run() == 0 and lines(tmp_path / 'a.txt') == 1 and lines(tmp_path / 'b.txt') == 1
