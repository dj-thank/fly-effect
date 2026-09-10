"""Explicit Linux FE-02 A/B run, 600s wall / 8192MiB monitored RSS ceiling.

Each stage has a new process group. Time/resource failure kills the whole group,
retains logs/receipts and is not a claim of physical infeasibility. No daemon.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

WALL_SECONDS = 600.
MEMORY_BYTES = 8192*1024**2
PROTECTED = ('organism_core/brain.py', 'organism_core/graph_lock.json', 'ACCEPTANCE.json')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def hashes(root, paths):
    return {str(path): hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}


def group_rss(group, *, proc=Path('/proc')):
    """Sum child process-group + supervisor RSS. Shared pages count conservatively."""
    total = 0; found_self = False
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            mine = int(entry.name) == os.getpid()
            if int(stat[2]) == group or mine:
                total += int((entry/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                found_self |= mine
        except FileNotFoundError:
            continue  # A process exited during the sample.
        except PermissionError:
            if int(entry.name) in (group, os.getpid()):
                raise RuntimeError('Resource monitor cannot inspect an owned process')
    if not found_self:
        raise RuntimeError('Resource monitor could not sample supervisor')
    return total


def run_stage(command, *, root, env, log, deadline, stage_limit, memory_limit=MEMORY_BYTES):
    import resource
    started = time.monotonic()
    remaining = min(stage_limit, deadline-started)
    if not remaining > 0:
        raise TimeoutError('Outer wall budget exhausted before stage')
    def limit_address_space():
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    peak = 0
    with Path(log).open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, preexec_fn=limit_address_space)
        try:
            while True:
                peak = max(peak, group_rss(process.pid))
                if peak > memory_limit:
                    raise MemoryError('Aggregate sampled process-group RSS budget exceeded')
                if time.monotonic()-started > remaining:
                    raise TimeoutError('Stage or outer wall budget exceeded')
                code = process.poll()
                if code is not None:
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    break
                time.sleep(.05)
        except Exception as exc:
            exc.stage_evidence = {'wall_seconds': time.monotonic()-started,
                'peak_sampled_group_plus_supervisor_rss_bytes': peak, 'status': 'failed'}
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {'wall_seconds': time.monotonic()-started, 'peak_sampled_group_plus_supervisor_rss_bytes': peak,
            'sampling_interval_s': .05, 'per_process_address_space_limit_bytes': memory_limit,
            'status': 'completed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); root = Path(__file__).resolve().parents[1]
    result = {'schema': 1, 'kind': 'FE-02-root-compliance-ab-v1', 'status': 'running',
              'scientific_outcome': 'not_run', 'wall_limit_s': WALL_SECONDS,
              'memory_limit_bytes': MEMORY_BYTES, 'stages': [], 'biological_validation': False,
              'walking_claimed': False, 'standing_claimed': False, 'CNS_executed': False}
    try:
        if sys.platform != 'linux' or not Path('/proc/self/statm').is_file():
            raise NotImplementedError('This bounded supervisor requires Linux /proc; do not run unmonitored')
        for dependency in ('numpy', 'mujoco', 'flygym', 'scipy'):
            if importlib.util.find_spec(dependency) is None:
                raise ModuleNotFoundError('Required experiment dependency: '+dependency)
        protected = hashes(root, PROTECTED); result['protected_before'] = protected
        code_paths = sorted(p.relative_to(root) for folder in ('organism_core', 'scripts')
                            for p in (root/folder).glob('*.py'))
        code_paths += [Path(n) for n in ('audit_source_tendons.py', 'transfer_six_leg_tendons.py', 'transfer_lf_tendons.py', 'pyproject.toml')]
        result['code_sha256'] = hashes(root, code_paths)
        result['github_sha'] = os.environ.get('GITHUB_SHA')
        result['protocol'] = {'source_static_pose': 'legacy source-study/support_qpos; shared, not independently settled',
                              'pose_count': 9, 'depth_count': 5, 'pose_wall_s': 120,
                              'condition_subprocess_timeout_s': 180, 'dynamic_duration_s': .25,
                              'seed': 0, 'root_changes': ['stiffness', 'damping'], 'armature_changed': False}
        write_json(workspace/'protocol.json', result)
        (workspace/'runs').mkdir()
        env = dict(os.environ, FLY_EFFECT_HOME=str(workspace),
                   FLYGYM_ASSET_CACHE_DIR=str(workspace/'cache/flygym-assets'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
        stages = [
            ('source-audit', [sys.executable, str(root/'audit_source_tendons.py')], 180),
            ('source-registration', [sys.executable, str(root/'transfer_six_leg_tendons.py')], 180),
            ('source-study', [sys.executable, '-c', 'from organism_core.tendon_study import run_study; import sys; run_study(sys.argv[1])', str(workspace/'source-study')], 300),
        ]
        for condition in ('legacy_anchored', 'unanchored_root_candidate'):
            stages.append((condition, [sys.executable, '-m', 'organism_core.root_ab', '--workspace', str(workspace), '--condition', condition], 180))
        stages.append(('comparison', [sys.executable, '-m', 'organism_core.root_ab', '--workspace', str(workspace)], 30))
        for name, command, limit in stages:
            result['active_stage'] = name
            write_json(workspace/'pipeline-result.json', result)
            evidence = run_stage(command, root=root, env=env, log=workspace/(name+'.log'),
                                 deadline=started+WALL_SECONDS, stage_limit=limit)
            result['stages'].append({'name': name, **evidence})
        result['protected_after'] = hashes(root, PROTECTED)
        if result['protected_after'] != protected or hashes(root, code_paths) != result['code_sha256']:
            raise ValueError('Protected input or executed code changed during the experiment')
        comparison = json.loads((workspace/'comparison.json').read_text())
        result.update(status='completed', scientific_outcome=comparison['scientific_outcome'],
                      comparison_sha256=hashlib.sha256((workspace/'comparison.json').read_bytes()).hexdigest(),
                      protected_unchanged=True)
    except Exception as exc:
        if hasattr(exc, 'stage_evidence'):
            result['stages'].append({'name': result.get('active_stage'), **exc.stage_evidence})
        outcome = 'inconclusive' if isinstance(exc, (TimeoutError, MemoryError)) else (
            'blocked' if isinstance(exc, (NotImplementedError, ModuleNotFoundError)) else 'invalid_comparison')
        # Propagate a classified worker failure rather than calling a timeout infeasibility.
        worker = workspace/str(result.get('active_stage', ''))/'result.json'
        if isinstance(exc, subprocess.CalledProcessError) and worker.is_file():
            state = json.loads(worker.read_text())
            if state.get('scientific_outcome') in ('inconclusive', 'blocked', 'invalid_comparison'):
                outcome = state['scientific_outcome']
        result.update(status='failed', scientific_outcome=outcome, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-started
        write_json(workspace/'pipeline-result.json', result)


if __name__ == '__main__':
    main()
