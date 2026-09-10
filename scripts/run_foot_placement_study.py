"""Explicit bounded FE-02 foot-placement experiment; no daemon or gait claim."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Works both as a module in tests and as the documented source-checkout CLI.
if __package__:
    from .run_root_ab_study import PROTECTED, hashes, run_stage, write_json
else:
    from run_root_ab_study import PROTECTED, hashes, run_stage, write_json


def main():
    from organism_core.foot_placement import PROTOCOL, protocol_hash
    from organism_core.root_ab import file_hash
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); root = Path(__file__).resolve().parents[1]
    result = {'schema': 1, 'kind': PROTOCOL['id'], 'status': 'running', 'stages': [],
              'scientific_outcome': 'not_run', 'parameters': PROTOCOL, 'protocol_sha256': protocol_hash(),
              'walking_claimed': False, 'standing_claimed': False, 'CNS_executed': False,
              'biological_validation': False, 'github_sha': os.environ.get('GITHUB_SHA')}
    try:
        if sys.platform != 'linux' or not Path('/proc/self/statm').is_file():
            raise NotImplementedError('Requires the tested Linux process-group resource supervisor')
        for dependency in ('numpy', 'mujoco', 'flygym', 'scipy'):
            if importlib.util.find_spec(dependency) is None:
                raise ModuleNotFoundError('Required experiment dependency: '+dependency)
        record_path = root/'examples/fe02-foot-placement.json'
        if json.loads(record_path.read_text())['parameters'] != PROTOCOL:
            raise ValueError('Versioned protocol differs from implemented experiment')
        protected = hashes(root, PROTECTED); result['protected_before'] = protected
        code_paths = sorted(p.relative_to(root) for folder in ('organism_core', 'scripts')
                            for p in (root/folder).glob('*.py'))
        code_paths += [Path(n) for n in ('audit_source_tendons.py', 'transfer_six_leg_tendons.py',
                       'transfer_lf_tendons.py', 'pyproject.toml', 'examples/fe02-foot-placement.json')]
        result['code_sha256'] = hashes(root, code_paths)
        # The immutable execution identity and all settings precede data collection.
        write_json(workspace/'protocol.json', result)
        (workspace/'runs').mkdir()
        env = dict(os.environ, FLY_EFFECT_HOME=str(workspace),
                   FLYGYM_ASSET_CACHE_DIR=str(workspace/'cache/flygym-assets'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
        stages = [
            ('source-audit', [sys.executable, str(root/'audit_source_tendons.py')], 180),
            ('source-registration', [sys.executable, str(root/'transfer_six_leg_tendons.py')], 180),
            ('source-study', [sys.executable, '-c', 'from organism_core.tendon_study import run_study; import sys; run_study(sys.argv[1])', str(workspace/'source-study')], 300),
            ('foot-placement', [sys.executable, '-m', 'organism_core.foot_placement', '--workspace', str(workspace)], 250),
            ('verification', [sys.executable, '-m', 'organism_core.foot_placement', '--workspace', str(workspace), '--verify'], 120),
        ]
        for name, command, limit in stages:
            result['active_stage'] = name
            write_json(workspace/'pipeline-result.json', result)
            evidence = run_stage(command, root=root, env=env, log=workspace/(name+'.log'),
                                 deadline=started+PROTOCOL['outer_wall_s'], stage_limit=limit,
                                 memory_limit=PROTOCOL['memory_mib']*1024**2)
            result['stages'].append({'name': name, **evidence})
        result['protected_after'] = hashes(root, PROTECTED)
        if result['protected_after'] != protected or hashes(root, code_paths) != result['code_sha256']:
            raise ValueError('Protected inputs or executed code changed')
        verdict = json.loads((workspace/'foot-placement/verification.json').read_text())
        if not verdict['evidence_valid'] or verdict['status'] != 'completed':
            raise ValueError('Evidence validation failed')
        result.update(status='completed', scientific_outcome=verdict['scientific_outcome'],
                      verification_sha256=file_hash(workspace/'foot-placement/verification.json'),
                      protected_unchanged=True)
    except Exception as exc:
        if hasattr(exc, 'stage_evidence'):
            result['stages'].append({'name': result.get('active_stage'), **exc.stage_evidence})
        outcome = 'inconclusive' if isinstance(exc, (TimeoutError, MemoryError)) else (
            'blocked' if isinstance(exc, (NotImplementedError, ModuleNotFoundError)) else 'invalid_experiment')
        worker = workspace/str(result.get('active_stage', ''))/'result.json'
        if isinstance(exc, subprocess.CalledProcessError) and worker.is_file():
            state = json.loads(worker.read_text())
            if state.get('scientific_outcome') in ('inconclusive', 'blocked', 'invalid_experiment'):
                outcome = state['scientific_outcome']
        result.update(status='failed', scientific_outcome=outcome, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-started
        write_json(workspace/'pipeline-result.json', result)


if __name__ == '__main__':
    main()
