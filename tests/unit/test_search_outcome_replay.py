"""Exercise real saved-input LP replay; these synthetic rows are not a body."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')
from organism_core import full_descent, global_sample
from organism_core.passive_margin import margin_solve
from organism_core.passive_posture import FEASIBLE_POSE, FIELDS, solve_subsystem
from organism_core.root_ab import file_hash, write_json
from organism_core.static_support import solve_support
from organism_core.verification_receipt import (
    input_hashes_unchanged, latest_outcome_receipt, outcome_source_hashes,
    validate_outcome_sources,
)


def saved_study(workspace, kind, *, feasible=False, stop='wall_budget_exhausted'):
    module = full_descent if kind == 'full' else global_sample
    directory = workspace / ('full-descent' if kind == 'full' else 'global-sample')
    directory.mkdir()
    T, C, b = np.zeros((7, 1)), np.zeros((7, 3)), np.zeros(7)
    C[2, 2] = b[2] = 1.
    b[6] = 0. if feasible else 1.
    limits, friction, intervals = np.ones(1), np.array([.5]), np.array([[-1., 1.]])
    inputs = dict(zip(FIELDS, (T, C, b, limits, friction)))
    support = solve_support(T, C, b, limits, friction)
    subsystem = solve_subsystem(T, C, b, limits, friction, list(range(7)))
    margin = margin_solve(T, C, b, limits, friction, list(range(6)), [6], intervals)
    residual, _ = full_descent._full_residual(support)
    assert support['status'] == ('feasible' if feasible else full_descent.INFEASIBLE)
    assert margin['status'] == 'measured'
    qpos = np.zeros(8); qpos[3] = 1.
    descent = {'status': stop, 'accepted_steps': 0, 'unresolved_evals': 0,
               'initial_R': residual, 'final_R': residual,
               'accepted': [{'accepted': 0, 'R': residual}],
               'terminal_qpos': qpos.tolist(), 'terminal_tilt_rad': 0.}
    cert = {'accepted_index': 0, 'status': 'completed', 'samples': [],
            'any_full_feasible': feasible, 'any_passive_feasible': feasible}
    arrays = {}
    prefix = 'fdesc_00_' if kind == 'full' else 'gdesc_00_'
    arrays[prefix+'acc_0000_qpos'] = qpos
    arrays.update({prefix+'acc_0000_'+k: v for k, v in inputs.items()})
    for index, depth in enumerate(module.PROTOCOL['depths_native']):
        sample = {'index': index, 'depth_native': depth, 'evidence_saved': True,
                  'passive_rows': [6],
                  'support': {'status': FEASIBLE_POSE if feasible else support['status']},
                  'passive_subsystem': {'status': subsystem['status']},
                  'margin': {'status': margin['status'], 't': margin['t']}}
        cert['samples'].append(sample)
        sp = prefix+f'cert_00_depth_{index:02d}_'
        arrays.update({sp+k: v for k, v in inputs.items()})
        arrays[sp+'margin_intervals'] = intervals
    old_outcome = ('descent_floored_without_feasibility' if kind == 'full'
                   else 'no_global_improvement')
    result = {'schema': 1, 'protocol_sha256': module.protocol_hash(),
              'scientific_outcome': old_outcome}
    if kind == 'full':
        (workspace/'passive-posture').mkdir()
        np.savez(workspace/'passive-posture/observations.npz', pose_00_passive_dofs=np.array([6]))
        result.update(declared_seeds=1, seeds=[{
            'meta': {'index': 0}, 'status': 'completed',
            'descent': descent, 'certifications': [cert]}])
    else:
        descent['certifications'] = [cert]
        samples = [{'index': i, 'status': 'evaluated', 'eligible': False,
                    'evidence_saved': False, 'R': None, 'probe_status': 'no_eligible_sample'}
                   for i in range(module.PROTOCOL['n_samples'])]
        samples[0].update(eligible=True, evidence_saved=True, R=residual,
                          probe_status='feasible' if feasible else 'infeasible',
                          descent=descent, descent_ordinal=0)
        arrays.update({'gsamp_000_'+k: v for k, v in inputs.items()})
        result.update(samples=samples, baseline_floor=.15)
        baseline_dir = workspace/'full-descent'; baseline_dir.mkdir()
        write_json(baseline_dir/'result.json', {'seeds': [
            {'status': 'completed', 'descent': {'final_R': .15}}]})
        result['input_sha256'] = {'full-descent/result.json': file_hash(baseline_dir/'result.json')}
    write_json(directory/'result.json', result)
    write_json(directory/'protocol.json', {'protocol': module.PROTOCOL})
    np.savez(directory/'observations.npz', **arrays)
    write_json(directory/'verification.json', {'schema': 1, 'legacy': 'keep these bytes'})
    return module, directory, result, arrays


@pytest.mark.parametrize('kind', ['full', 'global'])
@pytest.mark.parametrize('stop', ['wall_budget_exhausted', 'eval_budget_exhausted'])
def test_replayed_negative_budget_stop_is_inconclusive_and_keeps_legacy(tmp_path, kind, stop):
    module, directory, _, _ = saved_study(tmp_path, kind, stop=stop)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    first = module.verify(tmp_path)
    second = module.verify(tmp_path)
    assert first['evidence_valid'] is True and first['scientific_outcome'] == 'inconclusive'
    assert first['outcome_reclassified'] is True
    assert first['verification_method'] == 'saved_input_lp_replay'
    assert first['source_sha256'] == outcome_source_hashes(directory)
    assert first['verification_receipt'] != second['verification_receipt']
    assert first['outcome_classification']['search_complete'] is False
    assert all((directory/name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize('kind', ['full', 'global'])
def test_completed_lp_positive_survives_descent_budget_stop(tmp_path, kind):
    module, _, _, _ = saved_study(tmp_path, kind, feasible=True)
    verdict = module.verify(tmp_path)
    assert verdict['evidence_valid'] is True
    assert verdict['scientific_outcome'] == 'found_full_support_pose'
    assert verdict['outcome_classification']['search_complete'] is False


@pytest.mark.parametrize('kind', ['full', 'global'])
@pytest.mark.parametrize('flag', ['any_full_feasible', 'any_passive_feasible'])
def test_negative_lp_replay_rejects_stale_positive_summary(tmp_path, kind, flag):
    module, directory, result, _ = saved_study(tmp_path, kind)
    certs = (result['seeds'][0]['certifications'] if kind == 'full'
             else result['samples'][0]['descent']['certifications'])
    certs[0][flag] = True
    write_json(directory/'result.json', result)
    verdict = module.verify(tmp_path)
    assert verdict['evidence_valid'] is False
    assert verdict['status'] == 'failed'
    assert any(flag+' disagrees' in error for error in verdict['errors'])


@pytest.mark.parametrize('kind', ['full', 'global'])
def test_replayed_step_checks_prefixed_terminal_qpos(tmp_path, kind):
    module, directory, result, _ = saved_study(tmp_path, kind)
    descent = (result['seeds'][0]['descent'] if kind == 'full'
               else result['samples'][0]['descent'])
    descent['terminal_qpos'][0] += 1.
    write_json(directory/'result.json', result)
    verdict = module.verify(tmp_path)
    assert verdict['evidence_valid'] is False
    assert any('terminal qpos' in error for error in verdict['errors'])


@pytest.mark.parametrize('name', ['result.json', 'observations.npz', 'protocol.json'])
def test_current_receipt_is_bound_to_exact_source_files(tmp_path, name):
    module, directory, _, _ = saved_study(tmp_path, 'full')
    verdict = module.verify(tmp_path)
    validate_outcome_sources(directory, verdict)
    with (directory/name).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='does not match'):
        validate_outcome_sources(directory, verdict)


def test_latest_failed_receipt_never_falls_back_to_success(tmp_path):
    module, directory, result, _ = saved_study(tmp_path, 'full')
    success = module.verify(tmp_path)
    result['seeds'][0]['certifications'][0]['any_full_feasible'] = True
    write_json(directory/'result.json', result)
    failed = module.verify(tmp_path)
    selected = json.loads(latest_outcome_receipt(directory).read_text(encoding='utf-8'))
    assert selected['verification_receipt'] == failed['verification_receipt']
    assert selected['verification_receipt'] != success['verification_receipt']
    assert selected['evidence_valid'] is False


def test_timeout_hash_gate_refuses_missing_or_changed_input(tmp_path):
    path = tmp_path/'input.json'; path.write_text('{}', encoding='utf-8')
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert input_hashes_unchanged(tmp_path, {'input.json': digest})
    path.write_text('changed', encoding='utf-8')
    assert not input_hashes_unchanged(tmp_path, {'input.json': digest})
    path.unlink()
    assert not input_hashes_unchanged(tmp_path, {'input.json': digest})


def test_lp_source_drift_during_replay_invalidates_receipt(tmp_path, monkeypatch):
    module, directory, _, _ = saved_study(tmp_path, 'full')
    from organism_core import static_support
    actual = static_support.solve_support
    def drifting_solve(*args, **kwargs):
        with (directory/'result.json').open('ab') as stream:
            stream.write(b' ')
        return actual(*args, **kwargs)
    monkeypatch.setattr(static_support, 'solve_support', drifting_solve)
    verdict = module.verify(tmp_path)
    assert verdict['evidence_valid'] is False
    assert any('Source evidence changed' in error for error in verdict['errors'])


def test_global_replay_cannot_invent_improvement_by_changing_cached_baseline(tmp_path):
    module, directory, result, _ = saved_study(tmp_path, 'global')
    result['baseline_floor'] = 2.
    write_json(directory/'result.json', result)
    verdict = module.verify(tmp_path)
    assert verdict['evidence_valid'] is False
    assert any('baseline differs' in error for error in verdict['errors'])


def test_all_selected_global_descents_skipped_after_real_sample_lp_replay(tmp_path):
    module, directory, result, arrays = saved_study(tmp_path, 'global')
    result['samples'][0]['descent'] = {'status': 'budget_exhausted'}
    write_json(directory/'result.json', result)
    np.savez(directory/'observations.npz', **{k: v for k, v in arrays.items() if k.startswith('gsamp_')})
    verdict = module.verify(tmp_path)
    assert verdict['evidence_valid'] is True
    assert verdict['scientific_outcome'] == 'inconclusive'
    assert 'descent:budget_exhausted' in verdict['outcome_classification']['stop_reasons']
