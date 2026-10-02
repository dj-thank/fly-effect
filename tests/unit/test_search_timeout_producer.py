"""Run producers retain completed witnesses when a later certify call times out.

The synthetic backend controls solver outcomes; run bookkeeping, model identity
audits, input receipts, and JSON/NPZ output use their real implementations.
"""
import hashlib
import importlib.metadata
import json

import numpy as np
import pytest

mj = pytest.importorskip('mujoco')

from organism_core import body as body_module
from organism_core import full_descent, global_sample
from organism_core.passive_posture import FEASIBLE_POSE, FIELDS
from organism_core.root_ab import audit_models, derive_xml, write_json
from organism_core.verification_receipt import (
    outcome_source_hashes, write_outcome_receipt,
)


def _support_inputs():
    return {
        'support_tendon_map': np.zeros((7, 90)),
        'support_contact_map': np.zeros((7, 0)),
        'support_target': np.zeros(7),
        'support_limits': np.ones(90),
        'support_friction': np.zeros(0),
    }


def _prepare_workspace(workspace):
    # A real free-root, unactuated 90-tendon model passes the unchanged audits.
    xml = (
        '<mujoco><worldbody><site name="anchor" pos="0 0 .3"/>'
        '<body><joint name="root" type="free" stiffness=".4" '
        'damping=".02" armature=".000001"/>'
        '<geom type="sphere" size=".1" mass="1"/>'
        '<site name="end" pos="0 0 .1"/>'
        '<body pos=".2 0 0"><joint name="passive" type="hinge" '
        'range="-1 1" limited="true" stiffness="1"/>'
        '<geom type="sphere" size=".05" mass=".1"/>'
        '</body></body></worldbody><tendon>'
        + ''.join(
            f'<spatial name="t{i}"><site site="anchor"/>'
            '<site site="end"/></spatial>' for i in range(90))
        + '</tendon></mujoco>'
    )
    legacy = mj.MjModel.from_xml_string(xml)
    candidate_xml = derive_xml(xml)
    candidate = mj.MjModel.from_xml_string(candidate_xml)
    digest = hashlib.sha256(xml.encode()).hexdigest()
    q0 = legacy.qpos0.copy()
    for name in ('source-study', 'foot-placement', 'passive-posture',
                 'iterative-shift', 'full-descent'):
        (workspace / name).mkdir()
    write_json(workspace / 'protocol.json', {'synthetic_test': True})
    write_json(workspace / 'source-study/result.json', {
        'body_sha256': digest,
        'units': {'length': 'mm'},
        'maximum_tensions_native': np.ones(90).tolist(),
    })
    np.savez_compressed(workspace / 'source-study/observations.npz',
                        neutral_qpos=q0)
    write_json(workspace / 'foot-placement/result.json', {
        'candidates': [{'force_accounting': {'root_dofs': list(range(6))}}],
    })
    write_json(workspace / 'foot-placement/model-receipt.json', {
        'parent_body_sha256': digest,
        'candidate_xml_sha256': hashlib.sha256(candidate_xml.encode()).hexdigest(),
        'compiled_invariants': audit_models(legacy, candidate),
        'model_identity': {'synthetic_model': digest},
    })
    np.savez_compressed(workspace / 'foot-placement/observations.npz',
                        pose_00_support_tendon_map=np.zeros((7, 90)))
    write_json(workspace / 'passive-posture/result.json', {'candidates': []})
    np.savez_compressed(workspace / 'passive-posture/observations.npz',
                        pose_00_passive_dofs=np.array([6], dtype=int),
                        pose_00_passive_qpos_adrs=np.array([7], dtype=int),
                        pose_00_passive_ranges=np.array([[-1., 1.]]),
                        pose_00_passive_stiffness=np.array([1.]))
    write_json(workspace / 'iterative-shift/result.json', {
        'candidates': [{
            'index': 0,
            'lineages': [{'iterates': [{
                'driver_margin_t': 0., 'driver_sample': 0,
                'samples': [{'index': 0}],
            }]}],
        }],
    })
    write_json(workspace / 'iterative-shift/verification.json',
               {'evidence_valid': True})
    np.savez_compressed(workspace / 'iterative-shift/observations.npz',
                        pose_00_lineage_00_it_00_depth_00_support_qpos=q0)
    full_directory = workspace / 'full-descent'
    write_json(full_directory / 'result.json', {
        'seeds': [{'status': 'completed', 'descent': {'final_R': 0.8}}],
    })
    np.savez_compressed(full_directory / 'observations.npz', neutral_qpos=q0)
    write_json(full_directory / 'protocol.json', {'protocol': full_descent.PROTOCOL})
    write_outcome_receipt(full_directory, {
        'evidence_valid': True,
        'outcome_classification': {'version': 2},
        'source_sha256': outcome_source_hashes(full_directory),
    })
    return xml, legacy, digest, q0


def _install_backend(monkeypatch, module, workspace, *, failure='timeout'):
    xml, legacy, digest, q0 = _prepare_workspace(workspace)
    if module is full_descent:
        # run owns creation of its output directory.
        for path in (workspace / 'full-descent').iterdir():
            path.unlink()
        (workspace / 'full-descent').rmdir()

    def init_body(self, *_args):
        if failure == 'before_input_link':
            raise TimeoutError('synthetic backend setup timeout')
        self.xml, self.m, self.digest = xml, legacy, digest

    monkeypatch.setattr(body_module.Body, '__init__', init_body)
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda name: 'synthetic-test' if name == 'flygym'
                        else real_version(name))

    def descend(*_args):
        record = {
            'solver_calls': 1, 'probes': 1, 'unresolved_evals': 0,
            'status': 'step_floor_reached', 'accepted_steps': 2,
            'initial_R': 0.9, 'initial_status': 'infeasible', 'final_R': 0.5,
            'accepted': [
                {'accepted': index, 'R': residual}
                for index, residual in enumerate([0.9, 0.7, 0.5])
            ],
            'terminal_qpos': q0.tolist(), 'terminal_tilt_rad': 0.,
        }
        arrays = {}
        for index in range(3):
            prefix = f'acc_{index:04d}_'
            arrays[prefix + 'qpos'] = q0.copy()
            arrays.update({prefix + key: value for key, value in _support_inputs().items()})
        return record, arrays, q0.copy(), None

    monkeypatch.setattr(module, '_descend_full', descend)
    if module is global_sample:
        monkeypatch.setattr(module, '_sample_qpos', lambda *_args: q0.copy())
        monkeypatch.setattr(module, '_probe_full', lambda *_args: (
            0.9, 'infeasible',
            {'index': 0, 'foot_legs': ['lf'], 'inputs': _support_inputs()},
            {'solver': 1},
        ))
    calls = []

    def certify(*args):
        prefix = args[-1]
        calls.append(prefix)
        if len(calls) == 2:
            source = workspace / 'source-study/result.json'
            if failure == 'input_drift':
                source.write_text(source.read_text() + '\n', encoding='utf-8')
            elif failure == 'missing_file':
                source.unlink()
            raise TimeoutError('synthetic second certification timeout')
        samples, arrays = [], {}
        for index, depth in enumerate(module.PROTOCOL['depths_native']):
            samples.append({
                'index': index, 'depth_native': depth, 'evidence_saved': True,
                'root_balance': {'feasible': True, 'status': 'feasible'},
                'support': {'status': FEASIBLE_POSE},
                'passive_subsystem': {'status': 'feasible'},
                'margin': {'status': 'measured'},
            })
            sample_prefix = prefix + f'depth_{index:02d}_'
            arrays[sample_prefix + 'qpos'] = q0.copy()
            arrays.update({sample_prefix + key: value
                           for key, value in _support_inputs().items()})
        return {
            'samples': samples, 'any_full_feasible': True,
            'any_passive_feasible': True, 'solver_calls': 1,
        }, arrays

    monkeypatch.setattr(module, '_certify', certify)
    return calls


def _read_output(workspace, module):
    directory = workspace / ('full-descent' if module is full_descent else 'global-sample')
    return json.loads((directory / 'result.json').read_text(encoding='utf-8')), directory


@pytest.mark.parametrize('module', [full_descent, global_sample], ids=['full', 'global'])
def test_run_saves_first_completed_witness_before_second_certification_times_out(tmp_path, monkeypatch, module):
    calls = _install_backend(monkeypatch, module, tmp_path)
    result = module.run(tmp_path)
    saved, directory = _read_output(tmp_path, module)
    assert saved == result
    assert result['status'] == 'timeout'
    assert result['scientific_outcome'] == 'found_full_support_pose'
    assert result['outcome_classification']['search_complete'] is False
    assert 'worker_wall_budget_exhausted' in result['outcome_classification']['stop_reasons']
    records = result['seeds'] if module is full_descent else result['samples']
    certs = (records[0]['certifications'] if module is full_descent
             else records[0]['descent']['certifications'])
    assert [cert['status'] for cert in certs] == ['completed', 'pending']
    assert [cert['accepted_index'] for cert in certs] == [2, 1]
    assert len(calls) == 2
    with np.load(directory / 'observations.npz', allow_pickle=False) as arrays:
        witnesses = [key for key in arrays.files if module.WITNESS.match(key)]
        assert len(witnesses) == len(module.PROTOCOL['depths_native'])
        for sample in certs[0]['samples']:
            key = calls[0] + f"depth_{sample['index']:02d}_support_tendon_map"
            assert key in arrays.files
            np.testing.assert_array_equal(arrays[key], _support_inputs()[FIELDS[0]])
        assert not any(key.startswith(calls[1]) for key in arrays.files)


@pytest.mark.parametrize('module', [full_descent, global_sample], ids=['full', 'global'])
@pytest.mark.parametrize('failure', ['input_drift', 'missing_file', 'before_input_link'])
def test_run_timeout_without_unchanged_linked_inputs_is_invalid(tmp_path, monkeypatch, module, failure):
    calls = _install_backend(monkeypatch, module, tmp_path, failure=failure)
    result = module.run(tmp_path)
    saved, directory = _read_output(tmp_path, module)
    assert saved == result
    assert result['status'] == 'error'
    assert result['scientific_outcome'] == 'invalid_experiment'
    assert 'Input evidence changed' in result['error']
    assert (directory / 'observations.npz').is_file()
    if failure == 'before_input_link':
        assert calls == []
        assert result.get('input_sha256') is None
    else:
        assert len(calls) == 2
        records = result['seeds'] if module is full_descent else result['samples']
        certs = (records[0]['certifications'] if module is full_descent
                 else records[0]['descent']['certifications'])
        assert certs[0]['status'] == 'completed'
        assert certs[0]['samples'][0]['support']['status'] == FEASIBLE_POSE
