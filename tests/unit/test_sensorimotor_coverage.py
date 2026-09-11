"""Runtime accounting tests use synthetic IDs and do not assert walking."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from organism_core.sensorimotor_coverage import summarize
from organism_core.connectivity_audit import ratio


def inputs():
    rows = [{'internal_index': i, 'body_id': 100+i,
             'required_for_walking': i in (2, 3),
             'anatomically_calibrated': False,
             'joint': 'knee' if i == 2 else None,
             'status': 'inferred_mechanical_transfer' if i == 2 else 'unresolved_target' if i == 3 else 'outside_current_walking_scope'}
            for i in (2, 3, 4)]
    return {'ids': np.arange(6)+100, 'motor_indices': np.array([2, 3, 4]),
            'rows': rows, 'assignments': {2: (0, 1)}, 'sensory_ids': np.array([100, 105]),
            'joint_names': ['knee'],
            'accounting': {'per_motor_spikes': {'2': 5, '3': 7, '4': 9},
                           'mapped_spikes': 5, 'unresolved_spikes': 7, 'outside_spikes': 9},
            'motor_spikes': 21}


def test_registry_walking_scope_assignment_and_activity_are_different():
    data = inputs(); result = summarize(**data)
    assert result['motor_registry'] == ratio(3, 6)
    assert result['nonmotor_registry'] == ratio(3, 6)
    assert result['walking_scope'] == ratio(2, 3)
    assert result['mapped_walking_motors'] == ratio(1, 2)
    assert result['unresolved_walking_motors'] == ratio(1, 2)
    assert result['outside_walking_scope'] == ratio(1, 3)
    assert result['anatomically_calibrated_mapped'] == ratio(0, 1)
    assert result['active_motor_neurons'] == ratio(3, 3)
    assert result['body_input_neurons'] == ratio(2, 6)
    assert result['motor_events']['total'] == 21
    assert result['walking_claimed'] is False
    assert data['accounting'] == inputs()['accounting']


def test_silence_is_not_missing_wiring():
    data = inputs(); data['accounting'] = {'per_motor_spikes': {'2': 0, '3': 0, '4': 0},
                                         'mapped_spikes': 0, 'unresolved_spikes': 0, 'outside_spikes': 0}
    data['motor_spikes'] = 0
    result = summarize(**data)
    assert result['active_motor_neurons'] == ratio(0, 3)
    assert result['mapped_walking_motors'] == ratio(1, 2)


def test_no_motor_scope_has_null_percent_not_failure_or_zero_coverage():
    result = summarize(np.array([1, 2]), np.array([], dtype=int), [], {}, np.array([1]),
                       {'per_motor_spikes': {}, 'mapped_spikes': 0, 'unresolved_spikes': 0, 'outside_spikes': 0},
                       joint_names=[], motor_spikes=0)
    assert result['mapped_walking_motors']['percent'] is None
    assert result['active_motor_neurons']['percent'] is None


@pytest.mark.parametrize('mutation', [
    'missing_row', 'duplicate_row', 'wrong_body_id', 'boolean_index', 'boolean_body_id',
    'nonboolean_scope', 'nonboolean_calibration', 'outside_assignment', 'missing_assignment',
    'missing_joint', 'wrong_joint', 'bad_joint_index', 'bad_polarity', 'boolean_joint',
    'sensory_motor_overlap', 'sensory_unknown', 'sensory_duplicate', 'sensory_float',
    'missing_counter', 'unknown_counter', 'negative_counter', 'boolean_counter',
    'mapped_total', 'outside_total', 'unresolved_total', 'neural_total', 'empty_status',
    'duplicate_joints', 'bad_joint_names'])
def test_inconsistent_runtime_coverage_is_rejected(mutation):
    data = inputs()
    if mutation == 'missing_row': data['rows'].pop()
    if mutation == 'duplicate_row': data['rows'].append(copy.deepcopy(data['rows'][0]))
    if mutation == 'wrong_body_id': data['rows'][0]['body_id'] = 103
    if mutation == 'boolean_index': data['rows'][0]['internal_index'] = True
    if mutation == 'boolean_body_id': data['rows'][0]['body_id'] = True
    if mutation == 'nonboolean_scope': data['rows'][0]['required_for_walking'] = 1
    if mutation == 'nonboolean_calibration': data['rows'][0]['anatomically_calibrated'] = 0
    if mutation == 'outside_assignment': data['assignments'][4] = (0, 1)
    if mutation == 'missing_assignment': data['assignments'].clear()
    if mutation == 'missing_joint': data['rows'][0]['joint'] = None
    if mutation == 'wrong_joint': data['rows'][0]['joint'] = 'other'
    if mutation == 'bad_joint_index': data['assignments'][2] = (10, 0)
    if mutation == 'bad_polarity': data['assignments'][2] = (0, 3)
    if mutation == 'boolean_joint': data['assignments'][2] = (True, 0)
    if mutation == 'sensory_motor_overlap': data['sensory_ids'] = np.array([102])
    if mutation == 'sensory_unknown': data['sensory_ids'] = np.array([999])
    if mutation == 'sensory_duplicate': data['sensory_ids'] = np.array([100, 100])
    if mutation == 'sensory_float': data['sensory_ids'] = np.array([100.])
    if mutation == 'missing_counter': data['accounting']['per_motor_spikes'].pop('4')
    if mutation == 'unknown_counter': data['accounting']['per_motor_spikes']['6'] = 0
    if mutation == 'negative_counter': data['accounting']['per_motor_spikes']['2'] = -1
    if mutation == 'boolean_counter': data['accounting']['per_motor_spikes']['2'] = True
    if mutation == 'mapped_total': data['accounting']['mapped_spikes'] = 6
    if mutation == 'outside_total': data['accounting']['outside_spikes'] = 10
    if mutation == 'unresolved_total': data['accounting']['unresolved_spikes'] = 8
    if mutation == 'neural_total': data['motor_spikes'] = 20
    if mutation == 'empty_status': data['rows'][0]['status'] = ''
    if mutation == 'duplicate_joints': data['joint_names'] = ['knee', 'knee']
    if mutation == 'bad_joint_names': data['joint_names'] = [None]
    with pytest.raises(ValueError): summarize(**data)


def report_only_engine():
    # Execute the actual report method without importing optional simulation engines.
    path = Path(__file__).resolve().parents[2]/'organism_core/engine.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Engine')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'report')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {'np': np, 'hashlib': hashlib, '__package__': 'organism_core'}
    exec(compile(module, str(path), 'exec'), scope)
    return scope['report']


def fake_engine():
    values = inputs()
    spikes = np.repeat([2, 3, 4], [5, 7, 9])
    return SimpleNamespace(
        observation=lambda: {'spike_i': spikes},
        brain=SimpleNamespace(ids=values['ids'], motor=values['motor_indices'], edges=np.zeros(8)),
        motor_map=SimpleNamespace(rows=values['rows'], assignments=values['assignments'], state=lambda: values['accounting']),
        sensory_ids=values['sensory_ids'], sensory=values['sensory_ids'], tick=10, input_count=2,
        identity={'kind': 'synthetic-test-only'}, rng=np.random.default_rng(1), load_rng=np.random.default_rng(2),
        load_receptors=SimpleNamespace(counts=np.array([0, 0])),
        body=SimpleNamespace(active_joint_names=values['joint_names'], m=SimpleNamespace(nu=0),
                             muscles=SimpleNamespace(activation=np.zeros(2))),
        settings={'muscle_model': 'antagonist', 'sensory_mode': 'six_leg', 'input_enabled': True},
        frame_force=[0.])


def test_actual_engine_report_preserves_old_fields_and_checks_new_coverage():
    engine = fake_engine(); report = report_only_engine()(engine)
    assert report['motor_spikes'] == 21
    assert report['motor_coverage'] == {'total': 3, 'required': 2, 'mapped': 1, 'unresolved_required': 1}
    assert report['sensorimotor_coverage']['mapped_walking_motors'] == ratio(1, 2)
    assert report['sensorimotor_coverage']['body_input_enabled'] is True
    assert report['walking_passed'] is False
    engine.settings['input_enabled'] = False
    assert report_only_engine()(engine)['sensorimotor_coverage']['body_input_enabled'] is False


def test_actual_report_refuses_inconsistent_monitor_total():
    engine = fake_engine(); engine.observation = lambda: {'spike_i': np.array([], dtype=int)}
    with pytest.raises(ValueError, match='neural spike'): report_only_engine()(engine)
