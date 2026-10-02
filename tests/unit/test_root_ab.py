"""Protocol/schema controls run without a body backend or biological data."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import time
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from organism_core.root_ab import CONDITIONS, TRIALS, compare_arrays, derive_xml, write_json

XML = '<mujoco><worldbody><body name="fly"><joint name="root" type="free" stiffness=".4" damping=".02" armature="1e-6"/><body name="leg"><joint name="hinge" stiffness="3" damping="4"/></body></body></worldbody></mujoco>'


def records():
    a = {'neutral_qpos': np.arange(8.), 'recorded_qpos': np.arange(8.),
         'neutral_lengths': np.ones(2), 'maximum_tensions': np.ones(2),
         'initial_integration_state': np.zeros(22), 'support_qpos': np.arange(8.),
         'support_tendon_map': np.ones((7, 2)), 'support_contact_map': np.ones((7, 3)),
         'support_limits': np.ones(2), 'support_friction': np.ones(1), 'support_target': np.ones(7)}
    for name in TRIALS:
        a[name+'_qpos'] = np.tile(a['recorded_qpos'], (251, 1))
        a[name+'_qvel'] = np.zeros((251, 7))
        a[name+'_time_s'] = np.arange(251)*.001
    return a, copy.deepcopy(a)


def test_derivation_only_changes_root_attributes():
    tree = ET.fromstring(derive_xml(XML))
    root = tree.find('./worldbody/body/joint')
    assert root.get('stiffness') == root.get('damping') == '0'
    assert root.get('armature') == '1e-6'
    limb = tree.find('./worldbody/body/body/joint')
    assert limb.get('stiffness') == '3' and limb.get('damping') == '4'
    assert 'stiffness=".4"' in XML


@pytest.mark.parametrize('xml', [
    '<mujoco><worldbody/></mujoco>',
    XML.replace('name="root" ', ''),
    XML.replace('type="free"', 'type="hinge"'),
    XML.replace('<joint name="root" type="free"', '<freejoint name="root"'),
    XML.replace('name="hinge"', 'name="hinge" type="free"'),
    XML.replace('name="hinge"', 'name="hinge" springdamper="1 1"'),
])
def test_unsupported_xml_is_rejected(xml):
    with pytest.raises(ValueError):
        derive_xml(xml)


def test_matching_inputs_allow_different_physical_results():
    a, b = records(); b['support_target'][2] += .255
    b['passive_qpos'][1:, 0] += .1
    assert 'maximum_tensions' in compare_arrays(a, b)


@pytest.mark.parametrize('key', ['neutral_qpos', 'recorded_qpos', 'neutral_lengths',
    'maximum_tensions', 'initial_integration_state', 'support_qpos', 'support_tendon_map',
    'support_contact_map', 'support_limits', 'support_friction'])
def test_unmatched_input_rejected(key):
    a, b = records(); b[key].flat[0] += 1
    with pytest.raises(ValueError, match='Unmatched'):
        compare_arrays(a, b)


@pytest.mark.parametrize('name', TRIALS)
def test_unmatched_trial_initial_state_rejected(name):
    a, b = records(); b[name+'_qpos'][0, 0] += 1
    with pytest.raises(ValueError, match='initial state'):
        compare_arrays(a, b)


def test_both_wrong_initial_states_are_not_matched_controls():
    a, b = records()
    for record in (a, b):
        record['extensor_pulse_qpos'][0, 0] += 1
    with pytest.raises(ValueError, match='shared'):
        compare_arrays(a, b)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_results_rejected(bad):
    a, b = records(); b['support_target'][0] = bad
    with pytest.raises(ValueError, match='Nonfinite'):
        compare_arrays(a, b)


def test_missing_required_observations_rejected():
    a, b = records(); a.pop('support_limits'); b.pop('support_limits')
    with pytest.raises(ValueError, match='Missing'):
        compare_arrays(a, b)


def test_partial_dynamic_window_rejected_even_in_both_conditions():
    a, b = records()
    for record in (a, b):
        record['passive_time_s'] = record['passive_time_s'][:-1]
    with pytest.raises(ValueError, match='250ms'):
        compare_arrays(a, b)


def test_json_does_not_serialize_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path/'result.json', {'value': float('nan')})


def supervisor():
    path = Path(__file__).resolve().parents[2]/'scripts/run_root_ab_study.py'
    spec = importlib.util.spec_from_file_location('root_ab_supervisor', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_outer_deadline_rejects_before_start(tmp_path):
    module = supervisor()
    # This branch is portable; importing POSIX resource is not needed on Windows.
    if sys.platform != 'linux':
        pytest.skip('Linux bounded supervisor')
    with pytest.raises(TimeoutError, match='before stage'):
        module.run_stage([sys.executable, '-c', 'raise RuntimeError()'], root=tmp_path,
                         env=os.environ.copy(), log=tmp_path/'log', deadline=time.monotonic()-1, stage_limit=10)


def test_proc_group_parser_includes_supervisor_and_excludes_other_groups(tmp_path):
    module = supervisor(); mine = os.getpid(); group = mine+100000
    for pid, pgrp, pages in [(mine, mine-1, 10), (group, group, 20), (group+1, group, 30), (group+2, group+2, 1000)]:
        path = tmp_path/str(pid); path.mkdir()
        (path/'stat').write_text(f'{pid} (name with ) brackets) S 1 {pgrp} 1 0')
        (path/'statm').write_text(f'5000 {pages} 0')
    if not hasattr(os, 'sysconf'):
        pytest.skip('Linux RSS page units')
    assert module.group_rss(group, proc=tmp_path) == 60*os.sysconf('SC_PAGE_SIZE')


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group and /proc enforcement')
def test_timeout_kills_child_group_and_retains_log(tmp_path):
    module = supervisor()
    with pytest.raises(TimeoutError) as error:
        module.run_stage([sys.executable, '-c', 'import time; print("started", flush=True); time.sleep(5)'],
            root=tmp_path, env=os.environ.copy(), log=tmp_path/'log', deadline=time.monotonic()+5,
            stage_limit=.3)
    assert (tmp_path/'log').is_file()
    assert error.value.stage_evidence['status'] == 'failed'
    assert error.value.stage_evidence['peak_sampled_group_plus_supervisor_rss_bytes'] > 0


def test_missing_dynamic_rows_rejected_even_with_complete_times():
    a, b = records()
    for record in (a, b):
        record['passive_qpos'] = record['passive_qpos'][:-1]
    with pytest.raises(ValueError, match='250ms'):
        compare_arrays(a, b)
