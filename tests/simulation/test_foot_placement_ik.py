"""Analytic-Jacobian IK controls on a tiny synthetic model, no external data."""
import time
import numpy as np
import pytest
mj = pytest.importorskip('mujoco')
from organism_core.foot_placement import fit_foot_origins


def model():
    return mj.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
    <worldbody><body pos="0 0 2"><freejoint/><geom size=".1" mass="1"/>
      <body><joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".03" mass=".1"/>
        <body name="foot" pos="1 0 0"><geom size=".04" mass=".01"/></body>
      </body></body></worldbody></mujoco>''')


def test_ik_reaches_target_without_model_or_root_changes():
    m = model(); q = m.qpos0.copy(); q_saved = q.copy(); ranges = m.jnt_range.copy()
    target = [[np.cos(.4), 0, 2-np.sin(.4)]]
    final, evidence = fit_foot_origins(m, q, [1], [3], target, deadline=time.monotonic()+5)
    np.testing.assert_array_equal(final[:7], q[:7]); np.testing.assert_array_equal(q, q_saved)
    np.testing.assert_array_equal(m.jnt_range, ranges)
    assert abs(final[7]-.4) < 1e-4 and evidence['maximum_target_error_native'] < 1e-4
    assert evidence['root_unchanged_during_ik'] and evidence['within_original_joint_limits']
    assert not evidence['walking_claimed']


def test_unreachable_target_does_not_relax_joint_limits():
    m = model(); q, evidence = fit_foot_origins(m, m.qpos0, [1], [3], [[-10, 0, 0]], deadline=time.monotonic()+5)
    assert -1 < q[7] < 1 and evidence['maximum_target_error_native'] > 1


@pytest.mark.parametrize('joints', [[0], [1, 1], [9], []])
def test_ik_refuses_root_or_invalid_joint_sets(joints):
    m = model()
    with pytest.raises(ValueError): fit_foot_origins(m, m.qpos0, joints, [3], [[1, 0, 2]], deadline=time.monotonic()+5)


def test_ik_deadline_is_not_a_physics_result():
    m = model()
    with pytest.raises(TimeoutError): fit_foot_origins(m, m.qpos0, [1], [3], [[1, 0, 2]], deadline=time.monotonic()-1)
