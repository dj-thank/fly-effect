from types import SimpleNamespace
import numpy as np
import pytest
mj = pytest.importorskip('mujoco')
from organism_core.body import Body
from organism_core.contacts import is_active_contact
from organism_core.support_diagnostics import contact_columns, evaluate_support


def make_body(distance, *, foot=True, dim=3, tilt=False):
    name = 'organism/lf_tarsus1' if foot else 'organism/c_thorax'
    rotation = 'euler="0 10 0"' if tilt else ''
    xml = f'''<mujoco><worldbody>
      <geom name="ground_plane" type="plane" size="1 1 .1" {rotation}/>
      <body pos="0 0 {0.1+distance}"><freejoint/>
      <geom name="{name}" type="sphere" size=".1" mass="1" margin=".02" gap=".01" condim="{dim}"/>
      </body></worldbody></mujoco>'''
    body = Body.__new__(Body)
    body.m = mj.MjModel.from_xml_string(xml); body.d = mj.MjData(body.m)
    if dim == 1: body.m.geom_condim[:] = 1
    mj.mj_forward(body.m, body.d)
    return body


def active_and_gap():
    body = make_body(.005)
    assert body.d.ncon == 1
    # Use the engine's inclusion threshold, independent of historical margin/gap formulas.
    margin = float(body.d.contact[0].includemargin)
    return make_body(margin-.001), make_body(margin+.001)


def test_margin_contact_included_gap_contact_excluded_everywhere():
    for body, active in zip(active_and_gap(), (True,False), strict=True):
        assert body.d.ncon == 1
        assert is_active_contact(body.d.contact[0]) == active
        assert body.contacts()[0]['lf'] == active
        C, mu, records, nonfoot = contact_columns(body.m,body.d)
        assert bool(records) == active and not nonfoot
        assert C.shape[1] == (3 if active else 0)
        if not active: np.testing.assert_array_equal(body.foot_loads(),np.zeros((6,3)))


def test_positive_distance_active_contact_has_force():
    body, _ = active_and_gap()
    assert body.d.contact[0].dist > 0
    assert body.contacts()[1][2] > 0 and body.foot_loads()[0,2] > 0
    assert contact_columns(body.m, body.d)[2][0]['distance_native'] > 0


def test_nonfoot_margin_contact_is_not_hidden():
    body = make_body(.005,foot=False)
    assert contact_columns(body.m,body.d)[3] == ['organism/c_thorax']


def test_frictionless_contact_cannot_gain_tangential_capacity():
    body = make_body(.005, dim=1)
    assert body.d.contact[0].dim == 1
    np.testing.assert_array_equal(contact_columns(body.m, body.d)[1], [0.])


def test_sloped_support_is_explicitly_unsupported():
    body = make_body(.005, tilt=True)
    with pytest.raises(ValueError, match='horizontal'): contact_columns(body.m, body.d)


def test_static_lp_uses_same_contact_and_does_not_mutate_state():
    body, _ = active_and_gap()
    before = body.state()
    trans = SimpleNamespace(matrix=lambda data: np.zeros((body.m.nv,1)))
    result, arrays = evaluate_support(body,trans,np.ones(1))
    assert result['contact_flags_match'] and result['status'] == 'feasible_at_tested_pose'
    assert result['feasible'] and not result['walking_claimed']
    np.testing.assert_array_equal(before,body.state())
    assert arrays['support_contact_map'].shape == (6,3)


def test_gap_support_is_ineligible_not_infeasible():
    _, body = active_and_gap()
    trans = SimpleNamespace(matrix=lambda data: np.zeros((body.m.nv,1)))
    result,_ = evaluate_support(body,trans,np.ones(1))
    assert result['status'] == 'not_eligible' and result['reason'] == 'no_active_foot_contacts'
    assert 'solver_status' not in result


def test_static_support_rejects_motion():
    body = make_body(.005); body.d.qvel[0] = .1
    with pytest.raises(ValueError,match='zero velocity'):
        evaluate_support(body,None,np.ones(1))


def test_recorded_pose_constraint_reconstruction_needs_no_force_solver():
    body, _ = active_and_gap()
    mj.mj_fwdPosition(body.m,body.d)
    assert is_active_contact(body.d.contact[0])
