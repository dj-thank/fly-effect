import numpy as np
import pytest
mj = pytest.importorskip('mujoco')
from organism_core.mechanics import (DEMO_XML, audit_virtual_work, pulling_force,
                                    site_tendon_paths, synthetic_audit)
from organism_core.calibration import run_mechanics_demo


@pytest.fixture
def fixture():
    model = mj.MjModel.from_xml_string(DEMO_XML)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data, site_tendon_paths(model)


def test_analytic_torque_and_negative_control():
    result = synthetic_audit()
    assert result['passed'] and result['reversed_force_rejected']
    assert result['analytical_match'] and result['internal_base_force_zero']
    assert not result['biological_validation'] and not result['walking_claimed']


@pytest.mark.parametrize('angle', [-.4, 0., .3, .9])
def test_virtual_work_at_translated_rotated_free_body(fixture, angle):
    model, data, paths = fixture
    data.qpos[:3] = [1., -.3, 2.]
    data.qpos[3:7] = [.5, .5, .5, .5]
    data.qpos[-1] = angle
    data.qvel[:] = np.linspace(-.2, .4, model.nv)
    mj.mj_forward(model, data)
    spec = mj.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mj.mj_stateSize(model, spec)); mj.mj_getState(model, data, before, spec)
    result = audit_virtual_work(model, data, paths, [2.])
    after = np.empty_like(before); mj.mj_getState(model, data, after, spec)
    np.testing.assert_array_equal(before, after)
    assert result['passed']
    assert result['mechanical_power_native'] == pytest.approx(result['finite_difference_power_native'], abs=1e-7)
    np.testing.assert_allclose(result['generalized_force_native'][:6], 0., atol=1e-12)


@pytest.mark.parametrize('forces', [[], [1., 2.], [[1.]], [-1.], [np.nan], [np.inf]])
def test_invalid_tensions_do_not_silently_zip(fixture, forces):
    model, data, paths = fixture
    with pytest.raises(ValueError): pulling_force(model, data, paths, forces)


@pytest.mark.parametrize('sites', [[-1, 1], [0, 9999], [0], [0, 0], [.0, 1.], [[0, 1]]])
def test_invalid_sites_rejected(fixture, sites):
    model, data, _ = fixture
    with pytest.raises(ValueError): pulling_force(model, data, [sites], [1.])


def test_degenerate_geometry_rejected_even_at_zero_tension(fixture):
    model, data, paths = fixture
    data.site_xpos[1] = data.site_xpos[0]
    with pytest.raises(ValueError, match='degenerate'): pulling_force(model, data, paths, [0.])


def test_force_does_not_mutate_applied_force_or_tensions(fixture):
    model, data, paths = fixture
    data.qfrc_applied[:] = .123
    tensions = np.array([2.])
    pulling_force(model, data, paths, tensions)
    np.testing.assert_array_equal(data.qfrc_applied, np.full(model.nv, .123))
    np.testing.assert_array_equal(tensions, [2.])


def test_fixed_tendon_cannot_be_misread_as_site_ids():
    xml = DEMO_XML.replace('<spatial name="pull"><site site="origin"/><site site="insertion"/></spatial>',
                           '<fixed name="pull"><joint joint="hinge" coef="1"/></fixed>')
    model = mj.MjModel.from_xml_string(xml)
    with pytest.raises(ValueError, match='site-routed'): site_tendon_paths(model)


@pytest.mark.parametrize('epsilon', [0., -1., np.nan, np.inf])
def test_invalid_finite_difference_steps(fixture, epsilon):
    model, data, paths = fixture
    with pytest.raises(ValueError): audit_virtual_work(model, data, paths, [2.], epsilon=epsilon)


def test_empty_paths_rejected(fixture):
    model, data, _ = fixture
    with pytest.raises(ValueError): pulling_force(model, data, [], [])


def test_mechanics_demo_writes_strict_receipt_and_never_overwrites(tmp_path):
    import json
    out = tmp_path / 'demo'
    result = run_mechanics_demo(out)
    saved = json.loads((out / 'result.json').read_text())
    assert result == saved and saved['passed']
    with pytest.raises(FileExistsError): run_mechanics_demo(out)
