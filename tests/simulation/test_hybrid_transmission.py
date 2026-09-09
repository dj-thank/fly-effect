from types import SimpleNamespace
import numpy as np
import pytest
mj = pytest.importorskip('mujoco')
from organism_core.mechanics import DEMO_XML, pulling_force
from organism_core.tendon_coupling import HybridMuscles
from organism_core.transmission import SiteTendonTransmission


@pytest.mark.parametrize('angle',[-.3,0.,.4])
def test_live_hybrid_command_uses_csr_without_losing_legacy_torque(angle):
    m = mj.MjModel.from_xml_string(DEMO_XML); d = mj.MjData(m)
    d.qpos[-1] = angle; mj.mj_forward(m,d)
    h = HybridMuscles.__new__(HybridMuscles)
    h.body = SimpleNamespace(m=m,d=d,active_dofs=np.array([6]))
    h.transmission = SiteTendonTransmission(m); h.sites = h.transmission.paths
    h.unapplied_dofs = np.arange(6); h.max_base_residual = 0.
    h.tendons = SimpleNamespace(command=lambda data: np.array([2.]))
    h.legacy = SimpleNamespace(command=lambda q,v: np.array([.25]))
    expected = pulling_force(m,d,h.sites,[2.])[6]+.25
    actual = h.command(np.array([angle]),np.array([0.]))
    np.testing.assert_allclose(actual,[expected],atol=1e-12)
    assert h.max_base_residual <= 1e-12
    actual[0] = 999
    assert h.last_torque[0] != 999


def test_unrepresented_force_cannot_be_silently_discarded():
    m = mj.MjModel.from_xml_string(DEMO_XML); d = mj.MjData(m); mj.mj_forward(m,d)
    h = HybridMuscles.__new__(HybridMuscles)
    h.body = SimpleNamespace(m=m,d=d,active_dofs=np.array([],dtype=int))
    h.transmission = SiteTendonTransmission(m); h.unapplied_dofs = np.arange(m.nv)
    h.max_base_residual = 0.
    h.tendons = SimpleNamespace(command=lambda data: np.array([2.]))
    h.legacy = SimpleNamespace(command=lambda q,v: np.array([]))
    with pytest.raises(ValueError,match='unrepresented'):
        h.command(np.array([]),np.array([]))
