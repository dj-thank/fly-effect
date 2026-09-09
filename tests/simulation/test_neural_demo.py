import pytest
pytest.importorskip("brian2")
from fly_effect import demo

def test_synthetic_demo_resumes(tmp_path):
    result=demo(tmp_path/"demo")
    assert result["checkpoint_resume_equal"]
    assert result["spike_count"]>=3
    assert not result["walking_claimed"]
