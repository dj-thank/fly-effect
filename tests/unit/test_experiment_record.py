import json
from pathlib import Path
import pytest
from scripts.check_experiment import validate

def example():
    return json.loads((Path(__file__).resolve().parents[2]/'examples/experiment.json').read_text(encoding='utf-8'))

def test_published_example_is_valid():
    validate(example())

def test_completion_without_evidence_and_synthetic_biology_rejected():
    record=example();record['status']='completed'
    with pytest.raises(ValueError):validate(record)
    record=example();record['biological_validation']=True
    with pytest.raises(ValueError):validate(record)
