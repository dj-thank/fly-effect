"""Malformed records must not bypass declared experiment resource budgets."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest
from scripts.check_experiment import validate, read_record

ROOT = Path(__file__).resolve().parents[2]


def example():
    return json.loads((ROOT/'examples/experiment.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('field', ['wall_seconds', 'memory_mib'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), True,
    False, 0, -1, '60', None, [], {}, 10**1000])
def test_invalid_budgets(field, bad):
    record = example(); record['budget'][field] = bad
    with pytest.raises(ValueError): validate(record)


@pytest.mark.parametrize('key,bad', [('schema_version', True), ('schema_version', 1.),
    ('seed', True), ('seed', -1), ('seed', 0.), ('question', ''), ('hypothesis', ' '),
    ('success_criterion', None), ('falsification', []), ('units', {}), ('units', []),
    ('units', {'time': ''}), ('budget', None), ('controls', 'control'), ('controls', []),
    ('controls', ['']), ('evidence', 'proof'), ('evidence', [None]), ('evidence', [{}]),
    ('biological_validation', 0), ('biological_validation', 'false')])
def test_strict_field_types(key, bad):
    record = example(); record[key] = bad
    with pytest.raises(ValueError): validate(record)


@pytest.mark.parametrize('record', [None, [], 1, 'record'])
def test_top_level_object_required(record):
    with pytest.raises(ValueError): validate(record)


def test_missing_budget_field_is_validation_error():
    record = example(); del record['budget']['memory_mib']
    with pytest.raises(ValueError): validate(record)


def test_nonfinite_extra_parameters_rejected():
    record = example(); record['parameters'] = {'x': [float('nan')]}
    with pytest.raises(ValueError): validate(record)


def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path/'record.json'
    text = json.dumps(example()).replace('"seed": 0', '"seed": 0, "seed": 1')
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'): read_record(path)


def test_valid_completed_record_and_structured_evidence():
    record = example(); record['status'] = 'completed'
    record['evidence'] = ['work/run/result.json', {'path': 'work/run/observations.npz'}]
    before = copy.deepcopy(record); validate(record); assert record == before


def test_published_examples_and_multiple_file_cli():
    paths = sorted((ROOT/'examples').glob('*.json'))
    for path in paths: read_record(path)
    result = subprocess.run([sys.executable, str(ROOT/'scripts/check_experiment.py'),
                             *map(str, paths)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_rejects_overflow_budget_without_traceback(tmp_path):
    path = tmp_path/'bad.json'; path.write_text(json.dumps(example()).replace('60', '1e999'), encoding='utf-8')
    result = subprocess.run([sys.executable, str(ROOT/'scripts/check_experiment.py'), str(path)], capture_output=True, text=True)
    assert result.returncode == 2 and 'Finite JSON' in result.stderr
    assert 'Traceback' not in result.stderr


def test_cyclic_or_excessively_deep_record_rejected():
    record = example(); record['cycle'] = record
    with pytest.raises(ValueError, match='nesting'): validate(record)
