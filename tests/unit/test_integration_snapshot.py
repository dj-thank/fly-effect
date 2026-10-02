"""Integration receipts attest declared source bytes, not scientific success."""
import json
from pathlib import Path
import pytest
from scripts.check_integration import verify, unique_object
ROOT = Path(__file__).resolve().parents[2]


def manifest():
    return json.loads((ROOT/'docs/integration-manifest.json').read_text())


def test_reviewed_source_snapshot():
    result = verify(ROOT, manifest())
    assert result['files_checked'] == 19
    assert not result['biological_validation'] and not result['main_merge_claimed']


@pytest.mark.parametrize('mutation', ['schema_bool', 'source_ref', 'missing_protected', 'digest',
                                      'path_traversal', 'duplicate_path', 'empty_section'])
def test_integration_mismatch_is_rejected(mutation):
    record = manifest()
    if mutation == 'schema_bool': record['schema'] = True
    elif mutation == 'source_ref': record['sources']['runtime'] = 'main'
    elif mutation == 'missing_protected': record['protected'].pop('ACCEPTANCE.json')
    elif mutation == 'digest': record['protected']['ACCEPTANCE.json'] = '0'*64
    elif mutation == 'empty_section': record['retained_research'] = {}
    elif mutation == 'duplicate_path': record['retained_runtime']['ACCEPTANCE.json'] = record['protected']['ACCEPTANCE.json']
    else: record['retained_runtime']['../escape'] = '0'*64
    with pytest.raises(ValueError): verify(ROOT, record)


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError):
        json.loads('{"schema": 1, "schema": 2}', object_pairs_hook=unique_object)
