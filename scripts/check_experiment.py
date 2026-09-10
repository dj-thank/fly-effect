"""Validate an experiment record's structure, not its scientific truth."""
import argparse
import json
import math
from pathlib import Path


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _json_values(value, depth=0):
    if depth > 100:
        raise ValueError('Experiment record nesting exceeds 100 levels')
    if type(value) is float and not math.isfinite(value):
        raise ValueError('Finite JSON numbers required')
    if value is None or type(value) in (str, bool, int, float):
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        raise ValueError('JSON-compatible values and string keys required')
    for child in children:
        _json_values(child, depth + 1)


def validate(record):
    if not isinstance(record, dict):
        raise ValueError('Experiment record must be an object')
    _json_values(record)
    required = ('schema_version', 'kind', 'status', 'question', 'hypothesis', 'seed',
                'units', 'budget', 'controls', 'success_criterion', 'falsification',
                'evidence', 'biological_validation')
    for key in required:
        if key not in record:
            raise ValueError('Missing ' + key)
    if type(record['schema_version']) is not int or record['schema_version'] != 1:
        raise ValueError('Unknown schema')
    if record['kind'] not in ('synthetic', 'reference', 'organism_candidate'):
        raise ValueError('Unknown experiment kind')
    if record['status'] not in ('planned', 'completed', 'failed'):
        raise ValueError('Unknown status')
    for key in ('question', 'hypothesis', 'success_criterion', 'falsification'):
        if not _text(record[key]):
            raise ValueError(key + ' must be nonempty text')
    if type(record['seed']) is not int or record['seed'] < 0:
        raise ValueError('Nonnegative integer seed required')
    units = record['units']
    if not (_text(units) or (isinstance(units, dict) and units
                            and all(_text(k) and _text(v) for k, v in units.items()))):
        raise ValueError('Named units or a nonempty unit description required')
    budget = record['budget']
    if not isinstance(budget, dict):
        raise ValueError('Resource budget must be an object')
    for key in ('wall_seconds', 'memory_mib'):
        value = budget.get(key)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('Positive finite numeric resource budget required: ' + key)
    controls = record['controls']
    if not isinstance(controls, list) or not controls or not all(_text(c) for c in controls):
        raise ValueError('Nonempty control descriptions required')
    evidence = record['evidence']
    if (not isinstance(evidence, list)
            or not all(_text(e) or (isinstance(e, dict) and bool(e)) for e in evidence)):
        raise ValueError('Evidence must be a list of nonempty locators or structured references')
    if record['status'] == 'completed' and not evidence:
        raise ValueError('Completed experiment needs evidence locators')
    if type(record['biological_validation']) is not bool:
        raise ValueError('biological_validation must be a boolean')
    if record['kind'] == 'synthetic' and record['biological_validation']:
        raise ValueError('Synthetic experiment cannot claim biological validation')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate experiment JSON key: ' + key)
        result[key] = value
    return result


def read_record(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    validate(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path, nargs='+')
    args = parser.parse_args()
    try:
        for path in args.record:
            read_record(path)
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))
    print('Structure valid; evidence and scientific claims require review.')


if __name__ == '__main__':
    main()
