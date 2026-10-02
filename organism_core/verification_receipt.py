"""Append outcome re-verification receipts without replacing prior evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid


def outcome_source_hashes(directory):
    digests = {}
    for name in ('result.json', 'observations.npz', 'protocol.json'):
        with (Path(directory)/name).open('rb') as stream:
            digests[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digests


def input_hashes_unchanged(directory, expected):
    if not isinstance(expected, dict) or not expected:
        return False
    try:
        for relative, digest in expected.items():
            with (Path(directory)/relative).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    return False
    except OSError:
        return False
    return True


def validate_outcome_sources(directory, verification):
    if (verification.get('verification_method') != 'saved_input_lp_replay'
            or verification.get('source_sha256') != outcome_source_hashes(directory)):
        raise ValueError('Full-descent outcome receipt does not match current source evidence')


def write_outcome_receipt(directory, verdict):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name = f'verification-outcomes-v2-{stamp}-{uuid.uuid4().hex}.json'
    receipt = dict(verdict, verification_receipt=name,
                   verification_method='saved_input_lp_replay')
    with (Path(directory)/name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt


def latest_outcome_receipt(directory):
    paths = sorted(Path(directory).glob('verification-outcomes-v2-*.json'))
    if not paths:
        raise ValueError('Full-descent evidence needs current outcome re-verification')
    return paths[-1]
