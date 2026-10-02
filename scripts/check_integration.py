"""Check a reviewed integration snapshot, not biological validity or git ancestry."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

PROTECTED = {'organism_core/brain.py', 'organism_core/graph_lock.json', 'ACCEPTANCE.json',
             'organism_core/body.py', 'organism_core/static_support.py', 'organism_core/foot_placement.py'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate integration manifest key: ' + key)
        result[key] = value
    return result


def verify(root, manifest):
    root = Path(root).resolve()
    if (not isinstance(manifest, dict) or type(manifest.get('schema')) is not int
            or manifest['schema'] != 1):
        raise ValueError('Invalid integration manifest schema')
    sources = manifest.get('sources')
    if (not isinstance(sources, dict) or set(sources) != {'runtime', 'research'}
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{40}', v)
                   for v in sources.values())):
        raise ValueError('Exact runtime and research source commits required')
    if not isinstance(manifest.get('protected'), dict) or set(manifest['protected']) != PROTECTED:
        raise ValueError('Protected integration file set changed')
    seen = set(); verified = {}
    for section in ('protected', 'retained_runtime', 'retained_research'):
        entries = manifest.get(section)
        if not isinstance(entries, dict) or not entries:
            raise ValueError('Nonempty integration hash section required: ' + section)
        for name, expected in entries.items():
            if not isinstance(name, str):
                raise ValueError('Integration path must be a string')
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or ':' in name or str(relative) != name or not relative.parts
                    or name.casefold() in seen):
                raise ValueError('Unsafe or duplicate integration path: ' + name)
            seen.add(name.casefold())
            path = (root/name).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Integration path escapes checkout: ' + name)
            if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
                raise ValueError('Invalid expected SHA256: ' + name)
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError('Integration source hash mismatch: ' + name)
            verified[name] = actual
    return {'status': 'verified', 'files_checked': len(verified), 'sources': sources,
            'sha256': verified, 'biological_validation': False, 'main_merge_claimed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); root = args.root.resolve()
    out = root/'work/validation/integration.json'
    try:
        manifest = json.loads((root/'docs/integration-manifest.json').read_text(encoding='utf-8'),
                              object_pairs_hook=unique_object)
        result = verify(root, manifest)
    except (OSError, ValueError, TypeError) as error:
        result = {'status': 'failed', 'error': str(error), 'biological_validation': False, 'main_merge_claimed': False}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        parser.error(str(error))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('Integration source hashes verified:', result['files_checked'])


if __name__ == '__main__':
    main()
