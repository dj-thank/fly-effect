"""Verify the recorded replay inside a built source distribution.

This checks the shipped bytes and notices, rather than the working checkout.
It does not establish biological validity or change publication authorization.
"""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

try:
    from replay_package import verify_package
except ModuleNotFoundError:
    from scripts.replay_package import verify_package


def verify_sdist(path):
    """Return replay-package failures from one tar source distribution."""
    prefix = ('examples', 'hae-recorded-replay')
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary)
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            roots = {PurePosixPath(m.name).parts[0] for m in members if m.name}
            if len(roots) != 1:
                return ['Source distribution must have one top-level directory']
            seen = set()
            for member in members:
                parts = PurePosixPath(member.name).parts
                if len(parts) < 3 or parts[1:3] != prefix:
                    continue
                relative = PurePosixPath(*parts[3:])
                if (member.name.startswith('/') or '..' in parts
                        or '\\' in member.name or any(':' in part for part in parts)
                        or member.issym() or member.islnk()):
                    return ['Unsupported replay archive member: ' + member.name]
                if member.isdir():
                    continue
                if not member.isfile() or not parts[3:]:
                    return ['Unsupported replay archive member: ' + member.name]
                if relative in seen:
                    return ['Duplicate replay archive member: ' + member.name]
                seen.add(relative)
                destination = package.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source, target)
        return verify_package(package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('distribution', type=Path, help='One .tar.gz or the dist directory')
    args = parser.parse_args()
    paths = sorted(args.distribution.glob('*.tar.gz')) if args.distribution.is_dir() else [args.distribution]
    if not paths:
        parser.exit(1, 'FAIL: No source distribution found\n')
    failed = False
    for path in paths:
        try:
            problems = verify_sdist(path)
        except (OSError, tarfile.TarError, ValueError) as exc:
            problems = [str(exc)]
        print(path.name + ': ' + ('FAIL' if problems else 'PASS'))
        for problem in problems:
            print('  ' + problem)
        failed |= bool(problems)
    raise SystemExit(int(failed))


if __name__ == '__main__':
    main()
