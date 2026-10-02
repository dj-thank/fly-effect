"""Check shipped replay contents, including the original manifest-only failure."""
from pathlib import Path
import tarfile

import pytest

from scripts.check_sdist import verify_sdist


PACKAGE = Path(__file__).resolve().parents[2]/'examples/hae-recorded-replay'


def source_archive(path, include):
    with tarfile.open(path, 'w:gz') as archive:
        for file in sorted(PACKAGE.rglob('*')):
            relative = file.relative_to(PACKAGE).as_posix()
            if file.is_file() and include(relative):
                archive.add(file, arcname='fly_effect-fixture/examples/hae-recorded-replay/'+relative)
    return path


def test_complete_source_distribution_preserves_public_replay(tmp_path):
    archive = source_archive(tmp_path/'complete.tar.gz', lambda path: True)
    assert verify_sdist(archive) == []


def test_original_manifest_only_distribution_fails(tmp_path):
    archive = source_archive(tmp_path/'incomplete.tar.gz', lambda path: path.endswith('.json'))
    failures = verify_sdist(archive)
    assert any('missing package files' in failure for failure in failures)


@pytest.mark.parametrize('missing', ['assets/actual-motion.mp4', 'assets/replay.js', 'LICENSES/Apache-2.0.txt'])
def test_source_distribution_requires_media_viewer_and_license(tmp_path, missing):
    archive = source_archive(tmp_path/'missing.tar.gz', lambda path: path != missing)
    assert any(missing in failure for failure in verify_sdist(archive))
