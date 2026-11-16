"""Install and verify the pinned ICORE runtime in the active Python environment."""
import argparse
from hashlib import sha256
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen

COMMIT = 'a63ebffb005725a02d2c3e4247fb73aec453e590'
ARCHIVE_SHA256 = 'f0d3da4ab6fe13d36b5547607787dfb15472272e29e6f3863bea4af825dd1d49'
VERSION = '0.20.0+' + COMMIT[:12]
OFFLINE_WHEEL_SHA256 = '1dd4205fb020a5689a053b65872529740a14d6bc6b908f9d0e71e6ce7147bce6'
ROOT = Path(__file__).resolve().parents[1]


def runtime_ok():
    """Check the actual installed API, not just distribution metadata.

    Isolated mode excludes the repository and old local icore directories.
    """
    code = (
        'from importlib.metadata import version; '
        'from icore import ChainQP, Solver; '
        f'assert version("icore") == {VERSION!r}'
    )
    return subprocess.run([sys.executable, '-I', '-c', code],
                          capture_output=True, text=True).returncode == 0


def install_wheel(wheel):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--no-deps', '--force-reinstall', str(wheel)], check=True)
    if not runtime_ok():
        raise RuntimeError(f'ICORE import verification failed in {sys.executable}')
    print(f'ICORE {VERSION}: Solver and ChainQP verified in {sys.executable}')


def stage_runtime(archive, destination):
    """Keep solver kernels, build inputs and licenses; omit benchmark-only code."""
    prefix = f'ICORE-{COMMIT}/'
    omitted = ('icore/benchmark.py', 'icore/_comparators.py', 'icore/vendor/README.md',
               'icore/vendor/hpipm/', 'icore/vendor/qoco/__init__.py',
               'icore/vendor/qoco/api.py', 'icore/vendor/qoco/bridge.c')
    licenses = ('LICENSE', 'LICENSE-NOTE.md', 'THIRD_PARTY_NOTICES.md')
    with tarfile.open(archive) as source:
        for member in source:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            name = member.name[len(prefix):]
            if name in licenses:
                name = 'icore/' + name
            elif not name.startswith('icore/') or name.startswith(omitted):
                continue
            if '..' in PurePosixPath(name).parts:
                raise ValueError('Invalid source archive path')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open('wb') as output:
                shutil.copyfileobj(stream, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, help='Install the included offline wheel.')
    args = parser.parse_args()
    if args.wheel:
        expected_name = f'icore-{VERSION}-py3-none-any.whl'
        if args.wheel.name != expected_name:
            parser.error(f'Expected {expected_name}')
        # The offline wheel is supplied with the update, not fetched from PyPI.
        if sha256(args.wheel.read_bytes()).hexdigest() != OFFLINE_WHEEL_SHA256:
            parser.error('ICORE offline wheel checksum mismatch')
        install_wheel(args.wheel.resolve())
        return
    if runtime_ok():
        print(f'ICORE {VERSION}: Solver and ChainQP verified in {sys.executable}')
        return
    cache = ROOT / '.build' / 'dependencies'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f'ICORE-{COMMIT}.tar.gz'
    if not archive.is_file():
        url = f'https://codeload.github.com/sametuzun781/ICORE/tar.gz/{COMMIT}'
        print(f'Downloading ICORE {COMMIT[:12]}', flush=True)
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError('ICORE archive checksum mismatch')
        archive.write_bytes(data)
    if sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError(f'ICORE archive checksum mismatch: remove {archive} and retry')
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        project = Path(directory)
        stage_runtime(archive, project)
        (project / 'pyproject.toml').write_text(f'''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "icore"
version = "{VERSION}"
description = "ICORE runtime pinned for CT-STL"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11"]

[project.urls]
Source = "https://github.com/sametuzun781/ICORE/tree/{COMMIT}"

[tool.setuptools.packages.find]
include = ["icore", "icore.*"]

[tool.setuptools.package-data]
icore = ["**/*"]
''')
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                        '--wheel-dir', str(cache), str(project)], check=True)
    wheel = cache / f'icore-{VERSION}-py3-none-any.whl'
    install_wheel(wheel)


if __name__ == '__main__':
    main()
