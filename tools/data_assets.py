"""Verify/import local Release archives or download them from a user-supplied release URL."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['verify', 'import', 'download'])
    ap.add_argument('--directory', type=Path, default=ROOT / 'downloads')
    ap.add_argument('--base-url', help='Release download base URL, ending with the release tag')
    ap.add_argument('--asset', choices=['training', 'comsol', 'advanced', 'all'], default='all')
    args = ap.parse_args()
    assets = json.loads((ROOT / 'data_manifest.json').read_text())['assets']
    if args.asset != 'all':
        assets = [a for a in assets if args.asset in a['name']]
    if args.action == 'download' and not args.base_url:
        ap.error('download requires --base-url')
    for asset in assets:
        archive = args.directory.resolve() / asset['name']
        if args.action == 'download' and not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix('.part')
            try:
                with urllib.request.urlopen(args.base_url.rstrip('/') + '/' + asset['name']) as response, partial.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
                if digest(partial) != asset['sha256']:
                    raise ValueError(f'Archive SHA-256 mismatch: {asset["name"]}')
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        if archive.stat().st_size != asset['bytes'] or digest(archive) != asset['sha256']:
            raise ValueError(f'Archive size/SHA-256 mismatch: {asset["name"]}')
        with zipfile.ZipFile(archive) as z:
            expected = {m['path']: m for m in asset['files']}
            if set(z.namelist()) != set(expected):
                raise ValueError('Unexpected or missing archive members')
            for name, info in expected.items():
                target = (ROOT / name).resolve()
                if not target.is_relative_to(ROOT):
                    raise ValueError(f'Unsafe archive path: {name}')
                h = hashlib.sha256()
                count = 0
                with z.open(name) as stream:
                    for block in iter(lambda: stream.read(4*1024*1024), b''):
                        h.update(block)
                        count += len(block)
                if h.hexdigest() != info['sha256'] or count != info['bytes']:
                    raise ValueError(f'Member checksum mismatch: {name}')
                if args.action in ('import', 'download'):
                    if target.exists():
                        if digest(target) != info['sha256']:
                            raise FileExistsError(f'Refusing to overwrite a different file: {name}')
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        partial = target.with_suffix(target.suffix + '.part')
                        try:
                            with z.open(name) as stream, partial.open('wb') as out:
                                shutil.copyfileobj(stream, out)
                            if digest(partial) != info['sha256']:
                                raise ValueError(f'Extracted checksum mismatch: {name}')
                            partial.replace(target)
                        finally:
                            partial.unlink(missing_ok=True)
        print(f'{args.action}: {asset["name"]} OK', flush=True)

if __name__ == '__main__':
    main()
