"""Package the Git-visible submission, excluding caches and ignored experiments."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def is_submission_file(name):
    """Exclude local runtimes even if they were accidentally added to Git."""
    parts = Path(name).parts
    excluded = {'node_modules', '__pycache__', '.pytest_cache', '.vercel', '.vite', '.venv'}
    if any(part in excluded for part in parts):
        return False
    if parts[0] in {'build', 'dist'} or parts[:2] == ('web', 'dist'):
        return False
    if parts[-1].startswith('.env') and parts[-1] != '.env.example':
        return False
    return (ROOT/name).is_file()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT.parent/'pushbridge.zip')
    args = parser.parse_args()
    names = subprocess.check_output([
        'git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'
    ]).decode().split('\0')
    files = sorted({name for name in names if name and is_submission_file(name)})
    required = {'README.md', 'index.html', '.github/workflows/tests.yml', 'dashboard/data.js',
                'dashboard/precision-preview.png', 'docs/REVIEWER_GUIDE.md', 'docs/PRECISION.md',
                'src/pushbridge/precision.py', 'results/precision_finish/evaluation.json',
                'results/precision_finish/video_verification.json', 'results/submission_audit.json',
                'docs/WEB_APP.md', 'web/package.json', 'web/pnpm-lock.yaml', 'web/index.html',
                'web/preview.png', 'web/public/data/manifest.json',
                'web/public/data/export-verification.json', 'scripts/export_web_data.py'}
    if not required.issubset(files):
        raise RuntimeError(f'Missing required deliverables: {sorted(required-set(files))}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in files:
            archive.write(ROOT/name, f'pushbridge/{name}')
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip():
            raise RuntimeError(f'ZIP integrity check failed: {bad}')
        if set(archive.namelist()) != {f'pushbridge/{name}' for name in files}:
            raise RuntimeError('ZIP file list differs from the selected submission file list')
    temporary.replace(args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix('.zip.sha256').write_text(f'{digest}  {args.out.name}\n')
    print(f'{args.out}: {len(files)} files, {args.out.stat().st_size/2**20:.2f} MiB; ZIP CRC verified.')


if __name__ == '__main__':
    main()
