"""Review the curated release, not ignored experiment/virtualenv directories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('src', 'scripts', 'tests', 'docs', 'experiments', 'results/reference', 'provenance', 'environment', '.github')
TOP = ('README.md', 'LICENSE', 'CITATION.cff', 'CONTRIBUTING.md', 'pyproject.toml', '.gitignore', '.gitattributes')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    result = [ROOT / name for name in TOP]
    for name in FOLDERS:
        result.extend(p for p in (ROOT / name).rglob('*') if p.is_file() and
                      '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--write-reference-manifest', action='store_true')
    parser.add_argument('--original-root', type=Path)
    args = parser.parse_args(); paths = files()
    findings = []
    secret = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    for p in paths:
        rel = p.relative_to(ROOT)
        if p.is_symlink() or p.suffix in ('.npz', '.npy', '.pt', '.pth', '.pkl', '.joblib', '.pem', '.key'):
            findings.append(f'Unexpected artifact/symlink: {rel}')
        if p.stat().st_size > 5_000_000: findings.append(f'Unexpected large file: {rel}')
        if p.suffix not in ('.png', '.pdf'):
            text = p.read_text()
            if re.search(r'[\u4e00-\u9fff]', text): findings.append(f'Non-English CJK text: {rel}')
            if secret.search(text): findings.append(f'Possible credential in {rel}')
            if re.search(r'/(?:beegfs/)?home/[^/\s]+/', text): findings.append(f'Personal server path in {rel}')
            if p.suffix == '.py': compile(text, str(rel), 'exec')
    reference_paths = [p for p in paths if p.is_relative_to(ROOT / 'results/reference') or
                       p.is_relative_to(ROOT / 'docs/figures') or
                       p.is_relative_to(ROOT / 'experiments/reference_configs') or
                       p.is_relative_to(ROOT / 'experiments/selections')]
    if args.write_reference_manifest:
        (ROOT / 'provenance/REFERENCE_FILES.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in reference_paths))
    verified = 0
    with (ROOT / 'provenance/SOURCE_FILES.csv').open(newline='') as f: records = list(csv.DictReader(f))
    by_destination = {r['release_file']: r for r in records}
    for p in reference_paths:
        row = by_destination[str(p.relative_to(ROOT))]
        if sha(p) != row['source_sha256']: findings.append(f'Changed original numeric/reference file: {p.relative_to(ROOT)}')
        verified += 1
    source_verified = 0
    if args.original_root:
        for row in records:
            source = args.original_root / row['source_project'] / row['source_file']
            if sha(source) != row['source_sha256']: findings.append(f'Original file changed: {row["source_file"]}')
            source_verified += 1
    result = {'status': 'PASS' if not findings else 'FAIL', 'curated_files': len(paths),
              'reference_files_byte_identical': verified, 'original_source_hashes_verified': source_verified,
              'credential_pattern_scan': 'heuristic; not a proof that every secret format is covered',
              'findings': findings}
    print(json.dumps(result, indent=2))
    if findings: raise SystemExit(1)


if __name__ == '__main__': main()
