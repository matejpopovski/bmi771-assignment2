"""Build a private detection handoff ZIP, including final weights, after audit.

This is a component archive for Matej, not the combined Canvas submission.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from verify_detection import audit

ROOT = Path(__file__).resolve().parent.parent


def main():
    result = audit()
    if not result['ready']:
        raise RuntimeError('Detector audit failed: ' + '; '.join(result['errors']))
    required_report = ['sean_detection.md', 'sean_detection.pdf', 'sean_detection_analysis.md', 'TEAM_HANDOFF.md']
    if any(not (ROOT / 'writeup' / name).is_file() for name in required_report):
        raise FileNotFoundError('Finish the report and handoff note before packaging')
    files = []
    roots = [ROOT / 'code', ROOT / 'results/fcos', ROOT / 'results/reference_fcos', ROOT / 'results/detection_review']
    for folder in roots:
        for path in folder.rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in ['__pycache__', '.pytest_cache'] for part in path.parts):
                continue
            if path.name in ['eval_results.json', '.DS_Store', '.gitignore'] or path.name.startswith('epoch_') or path.name.endswith('.partial'):
                continue
            # Extra local reference validation does not require a second copy
            # of the instructor's large checkpoint in the handoff archive.
            if folder.name == 'detection_review' and path.name.endswith(('.pth', '.pth.tar', '.pt')):
                continue
            files.append(path)
    files += [ROOT / 'writeup' / name for name in required_report]
    files += list((ROOT / 'results/figures').glob('*fcos*.png'))
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest_path = ROOT / 'results/detection_review/component_sha256.json'
    manifest.pop(str(manifest_path.relative_to(ROOT)), None)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    files = sorted(set(files + [manifest_path]))
    temporary = ROOT / 'sean_detection_component.zip.partial'
    destination = ROOT / 'sean_detection_component.zip'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(ROOT))
    with ZipFile(temporary) as z:
        if z.testzip():
            raise RuntimeError('Archive CRC check failed')
        for name, digest in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise RuntimeError('Archive digest check failed: ' + name)
    temporary.replace(destination)
    print(f'Private component handoff: {destination} ({destination.stat().st_size/1e6:.1f} MB)')
    print('Includes course instructor material. Keep private; this is not the final Canvas ZIP.')


if __name__ == '__main__':
    main()
