"""Package the verified classifier contribution and downloadable final weights."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from pypdf import PdfReader
from verify_classification import audit, NAMES

ROOT = Path(__file__).resolve().parent.parent


def make_zip(destination, files):
    temp = destination.with_suffix('.zip.partial')
    with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path, path.relative_to(ROOT))
    with ZipFile(temp) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
    temp.replace(destination)


def main():
    result = audit()
    if not result['ready']:
        raise RuntimeError('Classifier artifacts are incomplete: ' + '; '.join(result['errors']))
    report = ROOT / 'writeup' / 'matej_classification.pdf'
    if not report.exists():
        raise RuntimeError('Generate the classification report first')
    report_text='\n'.join(p.extract_text() or '' for p in PdfReader(report).pages)
    if 'Pending' in report_text or 'DRAFT:' in report_text:
        raise RuntimeError('The report still contains unfinished results')
    files = [p for p in (ROOT/'code').rglob('*') if p.is_file() and
             not any(x in p.parts for x in ('__pycache__','.pytest_cache','.DS_Store'))]
    files.append(report)
    for name in NAMES:
        files += [p for p in (ROOT/'results'/name).rglob('*') if p.is_file() and not p.name.startswith('epoch_')]
    for filename in ('classification_audit.json','local_tests.txt','cloud_tests.txt',
                     'environment_local.json','environment_cloud.txt','packages_local.txt',
                     'data_audit.json','model_parameters.json','classification_compute.json'):
        path=ROOT/'results'/filename
        if path.exists(): files.append(path)
    files += [ROOT/'results/figures'/name for name in ('classification_curves.png','classification_loss_top5.png')]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path=ROOT/'results/classification_files_sha256.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    files.append(manifest_path)
    make_zip(ROOT/'matej_classification.zip',files)
    weights=[ROOT/'results'/name/'model_final.pth.tar' for name in NAMES]
    make_zip(ROOT/'classification-checkpoints.zip',weights)
    print('Verified classification contribution: matej_classification.zip')
    print('Final weights: classification-checkpoints.zip')
    print('These are component artifacts, not the finished two-person Canvas submission.')


if __name__ == '__main__':
    main()
