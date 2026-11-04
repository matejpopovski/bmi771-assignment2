"""The final archive preserves required weights/logs and omits transient dumps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_submission as package


def test_submission_archive_selection_and_manifest(tmp_path, monkeypatch):
    files = {
        'code/example.py': 'code',
        'writeup/report.pdf': 'final report',
        'writeup/report_draft.pdf': 'outdated report',
        'writeup/matej_classification_draft.pdf': 'outdated classifier report',
        'writeup/team_contributions_draft.md': 'unconfirmed contributions',
        'results/fcos/model_final.pth.tar': 'required final weights',
        'results/fcos/logs/events.out.tfevents.example': 'required event log',
        'results/fcos/epoch_003.pth.tar': 'intermediate',
        'results/fcos/model_final.pth.tar.partial': 'unfinished write',
        'results/fcos/eval_results.json': 'large prediction dump',
        'results/detection_review/reference_mps/model_final.pth.tar': 'duplicate reference weights',
        'results/detection_review/reference_mps/metrics.json': 'supplemental metrics',
    }
    for relative, value in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    monkeypatch.setattr(package, 'ROOT', tmp_path)
    monkeypatch.setattr(package, 'audit', lambda bonus: [])
    assert package.main(argparse.Namespace(bonus=False)) == 0
    with ZipFile(tmp_path/'submission.zip') as z:
        names = set(z.namelist())
        assert 'results/fcos/model_final.pth.tar' in names
        assert 'results/fcos/logs/events.out.tfevents.example' in names
        assert 'results/detection_review/reference_mps/metrics.json' in names
        assert not any(name.endswith('.partial') or '/epoch_' in name or name.endswith('eval_results.json') for name in names)
        assert 'results/detection_review/reference_mps/model_final.pth.tar' not in names
        assert not any(Path(name).stem.endswith('_draft') for name in names)
        assert {name.split('/')[0] for name in names} == {'code', 'writeup', 'results'}
        manifest = json.loads(z.read('results/file_sha256.json'))
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
