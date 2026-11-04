"""Regression checks for inherited reference metrics and download selection."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
import run_experiments as runner
import download_datasets as download


def test_imported_reference_metrics_are_preserved_and_recomputed(tmp_path, monkeypatch):
    (tmp_path/'pretrained').mkdir()
    (tmp_path/'data').mkdir()
    (tmp_path/'code').mkdir()
    (tmp_path/'code/eval_det.py').write_text('evaluation source')
    (tmp_path/'data/test.json').write_text('{}')
    (tmp_path/'pretrained/det_voc_res18.pth.tar').write_bytes(b'instructor')
    (tmp_path/'pretrained/det_voc_res18.yaml').write_text('devices: [mps]\nloader: {batch_size: 32}\n')
    dest = tmp_path/'results/reference_fcos'
    dest.mkdir(parents=True)
    (dest/'config.yaml').write_text('devices: [mps]\n')
    (dest/'metrics.json').write_text('{"old_mps": true}')
    calls = []

    def fake_run(command, log):
        calls.append(command)
        log.write_text('new evaluation')
        (dest/'metrics.json').write_text('{"AP50": 0.6}')

    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'CODE', tmp_path/'code')
    monkeypatch.setattr(runner, 'run', fake_run)
    runner.evaluate_reference('cpu', workers=0)
    assert len(calls) == 1
    archive = next((tmp_path/'logs/reference_fcos_previous').iterdir())
    assert json.loads((archive/'metrics.json').read_text()) == {'old_mps': True}
    assert yaml.safe_load((archive/'config.yaml').read_text())['devices'] == ['mps']
    assert yaml.safe_load((dest/'config.yaml').read_text())['devices'] == ['cpu']
    runner.evaluate_reference('cpu', workers=0)
    assert len(calls) == 1  # reuse genuinely matching provenance
    (tmp_path/'code/eval_det.py').write_text('changed evaluation source')
    runner.evaluate_reference('cpu', workers=0)
    assert len(calls) == 2


def test_detection_only_download_does_not_fetch_miniplaces(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download, 'DATA', tmp_path)
    monkeypatch.setattr(download, 'run', lambda *args: calls.append(args))
    download.main(detection_only=True)
    assert len(calls) == 4  # two archives, each followed by extraction
    assert all('miniplaces' not in ' '.join(call).lower() and 'gdown' not in call for call in calls)
    assert 'VOCtrainval' in ' '.join(calls[0])
    assert 'VOCtest' in ' '.join(calls[2])
