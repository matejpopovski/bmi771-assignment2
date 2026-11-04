"""Audit must distinguish available artifacts from an actual AP recheck."""
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter
import verify_detection as verifier
from make_detection_report import audit_matches_artifacts


@pytest.fixture
def detector_artifacts(tmp_path, monkeypatch):
    # Tiny strict-loadable checkpoints isolate artifact validation from FCOS
    # numerics (covered by the detector tests), without downloading any weights.
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    monkeypatch.setattr(verifier, 'FCOS', lambda **kwargs: torch.nn.Linear(1, 1))
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/test.json').write_text('{}')
    (tmp_path / 'pretrained').mkdir()
    review = tmp_path / 'results/detection_review'
    review.mkdir(parents=True)
    (review / 'data_audit.json').write_text(json.dumps({
        'split_filename_overlap': 0,
        'trainval': {'images_checked': True}, 'test': {'images_checked': True}}))
    for index, name in enumerate(['reference_fcos', 'fcos']):
        folder = tmp_path / 'results' / name
        (folder / 'viz').mkdir(parents=True)
        for image_id in range(1, 13):
            Image.new('RGB', (2, 2)).save(folder / 'viz' / f'{image_id:06d}.png')
        (folder / 'config.yaml').write_text('opt: {epochs: 1, warmup: false}\n')
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(index)
        torch.save({'epoch': 0, 'state_dict': model.state_dict()}, folder / 'model_final.pth.tar')
        (folder / 'metrics.json').write_text(json.dumps({
            'AP': .3, 'AP50': .6, 'AP75': .3, 'num_images': 4952,
            'model_seconds': 1, 'inference_seconds': 2, 'total_seconds': 3,
            'device': 'cuda:0'}))
        (folder / 'evaluation.log').write_text('fixture evaluation\n')
        if name == 'reference_fcos':
            shutil.copy2(folder / 'model_final.pth.tar', tmp_path / 'pretrained/det_voc_res18.pth.tar')
            (folder / 'PROVENANCE.txt').write_text('Fixture reference checkpoint\n')
            (folder / 'reference_run.json').write_text(json.dumps({
                'checkpoint_sha256': verifier.sha256(folder / 'model_final.pth.tar'),
                'annotations_sha256': verifier.sha256(tmp_path / 'data/test.json'), 'sources': {}}))
        else:
            (folder / 'history.jsonl').write_text(json.dumps({
                'epoch': 1, 'train': {'final_loss': 1., 'seconds': 1.}}) + '\n')
            (folder / 'training.log').write_text('fixture training\n')
            (folder / 'run_metadata.json').write_text('{}')
            for tag in ['cls_loss', 'reg_loss', 'ctr_loss']:
                with SummaryWriter(str(folder / 'logs' / ('train_all_losses_' + tag))) as writer:
                    writer.add_scalar(tag, 1., 0)
    return tmp_path


def test_explicit_ap_recheck_fails_when_predictions_are_missing(detector_artifacts):
    baseline = verifier.audit()
    assert baseline['ready']  # all other artifact requirements pass
    assert baseline['prediction_recheck_complete'] is False
    checked = verifier.audit(recompute_ap=True)
    assert checked['ready'] is False
    assert len(checked['errors']) == 2
    assert all('--recompute-ap requires eval_results.json' in error for error in checked['errors'])


def test_separate_audit_output_preserves_supplied_record(detector_artifacts):
    original = detector_artifacts / 'results/detection_review/detector_audit.json'
    original.write_text('{"original_record": true}')
    output = detector_artifacts / 'new_review/audit.json'
    result = verifier.audit(output_path=output)
    assert result['ready']
    assert json.loads(output.read_text()) == result
    assert original.read_text() == '{"original_record": true}'


def test_cached_ap_is_not_reused_for_a_different_checkpoint(detector_artifacts):
    original = verifier.audit()
    for name in ['reference_fcos', 'fcos']:
        predictions = detector_artifacts / 'results' / name / 'eval_results.json'
        predictions.write_text('[]')
        record = original['experiments'][name]
        record.update({
            'prediction_sha256': verifier.sha256(predictions),
            'ground_truth_sha256': verifier.sha256(detector_artifacts / 'data/test.json'),
            'prediction_recheck': {'AP': .3, 'AP50': .6, 'AP75': .3},
            'checkpoint_sha256': 'a-different-checkpoint'})
    (detector_artifacts / 'results/detection_review/detector_audit.json').write_text(json.dumps(original))
    checked = verifier.audit()
    assert checked['ready']
    assert checked['prediction_recheck_complete'] is False
    assert all('prediction_recheck' not in result for result in checked['experiments'].values())


@pytest.mark.parametrize('change', ['checkpoint', 'metrics', 'missing'])
def test_report_rejects_stale_or_unavailable_audited_artifacts(detector_artifacts, change):
    audit = verifier.audit()
    results = detector_artifacts / 'results'
    assert audit_matches_artifacts(audit, results)
    if change == 'checkpoint':
        (results / 'fcos/model_final.pth.tar').write_bytes(b'a different checkpoint')
    elif change == 'metrics':
        (results / 'fcos/metrics.json').write_text('{"AP50": 0.1}')
    else:
        (results / 'fcos/model_final.pth.tar').unlink()
    assert not audit_matches_artifacts(audit, results)
