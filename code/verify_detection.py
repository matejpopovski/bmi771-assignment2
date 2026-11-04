"""Audit the trained detector separately from the instructor-supplied reference.

Optionally recompute COCO AP from saved predictions and the original release GT.
This validates artifacts; it does not claim personal or team review is complete.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from det_libs.config import load_config
from det_libs.modeling import FCOS

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(recompute_ap=False):
    torch.set_num_threads(2)
    errors, summaries, checkpoints = [], {}, {}
    review = ROOT / 'results/detection_review'
    review.mkdir(exist_ok=True)
    previous_path = review / 'detector_audit.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    for name in ['reference_fcos', 'fcos']:
        folder = ROOT / 'results' / name
        required = ['config.yaml', 'model_final.pth.tar', 'metrics.json', 'evaluation.log']
        required += ['PROVENANCE.txt', 'reference_run.json'] if name == 'reference_fcos' else ['history.jsonl', 'training.log', 'run_metadata.json']
        missing = [file for file in required if not (folder / file).is_file()]
        if missing:
            errors.append(f'{name}: missing {missing}')
            continue
        cfg = load_config(folder / 'config.yaml')
        if cfg['dataset']['name'] != 'VOC2007' or cfg['dataset']['train'] != 'trainval' or cfg['dataset']['test'] != 'test':
            errors.append(f'{name}: incorrect data split')
        metrics = json.loads((folder / 'metrics.json').read_text())
        if metrics.get('num_images') != 4952:
            errors.append(f'{name}: evaluation must cover 4952 images')
        for key in ['AP', 'AP50', 'AP75']:
            value = metrics.get(key, float('nan'))
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f'{name}: invalid {key}')
        for key in ['model_seconds', 'inference_seconds', 'total_seconds']:
            if not math.isfinite(metrics.get(key, float('nan'))) or metrics.get(key, 0) <= 0:
                errors.append(f'{name}: invalid timing {key}')
        if not metrics['model_seconds'] <= metrics['inference_seconds'] <= metrics['total_seconds']:
            errors.append(f'{name}: inconsistent timing scopes')
        if not str(metrics.get('device', '')).startswith('cuda'):
            errors.append(f'{name}: required CUDA measurement missing')
        cfg['model']['backbone_weights'] = None
        model = FCOS(**cfg['model'])
        checkpoint = torch.load(folder / 'model_final.pth.tar', map_location='cpu', weights_only=True)
        strict_load = True
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except RuntimeError as error:
            strict_load = False
            errors.append(f'{name}: checkpoint/config mismatch: {error}')
        if any(torch.is_floating_point(value) and not torch.isfinite(value).all()
               for value in checkpoint['state_dict'].values()):
            errors.append(f'{name}: nonfinite saved tensor')
        checkpoints[name] = checkpoint
        checkpoint_hash = sha256(folder / 'model_final.pth.tar')
        samples = sorted((folder / 'viz').glob('*.png'))
        if len(samples) != 12:
            errors.append(f'{name}: expected 12 saved samples')
        for path in samples:
            with Image.open(path) as image:
                image.verify()
        summary = {'checkpoint_sha256': checkpoint_hash, 'epoch_zero_based': checkpoint.get('epoch'),
                   'strict_load': strict_load, 'metrics': metrics, 'sample_images': [path.name for path in samples]}
        if name == 'reference_fcos':
            provenance = json.loads((folder / 'reference_run.json').read_text())
            if provenance['checkpoint_sha256'] != checkpoint_hash or checkpoint_hash != sha256(ROOT / 'pretrained/det_voc_res18.pth.tar'):
                errors.append('reference_fcos: instructor checkpoint digest mismatch')
            if provenance['annotations_sha256'] != sha256(ROOT / 'data/test.json'):
                errors.append('reference_fcos: annotation provenance mismatch')
            for relative, digest in provenance['sources'].items():
                if sha256(ROOT / 'code' / relative) != digest:
                    errors.append(f'reference_fcos: evaluated source changed: {relative}')
        else:
            opt = cfg['opt']
            expected_epochs = opt['epochs'] + (opt['warmup_epochs'] if opt['warmup'] else 0)
            history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
            if [row['epoch'] for row in history] != list(range(1, expected_epochs + 1)):
                errors.append('fcos: incomplete or duplicated epoch history')
            if checkpoint.get('epoch', -1) + 1 != expected_epochs:
                errors.append('fcos: checkpoint is not the final scheduled epoch')
            for row in history:
                if not all(math.isfinite(value) for value in row['train'].values()):
                    errors.append(f'fcos: nonfinite training metrics in epoch {row["epoch"]}')
            summary['epochs'] = len(history)
            summary['training_seconds'] = sum(row['train']['seconds'] for row in history)
            summary['initial_loss'] = history[0]['train']['final_loss']
            summary['final_loss'] = history[-1]['train']['final_loss']
            event_counts = {}
            for tag in ['cls_loss', 'reg_loss', 'ctr_loss']:
                logdir = folder / 'logs' / ('train_all_losses_' + tag)
                events = EventAccumulator(str(logdir)).Reload()
                count = sum(len(events.Scalars(key)) for key in events.Tags()['scalars'])
                event_counts[tag] = count
                if count == 0:
                    errors.append(f'fcos: no TensorBoard scalar records for {tag}')
            summary['tensorboard_scalar_counts'] = event_counts
        prediction_file = folder / 'eval_results.json'
        prior = previous.get('experiments', {}).get(name, {})
        if (not recompute_ap and prediction_file.exists()
                and prior.get('prediction_sha256') == sha256(prediction_file)
                and prior.get('ground_truth_sha256') == sha256(ROOT / 'data/test.json')):
            for key in ['prediction_recheck', 'prediction_sha256', 'ground_truth_sha256']:
                summary[key] = prior[key]
            for key, value in prior['prediction_recheck'].items():
                if abs(value - metrics[key]) > 1e-10:
                    errors.append(f'{name}: cached independent {key} disagrees')
        if recompute_ap and prediction_file.exists():
            log = review / f'{name}_prediction_recheck.log'
            with log.open('w') as output, redirect_stdout(output):
                ground_truth = COCO(str(ROOT / 'data/test.json'))
                predictions = ground_truth.loadRes(str(prediction_file))
                evaluator = COCOeval(ground_truth, predictions, 'bbox')
                evaluator.params.imgIds = sorted(ground_truth.getImgIds())
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            recheck = {'AP': float(evaluator.stats[0]), 'AP50': float(evaluator.stats[1]), 'AP75': float(evaluator.stats[2])}
            summary['prediction_recheck'] = recheck
            summary['prediction_sha256'] = sha256(prediction_file)
            summary['ground_truth_sha256'] = sha256(ROOT / 'data/test.json')
            for key, value in recheck.items():
                if abs(value - metrics[key]) > 1e-10:
                    errors.append(f'{name}: independently recomputed {key} disagrees')
        summaries[name] = summary
        del model
    if set(checkpoints) == {'reference_fcos', 'fcos'}:
        reference, own = [checkpoints[name]['state_dict'] for name in ['reference_fcos', 'fcos']]
        changed = [key for key in own if torch.is_floating_point(own[key]) and not torch.equal(own[key], reference[key])]
        if not changed:
            errors.append('Trained detector is identical to instructor detector')
        summaries['fcos']['floating_tensors_different_from_instructor'] = len(changed)
        if summaries['reference_fcos']['sample_images'] != summaries['fcos']['sample_images']:
            errors.append('Reference and trained-model sample image IDs differ')
    data = json.loads((review / 'data_audit.json').read_text())
    if data.get('split_filename_overlap') != 0 or not all(data[s]['images_checked'] for s in ['trainval', 'test']):
        errors.append('Full data/image audit missing or overlapping splits')
    result = {'ready': not errors, 'scope': 'required detection/Transformer component; personal and team review remain separate',
              'errors': errors, 'experiments': summaries}
    (review / 'detector_audit.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute-ap', action='store_true')
    args = parser.parse_args()
    result = audit(args.recompute_ap)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 1)
