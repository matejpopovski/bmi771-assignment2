"""Audit Matej's four classifier experiments without downloading any weights."""
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from cls_libs.config import load_config
from cls_libs.utils import build_model

ROOT = Path(__file__).resolve().parent.parent
NAMES = ('simple', 'cnn', 'vit', 'resnet18')


def audit():
    torch.set_num_threads(2)
    errors, summaries = [], {}
    for name in NAMES:
        folder = ROOT / 'results' / name
        required = ['config.yaml', 'model_final.pth.tar', 'metrics.json',
                    'history.jsonl', 'training.log', 'evaluation.log']
        missing = [filename for filename in required if not (folder / filename).is_file()]
        if missing:
            errors.append(f'{name}: missing ' + ', '.join(missing))
            continue
        cfg = load_config(folder / 'config.yaml')
        metrics = json.loads((folder / 'metrics.json').read_text())
        history = {row['epoch']: row for row in
                   (json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines())}
        opt = cfg['opt']
        epochs = opt['epochs'] + (opt['warmup_epochs'] if opt['warmup'] else 0)
        if set(history) != set(range(1, epochs + 1)):
            errors.append(f'{name}: incomplete scheduled epoch history')
        if cfg['dataset']['train'] != 'train' or cfg['dataset']['test'] != 'val':
            errors.append(f'{name}: unexpected data split')
        if metrics.get('num_images') != 10000 or metrics.get('flip_augmentation') is not False:
            errors.append(f'{name}: evaluation must be 10,000 images without flip averaging')
        for key in ('top1', 'top5'):
            score = metrics.get(key, float('nan'))
            if not math.isfinite(score) or not 0 <= score <= 100:
                errors.append(f'{name}: invalid {key}')
            if epochs in history and abs(score - history[epochs]['val'][key]) > 0.02:
                errors.append(f'{name}: final checkpoint evaluation disagrees with final epoch {key}')
        if metrics.get('top1', 101) > metrics.get('top5', -1):
            errors.append(f'{name}: top-1 exceeds top-5')
        for row in history.values():
            if row['val']['num_images'] != 10000 or not all(math.isfinite(v) for v in row['train'].values()):
                errors.append(f'{name}: invalid epoch metrics at {row["epoch"]}')
        if not list((folder / 'logs').rglob('events.out.tfevents.*')):
            errors.append(f'{name}: TensorBoard events missing')
        checkpoint_path = folder / 'model_final.pth.tar'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['epoch'] + 1 != epochs:
            errors.append(f'{name}: checkpoint is not the final scheduled epoch')
        model = build_model(cfg, pretrained=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if any(torch.is_floating_point(v) and not torch.isfinite(v).all()
               for v in checkpoint['state_dict'].values()):
            errors.append(f'{name}: nonfinite saved weights')
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, cfg['input']['img_crop_size'], cfg['input']['img_crop_size']))
        if output.shape != (2, 100) or not torch.isfinite(output).all():
            errors.append(f'{name}: invalid reloaded-model output')
        summaries[name] = dict(epochs=epochs, parameters=sum(p.numel() for p in model.parameters()),
                               metrics=metrics, device=cfg['devices'][0],
                               batch_size=cfg['loader']['batch_size'],
                               checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
        del model, checkpoint
    result = dict(ready=not errors, scope='classification contribution only; not a complete team submission',
                  errors=errors, experiments=summaries)
    (ROOT / 'results' / 'classification_audit.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    result = audit()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['ready'] else 1)
