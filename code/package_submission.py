"""Audit required artifacts and build a clean Canvas submission ZIP."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import torch
import yaml
from pypdf import PdfReader
from cls_libs.config import load_config as load_cls_config
from cls_libs.utils import build_model
from det_libs.config import load_config as load_det_config
from det_libs.modeling import FCOS

ROOT = Path(__file__).resolve().parent.parent


def audit(bonus=False):
    torch.set_num_threads(4)
    errors = []
    required = ['simple', 'cnn', 'vit', 'resnet18', 'fcos', 'reference_fcos']
    if bonus:
        required += ['swin', 'coco']
    for name in required:
        folder = ROOT / 'results' / name
        for filename in ['metrics.json', 'config.yaml', 'model_final.pth.tar', 'evaluation.log']:
            if not (folder / filename).is_file():
                errors.append(f'{name}: missing {filename}')
        if not (folder / 'metrics.json').exists():
            continue
        metrics = json.loads((folder / 'metrics.json').read_text())
        expected = 10000 if name in ('simple', 'cnn', 'vit', 'resnet18') else (1000 if name == 'coco' else 4952)
        if metrics.get('num_images') != expected:
            errors.append(f'{name}: evaluation image count differs from {expected}')
        classification = name in ('simple', 'cnn', 'vit', 'resnet18')
        for key in (('top1', 'top5') if classification else ('AP', 'AP50', 'AP75')):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= (100 if classification else 1):
                errors.append(f'{name}: invalid metric {key}')
        checkpoint = None
        if all((folder / f).is_file() for f in ('config.yaml', 'model_final.pth.tar')):
            try:
                checkpoint = torch.load(folder / 'model_final.pth.tar', map_location='cpu', weights_only=True)
                if classification:
                    cfg = load_cls_config(folder / 'config.yaml')
                    model = build_model(cfg, pretrained=False)
                else:
                    cfg = load_det_config(folder / 'config.yaml')
                    cfg['model']['backbone_weights'] = None
                    model = FCOS(**cfg['model'])
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                for key, value in checkpoint['state_dict'].items():
                    if torch.is_floating_point(value) and not torch.isfinite(value).all():
                        errors.append(f'{name}: nonfinite checkpoint tensor {key}')
                del model
            except (RuntimeError, KeyError, ValueError) as error:
                errors.append(f'{name}: checkpoint/config validation failed: {error}')
        if name == 'reference_fcos':
            if not (folder / 'PROVENANCE.txt').exists():
                errors.append('Reference checkpoint provenance is missing')
            continue
        for filename in ['history.jsonl', 'training.log']:
            if not (folder / filename).is_file():
                errors.append(f'{name}: missing {filename}')
        if not list((folder / 'logs').rglob('events.out.tfevents.*')):
            errors.append(f'{name}: TensorBoard events are missing')
        if all((folder / f).exists() for f in ('config.yaml', 'model_final.pth.tar', 'history.jsonl')):
            config = yaml.safe_load((folder / 'config.yaml').read_text())
            opt = config['opt']
            expected_epochs = opt['epochs'] + (opt['warmup_epochs'] if opt['warmup'] else 0)
            if checkpoint is None or checkpoint.get('epoch', -1) + 1 != expected_epochs:
                errors.append(f'{name}: checkpoint is not from the final scheduled epoch')
            epochs = {json.loads(line)['epoch'] for line in (folder / 'history.jsonl').read_text().splitlines()}
            if epochs != set(range(1, expected_epochs + 1)):
                errors.append(f'{name}: epoch history is incomplete')
    team_path = ROOT / 'writeup' / 'team.json'
    if not team_path.exists():
        errors.append('Factual team roster/contributions are missing')
    else:
        team = json.loads(team_path.read_text())
        members = team.get('members', [])
        if any(not m.get('name') or not m.get('contribution') for m in members):
            errors.append('Incomplete team member metadata')
        if len(members) not in (2, 3) and not team.get('solo_approval'):
            errors.append('Team must have 2-3 members, or documented instructor approval')
    report = ROOT / 'writeup' / 'report.pdf'
    if not report.exists():
        errors.append('Final writeup/report.pdf is missing')
    else:
        text = '\n'.join(p.extract_text() or '' for p in PdfReader(report).pages)
        if any(x in text for x in ['IN-PROGRESS REPORT', 'Pending', 'still pending', 'not been supplied']):
            errors.append('Report still contains unfinished sections')
    (ROOT / 'results' / 'submission_audit.json').write_text(json.dumps(
        {'ready': not errors, 'include_bonus': bonus, 'errors': errors}, indent=2))
    return errors


def main(args):
    errors = audit(args.bonus)
    if errors:
        print('NOT READY\n' + '\n'.join('- ' + error for error in errors))
        return 1
    files = []
    for name in ['code', 'writeup', 'results']:
        for path in (ROOT / name).rglob('*'):
            if not path.is_file() or path.is_symlink():
                continue
            # The combined report supersedes component PDFs and handoff notes.
            if name == 'writeup' and path.relative_to(ROOT / name).as_posix() not in ('report.pdf', 'team.json'):
                continue
            if any(part in ('__pycache__', '.pytest_cache', '.DS_Store') for part in path.parts):
                continue
            if (path.name.startswith('epoch_') or path.name.endswith('.partial')
                    or path.stem.endswith('_draft') or path.name in ('.gitignore', 'file_sha256.json', 'eval_results.json')):
                continue
            # Supplemental local checks reuse the same instructor checkpoint;
            # the required reference_fcos copy is sufficient in the final ZIP.
            if 'detection_review' in path.parts and path.name.endswith(('.pth', '.pth.tar', '.pt')):
                continue
            if 'detection_review' in path.parts and path.name in ('team_readiness.json', 'component_sha256.json'):
                continue
            files.append(path)
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest_path = ROOT / 'results' / 'file_sha256.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    files.append(manifest_path)
    destination = ROOT / 'submission.zip'
    temporary = ROOT / 'submission.zip.partial'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(temporary) as archive:
        if archive.testzip():
            raise RuntimeError('ZIP integrity test failed')
        assert {name.split('/')[0] for name in archive.namelist()} == {'code', 'writeup', 'results'}
    temporary.replace(destination)
    print(f'READY: {destination} ({destination.stat().st_size / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bonus', action='store_true', help='require completed Swin and COCO experiments too')
    sys.exit(main(parser.parse_args()))
