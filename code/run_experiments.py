"""Run, resume, evaluate, and collect the assignment's reproducible experiments.

Run from any directory: python code/run_experiments.py --device cuda:0
All experiment paths in submitted configurations remain relative to code/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
EXPERIMENTS = {
    'cnn': ('cls_miniplaces_cnn', 'cls'),
    'vit': ('cls_miniplaces_vit', 'cls'),
    'simple': ('cls_miniplaces_simple', 'cls'),
    'resnet18': ('cls_miniplaces_resnet18', 'cls'),
    'fcos': ('det_voc_fcos', 'det'),
    'swin': ('det_voc_swin', 'det'),
    'coco': ('det_coco_fcos', 'det'),
}


def run(command, log):
    print('RUN', ' '.join(map(str, command)), flush=True)
    with open(log, 'a', buffering=1) as stream:
        result = subprocess.run([str(x) for x in command], cwd=CODE,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(Path(log).read_text()[-5000:], flush=True)
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def collect(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name.startswith('epoch_') or path.name == 'viz':
            continue
        target = destination / path.name
        if path.is_dir():
            # Published results may already contain an earlier run's events.
            # Replace the collected directory so TensorBoard cannot mix runs.
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    if (source / 'viz').exists():
        (destination / 'viz').mkdir(exist_ok=True)
        # A compact deterministic selection is enough for the report.
        for path in sorted((source / 'viz').glob('*.png'))[:12]:
            shutil.copy2(path, destination / 'viz' / path.name)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_reference(device, workers=2, force=False):
    """Reuse only a measured run with matching code, checkpoint and settings.

    Imported metrics without provenance must be reproduced. Archive their files
    before writing a new config so old measurements never acquire a new device.
    """
    dest = ROOT / 'results' / 'reference_fcos'
    dest.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'pretrained' / 'det_voc_res18.pth.tar'
    config = yaml.safe_load((ROOT / 'pretrained' / 'det_voc_res18.yaml').read_text())
    config['devices'] = [device]
    config['loader'] = {'batch_size': 8, 'num_workers': workers}
    import torch
    import torchvision
    signature = {
        'checkpoint_sha256': file_hash(source), 'config': config,
        'runtime': {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                    'gpu': torch.cuda.get_device_name(device) if device.startswith('cuda') else None},
        'sources': {str(path.relative_to(CODE)): file_hash(path) for path in
                    [CODE / 'eval_det.py', *sorted((CODE / 'det_libs').glob('*.py'))]},
        'annotations_sha256': file_hash(ROOT / 'data' / 'test.json'),
    }
    record = dest / 'reference_run.json'
    ckpt = dest / 'model_final.pth.tar'
    if (not force and record.exists() and (dest / 'metrics.json').exists()
            and ckpt.exists() and file_hash(ckpt) == signature['checkpoint_sha256']
            and json.loads(record.read_text()) == signature):
        print('ALREADY VERIFIED reference_fcos', flush=True)
        return dest
    previous = [path for path in dest.iterdir() if path.name != ckpt.name]
    if previous:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = ROOT / 'logs' / 'reference_fcos_previous' / stamp
        archive.mkdir(parents=True)
        for path in previous:
            shutil.move(str(path), archive / path.name)
        print('Preserved previous reference files:', archive, flush=True)
    config_path = dest / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if not ckpt.exists() or file_hash(ckpt) != signature['checkpoint_sha256']:
        shutil.copy2(source, ckpt)
    (dest / 'PROVENANCE.txt').write_text(
        'Instructor-supplied reference checkpoint. Evaluated only; not trained by this team.\n'
        f'SHA-256: {signature["checkpoint_sha256"]}\n')
    run([sys.executable, '-u', 'eval_det.py', config_path, ckpt, '-v'], dest / 'evaluation.log')
    record.write_text(json.dumps(signature, indent=2))
    return dest


def main(args):
    os.chdir(CODE)
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    if args.reference:
        evaluate_reference(args.device, args.workers, getattr(args, 'reference_force', False))
    for name in args.experiments:
        config_name, kind = EXPERIMENTS[name]
        config_path = CODE / 'configs' / (config_name + '.yaml')
        config = yaml.safe_load(config_path.read_text())
        config['devices'] = [args.device]
        config['loader']['num_workers'] = args.workers
        if args.device == 'mps':
            config['dtype'] = 'fp32'
            config['loader']['batch_size'] = min(config['loader']['batch_size'], 128 if kind == 'cls' else 8)
        # Resolved runtime overrides are saved as a new YAML, preserving originals.
        runtime_config = CODE / 'configs' / (config_name + '_run.yaml')
        runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))
        folder = ROOT / 'logs' / (runtime_config.stem + '_assignment2')
        folder.mkdir(parents=True, exist_ok=True)
        dest = ROOT / 'results' / name
        if (dest / 'model_final.pth.tar').exists() and (dest / 'metrics.json').exists():
            print('ALREADY COMPLETE', name, flush=True)
            continue
        if not (folder / 'model_final.pth.tar').exists():
            command = [sys.executable, '-u', f'train_{kind}.py', runtime_config,
                       '--output', 'assignment2', '-c', '5' if kind == 'cls' else '1', '-p', '50']
            epochs = sorted(folder.glob('epoch_*.pth.tar'))
            if epochs:
                command += ['--resume', epochs[-1]]
            started = time.time()
            run(command, folder / 'training.log')
            (folder / 'run_metadata.json').write_text(json.dumps({
                'device': args.device, 'wall_seconds_this_invocation': time.time() - started,
                'experiment': name, 'seed': 771}, indent=2))
        if not (folder / 'metrics.json').exists():
            command = [sys.executable, '-u', f'eval_{kind}.py', runtime_config, folder]
            if kind == 'det':
                command += ['-v']
            run(command, folder / 'evaluation.log')
        collect(folder, dest)
        print('COMPLETE', name, (dest / 'metrics.json').read_text(), flush=True)
    print('EXPERIMENTS FINISHED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--experiments', nargs='*', choices=list(EXPERIMENTS), default=['cnn','vit','simple','resnet18','fcos'])
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--reference-force', action='store_true', help='repeat reference evaluation even if provenance matches')
    main(parser.parse_args())
