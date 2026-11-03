"""Run, resume, evaluate, and collect the assignment's reproducible experiments.

Run from any directory: python code/run_experiments.py --device cuda:0
All experiment paths in submitted configurations remain relative to code/.
"""
import argparse
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
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)
    if (source / 'viz').exists():
        (destination / 'viz').mkdir(exist_ok=True)
        # A compact deterministic selection is enough for the report.
        for path in sorted((source / 'viz').glob('*.png'))[:12]:
            shutil.copy2(path, destination / 'viz' / path.name)


def main(args):
    os.chdir(CODE)
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    if args.reference:
        dest = ROOT / 'results' / 'reference_fcos'
        dest.mkdir(exist_ok=True)
        config = yaml.safe_load((ROOT / 'pretrained' / 'det_voc_res18.yaml').read_text())
        config['devices'] = [args.device]
        config['loader'] = {'batch_size': 8, 'num_workers': 2}
        config_path = dest / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        ckpt = dest / 'model_final.pth.tar'
        if not ckpt.exists():
            shutil.copy2(ROOT / 'pretrained' / 'det_voc_res18.pth.tar', ckpt)
        (dest / 'PROVENANCE.txt').write_text('Instructor-supplied reference checkpoint. Not a model trained by this team.\n')
        if not (dest / 'metrics.json').exists():
            run([sys.executable, '-u', 'eval_det.py', config_path, ckpt, '-v'], dest / 'evaluation.log')
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
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=['cnn','vit','simple','resnet18','fcos'])
    parser.add_argument('--reference', action='store_true')
    main(parser.parse_args())
