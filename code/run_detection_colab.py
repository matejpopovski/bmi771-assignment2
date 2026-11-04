"""Run required detector experiments with durable Drive logs and checkpoints.

Run inside an authenticated Colab runtime after mounting Google Drive. Images
stay on the runtime's local disk. Only logs/results/checkpoints go to Drive.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent


def restore_release(archive, root):
    required = ['data/trainval.json', 'data/test.json',
                'pretrained/det_voc_res18.yaml', 'pretrained/det_voc_res18.pth.tar']
    with ZipFile(archive) as source:
        for relative in required:
            candidates = [name for name in source.namelist()
                          if name == relative or name.endswith('/' + relative)]
            if len(candidates) != 1:
                raise ValueError(f'Release must contain exactly one {relative}')
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(candidates[0]) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def copy_artifacts(source, destination):
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def main(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select a CUDA GPU runtime before starting')
    if not args.release.is_file():
        raise FileNotFoundError(f'Canvas ZIP not found: {args.release}')
    backup = args.backup_dir.resolve()
    backup.mkdir(parents=True, exist_ok=True)
    logs = backup / 'logs'
    logs.mkdir(exist_ok=True)
    local_logs = ROOT / 'logs'
    if local_logs.is_symlink():
        if local_logs.resolve() != logs:
            raise ValueError('Existing logs symlink points at a different backup')
    elif local_logs.exists():
        if any(local_logs.iterdir()):
            raise ValueError('Use a fresh checkout; existing logs must be preserved')
        local_logs.rmdir()
        local_logs.symlink_to(logs, target_is_directory=True)
    else:
        local_logs.symlink_to(logs, target_is_directory=True)

    os.chdir(ROOT)
    restore_release(args.release, ROOT)
    # Restore completed runs and reference provenance after runtime replacement.
    for name in ['fcos', 'reference_fcos']:
        copy_artifacts(backup / 'results' / name, ROOT / 'results' / name)
    review = ROOT / 'results' / 'detection_review'
    review.mkdir(exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    metadata = {'git_commit': revision, 'gpu': torch.cuda.get_device_name(0),
                'torch': str(torch.__version__), 'python': sys.version,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'backup': str(backup), 'training_split': 'VOC2007 trainval',
                'evaluation_split': 'VOC2007 test (4952 images)'}
    (review / 'cloud_environment.json').write_text(json.dumps(metadata, indent=2))
    with (review / 'cloud_packages.txt').open('w') as output:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=output, check=True)
    print(json.dumps(metadata, indent=2), flush=True)

    def run(*command, output=None):
        print('RUN', ' '.join(map(str, command)), flush=True)
        if output:
            with output.open('w') as stream:
                subprocess.run(list(map(str, command)), check=True, stdout=stream,
                               stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='2'))
            print(output.read_text()[-6000:], flush=True)
        else:
            subprocess.run(list(map(str, command)), check=True)

    try:
        run(sys.executable, '-m', 'pytest', 'code/tests', '-q', output=review / 'cloud_tests.txt')
        run(sys.executable, 'code/download_datasets.py', '--detection-only')
        run(sys.executable, 'code/verify_detection_data.py')
        copy_artifacts(review, backup / 'results/detection_review')
        run(sys.executable, '-u', 'code/run_experiments.py', '--device', 'cuda:0',
            '--workers', args.workers, '--reference', '--experiments')
        copy_artifacts(ROOT / 'results/reference_fcos', backup / 'results/reference_fcos')
        reference = json.loads((ROOT / 'results/reference_fcos/metrics.json').read_text())
        if reference['num_images'] != 4952 or abs(reference['AP50'] - .609) > .02:
            raise RuntimeError('Reference verification differs substantially from the assignment; review before training')
        run(sys.executable, '-u', 'code/run_experiments.py', '--device', 'cuda:0',
            '--workers', args.workers, '--experiments', 'fcos')
        copy_artifacts(ROOT / 'results/fcos', backup / 'results/fcos')
        run(sys.executable, 'code/make_detection_report.py')
        print('REQUIRED DETECTION RUNS FINISHED. Review the measured results before submission.', flush=True)
    finally:
        for name in ['reference_fcos', 'fcos', 'detection_review']:
            copy_artifacts(ROOT / 'results' / name, backup / 'results' / name)
        for name in ['sean_detection.md']:
            source = ROOT / 'writeup' / name
            if source.exists():
                (backup / 'writeup').mkdir(exist_ok=True)
                shutil.copy2(source, backup / 'writeup' / name)
        figures = ROOT / 'results/figures'
        if figures.exists():
            target = backup / 'results/figures'
            target.mkdir(exist_ok=True)
            for source in figures.glob('*fcos*'):
                shutil.copy2(source, target / source.name)
        print('Persistent checkpoints/logs:', logs, flush=True)
        print('Collected artifacts:', backup / 'results', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--backup-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
