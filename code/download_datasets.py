"""Download the provided MiniPlaces and VOC-2007 datasets, with checked exits."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)


def run(*args):
    print('RUN', ' '.join(args), flush=True)
    subprocess.run(args, cwd=DATA, check=True)


def main(classification_only=False):
    if not (DATA / 'images' / 'val').is_dir():
        if not (DATA / 'data.tar.gz').is_file():
            run(sys.executable, '-m', 'gdown', '16GYHdSWS3iMYwMPv5FpeDZN2rH7PR0F2', '-O', 'data.tar.gz', '--quiet')
        run('tar', '-xzf', 'data.tar.gz')
    for split in ['train', 'val']:
        if not (DATA / f'{split}.txt').exists():
            run('curl', '-fLsS', f'https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/{split}.txt', '-o', f'{split}.txt')
    archives = [] if classification_only else [('VOCtrainval_06-Nov-2007.tar','trainval'), ('VOCtest_06-Nov-2007.tar','test')]
    for archive, marker in archives:
        if not (DATA / 'VOCdevkit' / 'VOC2007' / 'ImageSets' / 'Main' / f'{marker}.txt').exists():
            if not (DATA / archive).is_file():
                run('curl', '-fLsS', '--retry', '3', f'https://data.brainchip.com/dataset-mirror/voc/{archive}', '-o', archive)
            run('tar', '-xf', archive)
    print('DATASETS READY', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification-only', action='store_true', help='download only MiniPlaces')
    main(parser.parse_args().classification_only)
