"""Prepare a seeded COCO-2017 feasibility subset using official annotations/images.

Downloads only selected images. Train and validation are sampled independently
from the official splits; validation images are never used for fitting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import urllib.request
import zipfile


def download(url, path):
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, temp)
            temp.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise


def main(args):
    root = Path(args.root)
    archive = root / 'annotations_trainval2017.zip'
    download('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip', archive)
    counts = {'train2017': args.train_images, 'val2017': args.val_images}
    mapping = None
    for split, count in counts.items():
        with zipfile.ZipFile(archive) as z:
            source = json.loads(z.read(f'annotations/instances_{split}.json'))
        categories = sorted(source['categories'], key=lambda c: c['id'])
        mapping = {c['id']: index + 1 for index, c in enumerate(categories)}
        selected = random.Random(args.seed).sample(sorted(source['images'], key=lambda i: i['id']), count)
        selected_ids = {image['id'] for image in selected}
        annotations = [dict(a, category_id=mapping[a['category_id']]) for a in source['annotations'] if a['image_id'] in selected_ids]
        # Remap sparse COCO category IDs to contiguous 1..80 in BOTH GT and
        # predictions. The original mapping is retained for reproducibility.
        output = dict(info=source.get('info', {}), licenses=source.get('licenses', []),
                      images=selected, annotations=annotations,
                      categories=[dict(c, id=mapping[c['id']]) for c in categories])
        json_path = root / f'{split}_subset.json'
        json_path.write_text(json.dumps(output))
        def fetch(image):
            download(f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{image['file_name']}", root / 'images' / image['file_name'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, selected))
        print(split, len(selected), 'images', len(annotations), 'annotations', flush=True)
    (root / 'subset_manifest.json').write_text(json.dumps({
        'seed': args.seed, 'counts': counts, 'original_to_contiguous_category_id': mapping,
        'source': 'COCO 2017 official train2017 and val2017; uniform random subset'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data/coco')
    parser.add_argument('--train-images', type=int, default=5000)
    parser.add_argument('--val-images', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=771)
    main(parser.parse_args())
