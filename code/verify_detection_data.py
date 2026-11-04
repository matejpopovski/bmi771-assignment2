"""Validate the supplied VOC splits before fitting or measuring a detector."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def audit(data, check_images=True):
    records, names = {}, {}
    for split, expected in [('trainval', 5011), ('test', 4952)]:
        path = data / f'{split}.json'
        annotation = json.loads(path.read_text())
        images = annotation['images']
        ids = [image['id'] for image in images]
        id_set = set(ids)
        filenames = [image['file_name'] for image in images]
        categories = {category['id'] for category in annotation['categories']}
        if len(images) != expected or len(set(ids)) != expected or len(set(filenames)) != expected:
            raise ValueError(f'{split}: expected {expected} unique images')
        if categories != set(range(1, 21)):
            raise ValueError(f'{split}: expected category IDs 1..20')
        if any(a['image_id'] not in id_set or a['category_id'] not in categories
               for a in annotation['annotations']):
            raise ValueError(f'{split}: invalid annotation references')
        image_root = (data / 'VOCdevkit/VOC2007/JPEGImages').resolve()
        paths = [(image_root / name).resolve() for name in filenames]
        if any(not path.is_relative_to(image_root) for path in paths):
            raise ValueError('Image filename escapes the VOC directory')
        if check_images and any(not path.is_file() for path in paths):
            raise ValueError(f'{split}: missing VOC images; run download_datasets.py --detection-only')
        names[split] = set(filenames)
        records[split] = {'images': len(images), 'annotations': len(annotation['annotations']),
                          'categories': len(categories), 'images_checked': check_images,
                          'annotations_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if names['trainval'] & names['test']:
        raise ValueError('Training and test image filenames overlap')
    records['split_filename_overlap'] = 0
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--annotations-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/detection_review/data_audit.json')
    args = parser.parse_args()
    result = audit(args.data, not args.annotations_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
