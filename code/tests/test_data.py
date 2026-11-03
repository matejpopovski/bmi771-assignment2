"""Check the COCO extension at the annotation/model label boundary."""
import json
from pathlib import Path
import sys

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from det_libs.dataset import build_dataset
from det_libs.transforms import ConvertAnnotations, RandomHorizontalFlip


def test_coco_labels_crowds_and_degenerate_boxes(tmp_path):
    Image.new('RGB', (40, 30)).save(tmp_path / 'fixture.jpg')
    annotations = [
        dict(id=1, image_id=7, category_id=80, bbox=[5, 6, 10, 12], area=120, iscrowd=0),
        dict(id=2, image_id=7, category_id=1, bbox=[41, 6, 10, 12], area=120, iscrowd=0),
        dict(id=3, image_id=7, category_id=2, bbox=[0, 0, 30, 20], area=600, iscrowd=1),
    ]
    data = dict(images=[dict(id=7, file_name='fixture.jpg', width=40, height=30)],
                annotations=annotations,
                categories=[dict(id=i, name=f'class-{i}') for i in range(1, 81)])
    (tmp_path / 'val2017_subset.json').write_text(json.dumps(data))
    dataset = build_dataset('COCO2017', 'val2017_subset', str(tmp_path), str(tmp_path))
    image, target = dataset[0]
    assert image.shape == (3, 30, 40)
    assert target['labels'].tolist() == [79]
    torch.testing.assert_close(target['boxes'], torch.tensor([[5., 6., 15., 18.]]))
    assert target['area'].tolist() == [120]
    assert target['iscrowd'].tolist() == [0]
    assert dataset.get_cls_names()[79] == 'class-80'
    # Evaluation still reads all original GT, including the crowd annotation.
    assert len(dataset.coco.anns) == 3


def test_detection_flip_and_empty_annotation_shapes():
    image = Image.new('RGB', (40, 30))
    target = {'boxes': torch.tensor([[5., 6., 15., 18.]])}
    _, flipped = RandomHorizontalFlip(p=1)(image, target)
    torch.testing.assert_close(flipped['boxes'], torch.tensor([[25., 6., 35., 18.]]))
    _, empty = ConvertAnnotations()(image, {'image_id': 1, 'annotations': []})
    assert empty['boxes'].shape == (0, 4)
    assert empty['labels'].shape == empty['area'].shape == empty['iscrowd'].shape == (0,)
