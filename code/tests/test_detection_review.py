"""Additional inference/loss and full-evaluator checks, independent of VOC files."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn
from det_libs.modeling import FCOS
from det_libs.engine import evaluate
from det_libs.dataset import build_dataloader
from test_models import bare_detector


def test_nms_suppresses_same_class_across_levels_but_keeps_other_class():
    model = bare_detector()
    points = [torch.tensor([[[8., 8.]]])] * 2
    cls = [torch.full((1, 2, 1, 1), 9.), torch.full((1, 2, 1, 1), 8.)]
    reg = [torch.ones(1, 4, 1, 1), torch.full((1, 4, 1, 1), .5)]
    ctr = [torch.full((1, 1, 1, 1), 10.)] * 2
    out = model.inference(points, [torch.tensor(4.), torch.tensor(8.)], cls, reg, ctr, [(32, 32)])[0]
    assert sorted(out['labels'].tolist()) == [1, 2]
    torch.testing.assert_close(out['boxes'], torch.tensor([[4., 4., 12., 12.]]).expand(2, -1))


def test_giou_nonzero_error_has_correct_descent_direction():
    model = bare_detector(1)
    pred = torch.full((1, 4, 1, 1), .5, requires_grad=True)
    losses = model.compute_loss(
        [{'boxes': torch.tensor([[0., 0., 16., 16.]]), 'labels': torch.tensor([0])}],
        [torch.tensor([[[8., 8.]]])], torch.tensor([8.]), torch.tensor([[0., 64.]]),
        [torch.zeros(1, 1, 1, 1, requires_grad=True)], [pred],
        [torch.zeros(1, 1, 1, 1, requires_grad=True)])
    # A centered half-size box has IoU 1/4 and needs larger offsets.
    assert losses['reg_loss'].item() == pytest.approx(.75)
    losses['reg_loss'].backward()
    assert (pred.grad < 0).all()


class FixtureDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.ones(3, 24, 24), {'image_id': torch.tensor([index+1])}

    def get_cls_names(self):
        return ('object',)


class ExactDetector(nn.Module):
    num_classes = 1

    def __init__(self, empty=False):
        super().__init__()
        self.empty = empty

    def forward(self, images, targets=None):
        count = 0 if self.empty else 1
        return [{'boxes': torch.tensor([[2., 3., 12., 15.]])[:count],
                 'scores': torch.tensor([.99])[:count],
                 'labels': torch.tensor([1])[:count]} for _ in images]


def annotation_fixture(tmp_path):
    path = tmp_path/'gt.json'
    path.write_text(json.dumps({
        'images': [{'id': i, 'width': 24, 'height': 24} for i in range(1, 4)],
        'categories': [{'id': 1, 'name': 'object'}],
        'annotations': [{'id': i, 'image_id': i, 'category_id': 1,
                         'bbox': [2, 3, 10, 12], 'area': 120, 'iscrowd': 0} for i in range(1, 4)]}))
    return path


@pytest.mark.parametrize('empty', [False, True])
def test_evaluator_full_coverage_timing_and_visualization_without_printing(tmp_path, empty):
    loader = build_dataloader(FixtureDataset(), False, batch_size=2, num_workers=0)
    metrics = evaluate(loader, ExactDetector(empty), str(tmp_path/'eval_results.json'),
                       str(annotation_fixture(tmp_path)), torch.device('cpu'),
                       print_freq=0, viz=True, viz_limit=2)
    assert metrics['num_images'] == 3  # includes final short batch
    assert metrics['AP50'] == pytest.approx(0. if empty else 1.)
    assert metrics['model_seconds'] > 0
    assert metrics['total_seconds'] >= metrics['inference_seconds'] >= metrics['model_seconds']
    assert metrics['visualizations'] == len(list((tmp_path/'viz').glob('*.png'))) == 2


def test_evaluator_rejects_dropped_test_images(tmp_path):
    loader = torch.utils.data.DataLoader(FixtureDataset(), batch_size=2, drop_last=True,
                                        collate_fn=lambda batch: tuple(zip(*batch)))
    with pytest.raises(ValueError, match='every dataset image'):
        evaluate(loader, ExactDetector(), str(tmp_path/'eval_results.json'),
                 str(annotation_fixture(tmp_path)), torch.device('cpu'), print_freq=0)
