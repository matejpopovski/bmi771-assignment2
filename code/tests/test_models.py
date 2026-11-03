"""Numerical and behavioral checks independent of any training dataset."""
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from cls_libs.modeling import Attention, TransformerBlock, SimpleViT, SceneCNN, SimpleNet
from cls_libs.blocks import window_partition, window_unpartition
from cls_libs.utils import build_optimizer
from cls_libs.config import load_default_config
from det_libs.modeling import FCOS
from det_libs.point_generator import PointGenerator


def bare_detector(classes=2):
    model = FCOS.__new__(FCOS)
    nn.Module.__init__(model)
    model.num_classes = classes
    model.center_sampling_radius = 1.5
    model.loss_normalizer = 1.0
    model.loss_normalizer_momentum = 0.0
    model.score_thresh = 0.1
    model.nms_thresh = 0.5
    model.topk_candidates = 1000
    model.detections_per_img = 100
    return model


@pytest.mark.parametrize('heads', [1, 2, 4])
def test_attention_against_pytorch_oracle(heads):
    torch.manual_seed(11)
    model = Attention(12, heads).double()
    x = torch.randn(2, 3, 5, 12, dtype=torch.double, requires_grad=True)
    q, k, v = model.qkv(x).reshape(2, 15, 3, heads, 12 // heads).permute(2, 0, 3, 1, 4)
    expected = F.scaled_dot_product_attention(q, k, v)
    expected = model.proj(expected.transpose(1, 2).reshape(2, 3, 5, 12))
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), x)[0])


def test_bad_heads():
    with pytest.raises(ValueError):
        Attention(13, 4)


@pytest.mark.parametrize('hw', [(8, 8), (5, 7), (1, 1)])
def test_windows_roundtrip(hw):
    x = torch.randn(2, *hw, 12)
    windows, padded = window_partition(x, 4)
    torch.testing.assert_close(window_unpartition(windows, 4, padded, hw), x)


@pytest.mark.parametrize('window', [0, 4])
def test_transformer_residual_identity(window):
    model = TransformerBlock(12, 3, window_size=window, drop_path=0.2).eval()
    nn.init.zeros_(model.attn.proj.weight)
    nn.init.zeros_(model.attn.proj.bias)
    nn.init.zeros_(model.mlp.fc2.weight)
    nn.init.zeros_(model.mlp.fc2.bias)
    x = torch.randn(2, 5, 7, 12)
    torch.testing.assert_close(model(x), x)


def test_local_attention_does_not_mix_windows():
    model = TransformerBlock(12, 3, window_size=2).eval()
    x = torch.randn(1, 4, 4, 12)
    y = x.clone()
    y[:, :2, :2] += torch.randn(1, 2, 2, 12)
    torch.testing.assert_close(model(x)[:, 2:, 2:], model(y)[:, 2:, 2:])


@pytest.mark.parametrize('constructor', [SceneCNN, SimpleNet, SimpleViT])
def test_classifiers_backward_and_optimizer(constructor):
    torch.manual_seed(7)
    model = constructor(num_classes=7)
    logits = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 7)
    loss = F.cross_entropy(logits, torch.tensor([0, 6]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    build_optimizer(model, load_default_config()['opt']).step()


def test_vit_position_interpolation():
    model = SimpleViT(embed_dim=24, depth=2, num_heads=3)
    assert model(torch.randn(2, 3, 144, 160)).shape == (2, 100)


def test_point_coordinates_non_square():
    gen = PointGenerator(128, [8], [(0, 64)])
    points, _, _ = gen([torch.zeros(1, 8, 2, 3)])
    assert points[0][1, 2].tolist() == [12, 20]


def test_decode_xy_stride_labels_multiclass_nms():
    model = bare_detector()
    points = [torch.tensor([[[10., 20.], [10., 20.]]])]
    cls = [torch.tensor([[[[8., 7.]], [[6., -30.]]]])]
    reg = [torch.tensor([[[[1., 1.]], [[2., 2.]], [[3., 3.]], [[4., 4.]]]])]
    ctr = [torch.full((1, 1, 1, 2), 8.)]
    out = model.inference(points, [torch.tensor(2.)], cls, reg, ctr, [(40, 60)])[0]
    assert out['labels'].tolist() == [1, 2]
    torch.testing.assert_close(out['boxes'], torch.tensor([[18., 6., 26., 18.]]).expand(2, -1))
    assert torch.all(out['scores'][:-1] >= out['scores'][1:])


def test_inference_empty_and_clipping():
    model = bare_detector(1)
    points = [torch.tensor([[[4., 12.]]])]
    reg = [torch.full((1, 4, 1, 1), 10.)]
    ctr = [torch.ones(1, 1, 1, 1) * 10]
    low = model.inference(points, [torch.tensor(8.)], [torch.full((1, 1, 1, 1), -40.)], reg, ctr, [(16, 24)])[0]
    assert low['boxes'].shape == (0, 4) and low['labels'].dtype == torch.int64
    high = model.inference(points, [torch.tensor(8.)], [torch.full((1, 1, 1, 1), 10.)], reg, ctr, [(16, 24)])[0]
    torch.testing.assert_close(high['boxes'], torch.tensor([[0., 0., 24., 16.]]))


def loss_inputs():
    return ([torch.tensor([[[8., 16.]]])], torch.tensor([8.]), torch.tensor([[0., 64.]]),
            [torch.zeros(1, 2, 1, 1, requires_grad=True)],
            [torch.ones(1, 4, 1, 1, requires_grad=True)],
            [torch.zeros(1, 1, 1, 1, requires_grad=True)])


def test_loss_analytic_exact_box_and_centerness():
    model = bare_detector()
    args = loss_inputs()
    target = [{'boxes': torch.tensor([[8., 0., 24., 16.]]), 'labels': torch.tensor([1])}]
    losses = model.compute_loss(target, *args)
    assert losses['reg_loss'].item() < 1e-6
    assert losses['ctr_loss'].item() == pytest.approx(math.log(2))
    assert losses['cls_loss'].item() == pytest.approx(0.25 * math.log(2))
    losses['final_loss'].backward()
    for tensor in args[3] + args[4] + args[5]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_empty_targets_backward():
    model = bare_detector()
    args = loss_inputs()
    target = [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
    losses = model.compute_loss(target, *args)
    assert losses['reg_loss'].item() == losses['ctr_loss'].item() == 0
    assert losses['cls_loss'].item() > 0
    losses['final_loss'].backward()
    for tensor in args[3] + args[4] + args[5]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_smallest_area_assignment_and_range():
    model = bare_detector()
    args = loss_inputs()
    targets = [{'boxes': torch.tensor([[0., 0., 32., 32.], [8., 0., 24., 16.]]),
                'labels': torch.tensor([0, 1])}]
    losses = model.compute_loss(targets, *args)
    assert losses['reg_loss'].item() < 1e-6
    args = list(loss_inputs())
    args[2] = torch.tensor([[64., 128.]])
    losses = model.compute_loss(targets, *args)
    assert losses['reg_loss'].item() == losses['ctr_loss'].item() == 0


def test_center_sampling_excludes_box_edges():
    model = bare_detector()
    args = list(loss_inputs())
    args[0] = [torch.tensor([[[2., 2.]]])]
    targets = [{'boxes': torch.tensor([[0., 0., 40., 40.]]), 'labels': torch.tensor([0])}]
    losses = model.compute_loss(targets, *args)
    assert losses['reg_loss'].item() == losses['ctr_loss'].item() == 0


def test_full_fcos_train_eval_without_downloads(tmp_path):
    from det_libs.config import load_config
    path = tmp_path / 'config.yaml'
    path.write_text('model:\n  backbone_weights: null\n  fpn_feats_dim: 32\ninput:\n  img_min_size: [64]\n  img_max_size: 96\n')
    model = FCOS(**load_config(path)['model'])
    images = [torch.rand(3, 64, 80), torch.rand(3, 72, 64)]
    targets = [{'boxes': torch.tensor([[8., 8., 48., 48.]]), 'labels': torch.tensor([0])},
               {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
    losses = model(images, targets)
    assert all(torch.isfinite(value) for value in losses.values())
    losses['final_loss'].backward()
    assert model.cls_head.cls_logits.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 2
    assert all(set(out) == {'boxes','scores','labels'} for out in outputs)


def test_swin_backbone_and_optimizer():
    from det_libs.modeling import SwinBackbone
    from det_libs.utils import build_optimizer as det_optimizer
    from det_libs.config import load_default_config as det_defaults
    model = SwinBackbone(weights=None).eval()
    with torch.no_grad():
        features = model(torch.randn(1, 3, 64, 96))
    assert [tuple(x.shape) for x in features.values()] == [(1,192,8,12),(1,384,4,6),(1,768,2,3)]
    det_optimizer(model, det_defaults()['opt'])


def test_off_center_centerness_and_stride_gradient():
    model = bare_detector()
    points = [torch.tensor([[[8., 12.]]])]
    cls = [torch.zeros(1, 2, 1, 1, requires_grad=True)]
    reg = [torch.tensor([[[[0.5]], [[1.]], [[1.5]], [[1.]]]], requires_grad=True)]
    ctr = [torch.tensor([[[[0.8]]]], requires_grad=True)]
    target = [{'boxes': torch.tensor([[8., 0., 24., 16.]]), 'labels': torch.tensor([0])}]
    losses = model.compute_loss(target, points, torch.tensor([8.]), torch.tensor([[0.,64.]]), cls, reg, ctr)
    truth_center = math.sqrt(1 / 3)
    expected_ctr = F.binary_cross_entropy_with_logits(torch.tensor(0.8), torch.tensor(truth_center))
    torch.testing.assert_close(losses['ctr_loss'], expected_ctr)
    assert losses['reg_loss'].item() < 1e-6
    losses['ctr_loss'].backward()
    assert ctr[0].grad.item() == pytest.approx(torch.sigmoid(torch.tensor(.8)).item() - truth_center)


def test_assignment_two_levels_and_two_images():
    model = bare_detector()
    # Same physical location; only the high-range level may match this box.
    points = [torch.tensor([[[40.,40.]]]), torch.tensor([[[40.,40.]]])]
    cls = [torch.zeros(2,2,1,1,requires_grad=True) for _ in range(2)]
    reg = [torch.full((2,4,1,1),value,requires_grad=True) for value in (5.,2.5)]
    ctr = [torch.zeros(2,1,1,1,requires_grad=True) for _ in range(2)]
    targets = [{'boxes':torch.tensor([[0.,0.,80.,80.]]),'labels':torch.tensor([1])},
               {'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long)}]
    losses=model.compute_loss(targets,points,torch.tensor([8.,16.]),torch.tensor([[0.,32.],[32.,64.]]),cls,reg,ctr)
    assert losses['reg_loss'].item() < 1e-6
    losses['final_loss'].backward()
    # Focal-loss gradient is negative only for the assigned foreground class.
    assert cls[1].grad[0,1,0,0] < 0
    assert cls[1].grad[0,0,0,0] > 0
    assert (cls[0].grad > 0).all()
    assert (cls[1].grad[1] > 0).all()
    assert ctr[1].grad[0,0,0,0].item() == pytest.approx(-.5)
    assert ctr[1].grad[1].abs().sum() == 0


def test_stochastic_depth_per_sample_and_eval():
    from cls_libs.blocks import DropPath
    layer=DropPath(.5)
    x=torch.ones(100,2,3,4)
    torch.manual_seed(771)
    out=layer(x)
    assert set(out.unique().tolist()) == {0.,2.}
    assert all(t.unique().numel()==1 for t in out)
    layer.eval()
    torch.testing.assert_close(layer(x),x)


def test_resnet_checkpoint_construction_does_not_download(monkeypatch):
    import torchvision.models._api as model_api
    from cls_libs.utils import build_model
    def forbid_download(*args, **kwargs):
        raise AssertionError('Checkpoint evaluation must work without downloading weights')
    monkeypatch.setattr(model_api, 'load_state_dict_from_url', forbid_download)
    cfg = load_default_config()
    cfg['model_type'] = 'RESNET18'
    model = build_model(cfg, pretrained=False)
    assert model.fc.out_features == 100
