"""Independent behavioral checks for Sean's shared TransformerBlock review.

No implementation or checkpoint changes are needed for these tests.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from cls_libs.modeling import TransformerBlock


@pytest.mark.parametrize('height,width,window', [(4, 6, 0), (4, 6, 2), (5, 7, 3), (1, 1, 4)])
def test_pre_ln_block_against_explicit_windows_and_gradients(height, width, window):
    torch.manual_seed(771)
    block = TransformerBlock(8, 2, window_size=window).double().eval()
    x = torch.randn(2, height, width, 8, dtype=torch.double, requires_grad=True)
    normalized = block.norm1(x)
    if window:
        # Explicit spatial loops avoid depending on the partition helpers.
        hp = ((height + window - 1) // window) * window
        wp = ((width + window - 1) // window) * window
        padded = F.pad(normalized, (0, 0, 0, wp-width, 0, hp-height))
        rows = []
        for y in range(0, hp, window):
            rows.append(torch.cat([block.attn(padded[:, y:y+window, z:z+window])
                                   for z in range(0, wp, window)], dim=2))
        attended = torch.cat(rows, dim=1)[:, :height, :width]
    else:
        attended = block.attn(normalized)
    expected = x + attended
    expected = expected + block.mlp(block.norm2(expected))
    actual = block(x)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad((actual*probe).sum(), x, retain_graph=True)[0],
                               torch.autograd.grad((expected*probe).sum(), x)[0])


def test_local_block_has_no_cross_window_input_gradient():
    block = TransformerBlock(8, 2, window_size=2).double().eval()
    x = torch.randn(1, 4, 6, 8, dtype=torch.double, requires_grad=True)
    gradient = torch.autograd.grad(block(x)[:, :2, :2].square().sum(), x)[0]
    assert torch.count_nonzero(gradient[:, :2, 2:]) == 0
    assert torch.count_nonzero(gradient[:, 2:]) == 0
    assert gradient[:, :2, :2].abs().sum() > 0


class Ones(nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


def test_drop_path_is_per_original_image_after_window_reassembly():
    block = TransformerBlock(8, 2, window_size=2, drop_path=.5).train()
    block.attn = Ones()
    # Isolate the attention residual; keep the MLP in the execution path.
    nn.init.zeros_(block.mlp.fc2.weight)
    nn.init.zeros_(block.mlp.fc2.bias)
    torch.manual_seed(771)
    out = block(torch.zeros(64, 5, 7, 8))
    assert set(out.unique().tolist()) == {0., 2.}
    assert all(sample.unique().numel() == 1 for sample in out)
    block.eval()
    torch.testing.assert_close(block(torch.zeros(2, 5, 7, 8)), torch.ones(2, 5, 7, 8))
