"""Count approximate forward MACs for Conv/Linear layers and attention products."""
import json
from pathlib import Path
import torch
from torch import nn
from cls_libs.config import load_config
from cls_libs.utils import build_model
from cls_libs.modeling import Attention

ROOT=Path(__file__).resolve().parent.parent


def main():
    torch.set_num_threads(2)
    counts={}
    for name in ['simple','cnn','vit','resnet18']:
        cfg=load_config(ROOT/'code/configs'/f'cls_miniplaces_{name}.yaml')
        model=build_model(cfg,pretrained=False).eval()
        parts={'convolution':0,'linear':0,'attention_products':0}
        def count(module,args,output):
            if isinstance(module,nn.Conv2d):
                parts['convolution']+=output.numel()*module.kernel_size[0]*module.kernel_size[1]*module.in_channels//module.groups
            elif isinstance(module,nn.Linear):
                parts['linear']+=output.numel()*module.in_features
            elif isinstance(module,Attention):
                b,h,w,c=args[0].shape
                parts['attention_products']+=2*b*(h*w)**2*c
        hooks=[m.register_forward_hook(count) for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear,Attention))]
        with torch.no_grad(): model(torch.zeros(1,3,128,128))
        for hook in hooks: hook.remove()
        counts[name]={'parameters':sum(p.numel() for p in model.parameters()),'macs':sum(parts.values()),'components':parts}
    result={'input_shape':[1,3,128,128],
            'method':'One MAC is one multiply-accumulate. Counts Conv2d, Linear, QK^T and attention-times-V; excludes normalization, nonlinearities, softmax, pooling, bias and residual additions. This is an operation estimate, not measured runtime.',
            'models':counts}
    (ROOT/'results/classification_compute.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
