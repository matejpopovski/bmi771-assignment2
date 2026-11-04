"""Produce the detection owner's report section from real saved artifacts only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    results = ROOT / 'results'
    figures = results / 'figures'
    figures.mkdir(exist_ok=True)
    writeup = ROOT / 'writeup'
    writeup.mkdir(exist_ok=True)
    reference = read_json(results / 'reference_fcos/metrics.json')
    own = read_json(results / 'fcos/metrics.json')
    history_path = results / 'fcos/history.jsonl'
    history = []
    if history_path.exists():
        rows = {r['epoch']: r for r in map(json.loads, history_path.read_text().splitlines())}
        history = [rows[e] for e in sorted(rows)]
    config_path = results / 'fcos/config.yaml'
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else None
    required = ['model_final.pth.tar', 'config.yaml', 'training.log', 'history.jsonl', 'evaluation.log']
    complete = (own and own.get('num_images') == 4952 and history and config
                and all((results / 'fcos' / name).exists() for name in required)
                and (results / 'reference_fcos/reference_run.json').exists())
    lines = [
        '# Transformer block and FCOS: methods and measured results', '',
        'Prepared with AI assistance for GitHub user seanchang1529 to review and explain. '
        'This section does not assert that personal review or team cross-review has finished.', '',
        '**Status:** ' + ('Required experiment artifacts are present; review still required.' if complete
                        else 'Incomplete: a verified cloud reference run and/or independently trained detector artifacts are missing.'), '',
        '## TransformerBlock review', '',
        'The existing block was independently checked without changing its forward behavior or parameter shapes. '
        'For input X in (batch, height, width, channels) order, its residual equations are '
        '`Y = X + DropPath(Attention(LN1(X)))` and '
        '`Z = Y + DropPath(MLP(LN2(Y)))`. Normalization precedes both learned branches (Pre-LN). '
        'A positive window size restricts attention to nonoverlapping spatial windows; zero selects global attention. '
        'Spatial padding is added after LN1, then removed before the attention residual is added. '
        'The supplied padding convention uses zero tokens without an attention mask. This was preserved for checkpoint compatibility.', '',
        'Stochastic depth samples one keep/drop decision per original image and residual branch, broadcasts it over '
        'all spatial positions and channels, and divides kept branches by the keep probability. '
        'It is applied after windows are reassembled, so separate windows from one image share a decision. '
        'The two calls draw independent masks. Evaluation disables stochastic depth. '
        'The added tests compare outputs and input gradients with an explicit window-loop oracle, verify zero '
        'cross-window gradients, and check image-level masks on padded, rectangular inputs. '
        'No saved ViT checkpoint has been modified.', '',
        '## FCOS inference', '',
        'PointGenerator stores (y, x), whereas boxes use (x, y). At each FPN level, class and centerness logits '
        'are passed through sigmoid and combined as sqrt(p_class * p_center). Thresholding and top-K selection '
        'operate on class-location pairs, allowing multiple class hypotheses at one location. '
        'Predicted (left, top, right, bottom) distances are multiplied by stride and decoded around the location. '
        'Boxes are clipped to resized image bounds; dimensions at most 0.01 pixel are rejected. '
        'Class-aware NMS is applied across levels, followed by the per-image limit. Labels receive +1 for the '
        'supplied COCO-format category IDs, and boxes are rescaled to original image dimensions.', '',
        '## FCOS assignment and multitask loss', '',
        'A location must lie strictly inside a ground-truth box and its clipped center-sampling region. '
        'The region extends center_sampling_radius times the level stride from the box center. '
        'The maximum pixel distance to the box sides must fall within that level\'s configured regression range. '
        'The smallest-area eligible ground-truth box wins overlaps. Class targets are zero-based; unmatched '
        'locations use an all-zero class vector. Regression targets are side distances divided by stride.', '',
        'The loss sums sigmoid focal classification (alpha 0.25, gamma 2) over all locations, unweighted GIoU '
        'regression over positives, and BCE-with-logits centerness over positives. The centerness target is '
        'sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)). Each loss is normalized by the starter\'s EMA of positive '
        'counts (initial value 100, momentum 0.9, lower bound 1). Empty-positive batches retain classification '
        'loss and differentiable zero regression/centerness losses. Assignment is vectorized over locations '
        'and boxes, with only the image dimension looped.', '',
        '## Data and training protocol', '',
        'Only the supplied VOC2007 trainval annotations (5,011 images) may be used for training; all 4,952 '
        'test images are reserved for evaluation. The preflight audit verifies image counts, category IDs, '
        'annotation references, image existence and disjoint filenames. The instructor detector is evaluated '
        'separately and is never loaded by train_det.py. Training initializes the ResNet-18 backbone from '
        'ImageNet; FPN and detection heads start fresh. No torchvision or other external FCOS implementation is used.', '',
    ]
    if config:
        opt = config['opt']
        lines += [
            f"The resolved training configuration uses batch size {config['loader']['batch_size']}, "
            f"SGD learning rate {opt['learning_rate']}, momentum {opt['momentum']}, weight decay {opt['weight_decay']}, "
            f"{opt['warmup_epochs']} warmup plus {opt['epochs']} cosine epochs, and seed {config.get('seed', 771)}. "
            f"Short-side scale jitter is {config['input']['img_min_size']} with maximum long side "
            f"{config['input']['img_max_size']}. Backbone BN freezing is {config['model']['backbone_freeze_bn']}. "
            f"CUDA training uses {config['dtype']}; evaluation uses FP32.", '',
        ]
    else:
        lines += ['The planned configuration is code/configs/det_voc_fcos.yaml. Final training settings '
                  'will be described from results/fcos/config.yaml after the run.', '']
    lines += ['## Results', '', '| Detector | AP50 | AP50:95 | Images |',
              '| --- | ---: | ---: | ---: |']
    for name, metrics in [('Instructor checkpoint (evaluation only)', reference), ('Independently trained FCOS', own)]:
        if metrics:
            lines += [f"| {name} | {100*metrics['AP50']:.2f}% | {100*metrics['AP']:.2f}% | {metrics['num_images']} |"]
        else:
            lines += [f'| {name} | Not measured | Not measured | Not measured |']
    lines += ['']
    if reference and not (results / 'reference_fcos/reference_run.json').exists():
        lines += ['The reference row above is inherited from the teammate\'s earlier MPS run and has not '
                  'yet been independently reproduced in this work.', '']
    for name, metrics in [('Reference', reference), ('Trained detector', own)]:
        if not metrics:
            continue
        lines += [f"{name}: data-loop time {metrics['inference_seconds']:.2f} seconds, "
                  f"{metrics['images_per_second']:.2f} images/second; full evaluation including COCOeval "
                  f"{metrics['total_seconds']:.2f} seconds.", '']
        if 'model_seconds' in metrics:
            lines += [f"Synchronized model time: {metrics['model_seconds']:.2f} seconds "
                      f"({metrics['model_images_per_second']:.2f} images/second), device {metrics['device']}, "
                      f"GPU {metrics.get('gpu')}, PyTorch {metrics.get('torch')}. "
                      'This includes model preprocessing, backbone, heads, decode/NMS and resizing boxes back; '
                      'it excludes data loading, host transfers, visualization and COCOeval. '
                      'The cold first batch is included. Data-loop time includes those loading/transfer/visualization costs.', '']
    if own and reference:
        lines += [f"The independently trained detector differs from the fixed reference by "
                  f"{100*(own['AP50']-reference['AP50']):+.2f} AP50 percentage points. "
                  'This is a single-seed experiment; the reference is a correctness check, not an independently trained result.', '']
    if history:
        lines += [f"Recorded training: {len(history)} epochs; summed epoch wall time "
                  f"{sum(row['train']['seconds'] for row in history)/60:.2f} minutes. "
                  f"Final normalized total loss: {history[-1]['train']['final_loss']:.4f}.", '']
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
        for key in ['cls_loss', 'reg_loss', 'ctr_loss', 'final_loss']:
            ax.plot([r['epoch'] for r in history], [r['train'][key] for r in history], label=key)
        ax.set(xlabel='Epoch (including warmup)', ylabel='Mean normalized loss', title='VOC2007 trainval: FCOS losses')
        ax.legend()
        fig.savefig(figures / 'fcos_losses.png', dpi=180)
        plt.close(fig)
        lines += ['![FCOS training losses](../results/figures/fcos_losses.png)', '']
    for name in ['reference_fcos', 'fcos']:
        if name == 'reference_fcos' and not (results / name / 'reference_run.json').exists():
            continue
        paths = sorted((results / name / 'viz').glob('*.png'))[:4]
        if paths:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
            for ax in axes.flat:
                ax.axis('off')
            for ax, path in zip(axes.flat, paths):
                with Image.open(path) as image:
                    ax.imshow(image.copy())
                ax.set_title('VOC test image ' + path.stem)
            fig.savefig(figures / f'{name}_detections.png', dpi=170)
            plt.close(fig)
            lines += [f'![{name} test detections](../results/figures/{name}_detections.png)', '']
    lines += [
        'Examples use the first test images in deterministic loader order, with a display threshold of 0.5. '
        'AP uses every image and the configured score threshold 0.1; NMS IoU is 0.6. '
        'Qualitative error analysis must be added after viewing the actual trained-model examples.', '',
        '## Verification, provenance and remaining review', '',
        'Existing code was preliminary and AI-assisted. This work independently reviewed the assigned '
        'paths, added mathematical/behavioral tests, fixed stale reference reuse and measurement provenance, '
        'verified evaluation coverage, bounded sample rendering, and made checkpoint writes atomic. '
        'TransformerBlock and Matej\'s classifier artifacts were preserved. '
        'Test and package-version records are under results/detection_review/. '
        'AI assistance covered code review, tests, workflow changes and this draft; '
        'the user must verify understanding and confirm actual contributions under the course policy.', '',
        'Matej should review target assignment/loss conventions, the trainval/test audit, '
        'checkpoint provenance, the real detection results and the report section before integration. '
        'The combined PDF, team contribution confirmation and final ZIP audit remain separate team steps. '
        'No merge or Canvas submission is performed by this workflow.', '',
    ]
    output = writeup / 'sean_detection.md'
    output.write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
