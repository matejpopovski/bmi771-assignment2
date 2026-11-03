"""Build Matej's classification contribution from measured, audited artifacts."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import yaml
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from make_report import make_plots, history, NAMES
from verify_classification import audit

ROOT = Path(__file__).resolve().parent.parent
ORDER = ['simple', 'cnn', 'vit', 'resnet18']


def main():
    verification = audit()
    complete = verification['ready']
    make_plots()
    summaries = verification['experiments']
    compute_path=ROOT/'results/classification_compute.json'
    compute=json.loads(compute_path.read_text())['models'] if compute_path.exists() else {}
    output = ROOT / 'writeup' / ('matej_classification.pdf' if complete else 'matej_classification_draft.pdf')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MainTitle', fontSize=24, leading=28, spaceAfter=14,
                              textColor=colors.HexColor('#153d4c'), fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='TextBody', fontSize=10, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallCaption', fontSize=8.5, leading=11, spaceAfter=10,
                              textColor=colors.HexColor('#52636a')))
    styles.add(ParagraphStyle(name='TableCell', fontSize=8.5, leading=11))
    styles['Heading1'].textColor = colors.HexColor('#153d4c')
    story = []
    def p(text, style='TextBody'):
        story.append(Paragraph(text, styles[style]))
    def heading(text): p(text, 'Heading1')
    def table(rows, widths):
        item = Table([[Paragraph(str(c), styles['TableCell']) for c in row] for row in rows],
                     colWidths=widths, repeatRows=1)
        item.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef1')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#74909d')),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dae1e5'))]))
        story.extend([item, Spacer(1, 12)])
    def figure(filename, caption, width=508):
        path=ROOT/'results/figures'/filename
        if path.exists():
            w,h=PILImage.open(path).size
            story.append(Image(str(path),width=width,height=width*h/w))
            p(caption,'SmallCaption')
    def setting(name):
        path=ROOT/'results'/name/'config.yaml'
        if not path.exists():
            matches=list((ROOT/'logs').glob(f'cls_miniplaces_{name}_run_assignment2/config.yaml'))
            path=matches[0] if matches else path
        return yaml.safe_load(path.read_text()) if path.exists() else None

    p('Image classification:<br/>CNNs and a vision Transformer', 'MainTitle')
    p('BMI / CS 771 Assignment 2 — classification contribution prepared for Matej Popovski')
    if not complete:
        p('<b>DRAFT: classifier experiments are still incomplete. No pending value is a measured result.</b>')
    heading('Scope and evaluation protocol')
    p('This contribution covers custom CNN design/training (2 points), comparison with SimpleNet and fine-tuned ResNet-18 (1), multi-head attention (1), and ViT design/training (2): six required points. The teammate owns the one-point Transformer-block task and five-point FCOS tasks. This is a component report for integration into the final team PDF, not the complete assignment submission.')
    p('MiniPlaces has 100 training classes, 100,000 training images, and 10,000 labeled validation images. Only the training split updates model parameters. Every reported classifier is evaluated on all 10,000 validation images using a single center crop, without flip averaging. CNN, SimpleNet, and ViT use random initialization; only the designated ResNet-18 comparison uses ImageNet pretraining.')
    p('All runs use seed 771. Training applies a random resized 128 x 128 crop, horizontal flip, color jitter 0.15, and rotation up to 15 degrees. Validation resizes the short side to 144 and center-crops 128 x 128. Both use ImageNet channel normalization. Final scheduled checkpoints are reported, rather than selecting the best validation epoch after seeing results.')
    heading('Measured validation results')
    rows=[['Model','Params (M)','MACs (M)','Top-1','Top-5','Epochs']]
    for name in ORDER:
        item=summaries.get(name)
        rows.append([NAMES[name],f"{item['parameters']/1e6:.2f}" if item else 'Pending',
                     f"{compute[name]['macs']/1e6:.1f}" if name in compute else '-',
                     f"{item['metrics']['top1']:.2f}%" if item else 'Pending',
                     f"{item['metrics']['top5']:.2f}%" if item else 'Pending',
                     item['epochs'] if item else 'Pending'])
    table(rows,[180,70,80,65,65,48])
    p('MACs estimate one forward pass at 128 x 128. They count convolution/linear operations and the two attention matrix products, excluding normalization, nonlinearities, softmax, pooling, bias and residual additions. One MAC is one multiply-accumulate; this count is not a runtime measurement.','SmallCaption')
    if complete:
        s,c,v,r=[summaries[n]['metrics']['top1'] for n in ORDER]
        p(f'SceneCNN changes top-1 accuracy by {c-s:+.2f} percentage points relative to SimpleNet and {c-r:+.2f} points relative to pretrained ResNet-18. ViT changes top-1 by {v-c:+.2f} points relative to SceneCNN. SceneCNN uses {100*summaries["cnn"]["parameters"]/summaries["resnet18"]["parameters"]:.1f}% as many parameters as the ResNet-18 comparison. These single-seed comparisons describe the measured configurations; they do not isolate architecture from pretraining, optimization, or device effects.')
    story.append(PageBreak())

    heading('CNN design and training scheme')
    table([
        ['Stage','Structure','Output for 128 x 128 input'],
        ['Stem','3 x 3 convolution, stride 2; batch norm; ReLU','32 x 64 x 64'],
        ['Residual stage 1','Two blocks, channels 32; first stride 2','32 x 32 x 32'],
        ['Residual stage 2','Two blocks, channels 64; first stride 2','64 x 16 x 16'],
        ['Residual stage 3','Two blocks, channels 128; first stride 2','128 x 8 x 8'],
        ['Residual stage 4','Two blocks, channels 256; first stride 2','256 x 4 x 4'],
        ['Classifier','Global mean pooling, dropout 0.2, linear 256 to 100','100 logits']
    ],[105,285,118])
    p('Each residual block has two 3 x 3 convolutions with batch normalization, a ReLU between them, and a ReLU after adding the shortcut. A 1 x 1 projection plus batch normalization aligns dimensions when stride or channels change. The last normalization scale in each residual branch starts at zero. Skip connections give gradients a shorter path; normalization stabilizes activation scales; narrow early stages and downsampling control computation. Global pooling and dropout limit the final classifier size and provide regularization.')
    if 'cnn' in compute and 'simple' in compute:
        p(f"SceneCNN has {compute['cnn']['parameters']/compute['simple']['parameters']:.2f} times the parameters of SimpleNet, while using {100*(1-compute['cnn']['macs']/compute['simple']['macs']):.1f}% fewer estimated forward MACs. Its larger channel capacity operates mostly on downsampled feature maps. This trades parameter storage for capacity while keeping the operation count modest; measured runtime also depends on the hardware and kernels.")
    p('The unchanged SimpleNet comparison uses three convolutional groups with bottleneck layers, ReLUs, max pooling, and global average pooling, without normalization or residual connections. The transfer-learning comparison replaces the ImageNet ResNet-18 head with a new 100-class linear layer and fine-tunes all parameters.')
    rows=[['Model','Optimizer and regularization','Schedule','Batch / device']]
    for name in ['simple','cnn','resnet18']:
        cfg=setting(name)
        if cfg:
            opt=cfg['opt']; total=opt['epochs']+opt['warmup_epochs']
            rows.append([NAMES[name],f"SGD; lr {opt['learning_rate']}; momentum {opt['momentum']}; weight decay {opt['weight_decay']}",
                         f"{opt['warmup_epochs']} warmup + {opt['epochs']} cosine epochs",f"{cfg['loader']['batch_size']} / {cfg['devices'][0]}"])
    table(rows,[125,173,120,90])
    p('The classifiers use cross-entropy loss. The learning-rate scheduler steps after each optimizer update. Training and validation metrics aggregate every mini-batch, including partial final batches; they do not depend on terminal print frequency. TensorBoard events and complete epoch histories are retained.')
    story.append(PageBreak())

    heading('Attention and ViT design')
    p('The attention module maps channels jointly to Q, K, and V, splits into heads, computes softmax(QK<super>T</super>/sqrt(d)) across keys, weights V, concatenates heads, and applies an output projection. The channel count must divide evenly across heads. Numerical tests compare both output values and input gradients with PyTorch scaled-dot-product attention for one, two, and four heads.')
    table([
        ['Component','Chosen design'],
        ['Patch embedding','8 x 8 non-overlapping patches; 16 x 16 = 256 tokens; 192 channels'],
        ['Position and depth','Learned absolute position embeddings; six blocks; four heads'],
        ['Attention pattern','4 x 4 local token windows in blocks 0, 1, 3, 4; global attention in blocks 2, 5'],
        ['Feed-forward layers','MLP expansion ratio 4, GELU; Pre-LN residual structure'],
        ['Regularization / output','Stochastic-depth rates linearly increase from 0 to 0.1; final normalization, mean token pooling, 100-class linear head']
    ],[135,373])
    p('A local block attends to 16 tokens per window instead of all 256 tokens. Its attention matrix therefore has 16 times fewer token pairs than a global block at this input size; projection and MLP costs remain. Periodic global blocks connect otherwise separate windows. Learned positions are interpolated if the input grid changes.')
    if 'vit' in compute and 'cnn' in compute:
        p(f"Despite similar parameter counts, this ViT uses about {compute['vit']['macs']/compute['cnn']['macs']:.2f} times the estimated forward MACs of SceneCNN. The token projections and MLPs dominate its operation count. Local attention reduces the attention-matrix cost but does not remove those other costs.")
    p('SimpleViT consumes the shared TransformerBlock interface (B, H, W, C). The current pre-team scaffold implements Pre-LN attention and MLP residual branches, padded window partition/unpartition, and per-sample stochastic depth. Independent review and the one-point Transformer-block writeup belong to the teammate under the work plan. Existing passing tests make the interface usable for the classifier experiments; they do not establish that the teammate has already performed their review.')
    cfg=setting('vit')
    if cfg:
        opt=cfg['opt']
        p(f"ViT training uses AdamW, learning rate {opt['learning_rate']}, weight decay {opt['weight_decay']}, batch size {cfg['loader']['batch_size']}, and {opt['warmup_epochs']} warmup plus {opt['epochs']} cosine-decay epochs on {cfg['devices'][0]}. CUDA training uses mixed FP16; evaluation uses FP32. It uses no ImageNet initialization.")
    else:
        p('ViT training is pending. The configured schedule is AdamW lr 0.001, weight decay 0.05, batch 256, and 1 warmup + 40 cosine epochs.')
    p('The convolutional model encodes local spatial structure through kernels and progressively downsampled feature maps. The ViT learns interactions between patch tokens and can use global context. Differences in the measured scores can reflect these choices as well as training settings; a single run is insufficient to establish a general advantage for either architecture.')
    story.append(PageBreak())

    heading('Learning curves')
    figure('classification_curves.png','Figure 1. Full-epoch training top-1 accuracy (dashed) and validation top-1 accuracy (solid); epochs include warmup.')
    figure('classification_loss_top5.png','Figure 2. Training cross-entropy and validation top-5 accuracy. Strong augmentation and training-only dropout mean training accuracy can be lower than validation accuracy.')
    story.append(PageBreak())

    heading('Learning dynamics and efficiency')
    rows=[['Model','Final train loss','Best val epoch / top-1','Train minutes','Eval seconds']]
    for name in ORDER:
        h=history(name); item=summaries.get(name)
        if h and item:
            best=max(h,key=lambda row:row['val']['top1'])
            rows.append([NAMES[name],f"{h[-1]['train']['loss']:.3f}",f"{best['epoch']} / {best['val']['top1']:.2f}%",
                         f"{sum(row['train']['seconds'] for row in h)/60:.1f}",f"{item['metrics']['seconds']:.1f}"])
    table(rows,[145,75,120,85,83])
    if complete:
        for name in ORDER:
            h=history(name); gap=h[-1]['train']['top1']-h[-1]['val']['top1']
            p(f"<b>{NAMES[name]}:</b> validation top-1 changes from {h[0]['val']['top1']:.2f}% after warmup to {h[-1]['val']['top1']:.2f}% at the final epoch. The final training-minus-validation accuracy gap is {gap:+.2f} points. The table's best epoch is descriptive; its checkpoint does not replace the final model in the headline results.")
    p('Train minutes sum measured training loops and exclude validation and setup. Evaluation seconds cover the full validation loader and model inference. SimpleNet, SceneCNN, and ResNet-18 run on an Apple M3 Max with 30 GPU cores and 36 GB unified memory (FP32); ViT runs on a Colab Tesla T4 (mixed FP16 training). Timing is not a controlled cross-device speed benchmark. Pretrained ResNet-18 also benefits from an external supervised dataset and a shorter fine-tuning schedule.')
    heading('Verification and reproducibility')
    p('The classifier audit checks all four final epoch numbers, complete histories, full validation counts, agreement between final-epoch and checkpoint evaluation, finite weights/metrics, TensorBoard files, and strict checkpoint loading against the saved model configuration. Reloaded models also produce finite 100-class outputs. Evaluation requires no initialization-weight download. SHA-256 digests identify the exact final checkpoints in results/classification_audit.json.')
    if complete:
        max_delta=max(abs(item['independent_evaluation']['metrics'][key]-item['metrics'][key])
                      for item in summaries.values() for key in ('top1','top5'))
        p(f'All four saved checkpoints were independently re-evaluated on the full 10,000-image validation split on MPS. The largest absolute top-1 or top-5 difference from the primary evaluation was {max_delta:.2f} percentage points. Each verification.json records the checkpoint digest, device, package version, and measured scores.')
    for label,filename in [('Local','local_tests.txt'),('Cloud','cloud_tests.txt')]:
        path=ROOT/'results'/filename
        if path.exists():
            lines=path.read_text().strip().splitlines()
            summary=next((line for line in reversed(lines) if 'passed' in line or 'failed' in line),lines[-1])
            p(f'{label} suite: '+escape(summary)+'.','SmallCaption')
    p('Seeds and saved configurations support repetition of the procedure, but exact bitwise equality across devices, package versions, or resumed data loaders is not promised. The numerical/integration suite covers attention, window operations, stochastic depth, classifier forward/backward passes, optimizer coverage, and the shared detector scaffold. Tests establish implementation checks; measured validation establishes the reported accuracy.')
    story.append(PageBreak())

    heading('Artifacts, integration, and sources')
    p('Each results/simple, results/cnn, results/vit, and results/resnet18 folder contains the final model checkpoint, resolved YAML configuration, training and evaluation logs, JSON epoch history, TensorBoard events, and metrics. Intermediate checkpoints and datasets are excluded from the contribution archive. The repository stores code and small artifacts; release assets hold the large trained weights.')
    p('Run from the assignment root: <font name="Courier" size="8">python code/verify_classification.py</font> to audit; <font name="Courier" size="8">python code/make_classification_report.py</font> to regenerate this PDF. Training uses <font name="Courier" size="8">python code/run_experiments.py --device cuda:0 --experiments simple cnn vit resnet18</font>. To reproduce the actual local device/batch choices, use the saved per-run configurations with train_cls.py from code/.')
    p('Required packages are listed in code/requirements.txt, with installed versions under results/. Additional tools are pytest for numerical checks, matplotlib for plots, reportlab for PDF generation, and pypdf for report inspection. The original assignment release supplies the starter code; existing third-party notices remain intact.')
    p('<b>Contribution provenance:</b> code, experiment orchestration, checking, and this report text were prepared with AI assistance for Matej. Matej should review and understand the material before adopting it in the team submission. The prospective six-point split is in TEAM_PLAN.md; the final team statement must describe actual work confirmed by both members. No teammate contribution or completed FCOS experiment is claimed in this component report.')
    refs=[
        'BMI / CS 771 Fall 2026 Assignment 2 release: rubric, starter code, and MiniPlaces protocol.',
        'MiniPlaces: https://github.com/CSAILVision/miniplaces',
        'Vaswani et al., Attention Is All You Need, NeurIPS 2017: https://arxiv.org/abs/1706.03762',
        'He et al., Deep Residual Learning for Image Recognition, CVPR 2016: https://arxiv.org/abs/1512.03385',
        'Dosovitskiy et al., An Image is Worth 16x16 Words, ICLR 2021: https://arxiv.org/abs/2010.11929',
        'Project repository: https://github.com/matejpopovski/bmi771-assignment2']
    for i,ref in enumerate(refs,1): p(f'[{i}] '+escape(ref),'SmallCaption')
    def footer(canvas,doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#d8e2e7')); canvas.line(52,42,560,42)
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#52636a'))
        canvas.drawString(52,29,'Classification contribution | '+('DRAFT' if not complete else 'For team integration'))
        canvas.drawRightString(560,29,str(doc.page)); canvas.restoreState()
    SimpleDocTemplate(str(output),pagesize=letter,leftMargin=52,rightMargin=52,topMargin=45,bottomMargin=54,
                      title='Assignment 2 - Classification contribution',author='Prepared for Matej Popovski with AI assistance').build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)


if __name__ == '__main__':
    main()
