"""Create the PDF report exclusively from measured experiment artifacts."""
from pathlib import Path
import json
import textwrap
import xml.sax.saxutils as xml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
WRITEUP = ROOT / 'writeup'
WRITEUP.mkdir(exist_ok=True)
NAMES = {'simple':'SimpleNet baseline', 'cnn':'SceneCNN (ours)', 'vit':'Local/global ViT (ours)',
         'resnet18':'ImageNet ResNet-18', 'reference_fcos':'Reference FCOS',
         'fcos':'FCOS ResNet-18 (ours)', 'swin':'FCOS Swin-T (bonus)', 'coco':'COCO subset (bonus)'}
REQUIRED = ['simple','cnn','vit','resnet18','reference_fcos','fcos']


def load_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def history(name):
    path = RESULTS / name / 'history.jsonl'
    if not path.exists():
        candidates = sorted((ROOT / 'logs').glob('*_run_assignment2/history.jsonl'))
        keys={'simple':'cls_miniplaces_simple', 'cnn':'cls_miniplaces_cnn', 'vit':'cls_miniplaces_vit',
              'resnet18':'cls_miniplaces_resnet18','fcos':'det_voc_fcos','swin':'det_voc_swin','coco':'det_coco_fcos'}
        path=next((p for p in candidates if p.parent.name.startswith(keys.get(name,'__none__'))),path)
    if not path.exists():
        return []
    rows={}
    for line in path.read_text().splitlines():
        try:
            row=json.loads(line)
            rows[row['epoch']]=row
        except json.JSONDecodeError:
            continue
    return [rows[e] for e in sorted(rows)]


def make_plots():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.grid':True,'grid.alpha':0.18})
    palette=['#536c85','#176b62','#94603c','#76578b']
    available=[name for name in ['simple','cnn','vit','resnet18'] if history(name)]
    if available:
        fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        for ax,name,color in zip(axs.flat,['simple','cnn','vit','resnet18'],palette):
            rows=history(name)
            if not rows:
                ax.text(.5,.5,'Run not completed',ha='center',transform=ax.transAxes)
            else:
                epochs=[r['epoch'] for r in rows]
                ax.plot(epochs,[r['train']['top1'] for r in rows],color=color,ls='--',label='Train (augmented)')
                ax.plot(epochs,[r['val']['top1'] for r in rows],color=color,lw=2,label='Validation')
                ax.legend(fontsize=8,loc='lower right')
            ax.set(title=NAMES[name],xlabel='Epoch (includes warmup)',ylabel='Top-1 accuracy (%)',ylim=(0,100))
        fig.savefig(FIGURES/'classification_curves.png',dpi=180)
        plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
        for name,color in zip(['simple','cnn','vit','resnet18'],palette):
            rows=history(name)
            if not rows: continue
            epochs=[r['epoch'] for r in rows]
            axs[0].plot(epochs,[r['train']['loss'] for r in rows],label=NAMES[name],color=color)
            axs[1].plot(epochs,[r['val']['top5'] for r in rows],label=NAMES[name],color=color)
        axs[0].set(xlabel='Epoch',ylabel='Training cross-entropy')
        axs[1].set(xlabel='Epoch',ylabel='Validation top-5 (%)',ylim=(0,100))
        axs[0].legend(fontsize=8)
        fig.savefig(FIGURES/'classification_loss_top5.png',dpi=180)
        plt.close(fig)
    for name in ['fcos','swin','coco']:
        rows=history(name)
        if rows:
            fig,ax=plt.subplots(figsize=(9,3.4),constrained_layout=True)
            for key,label in [('cls_loss','Focal classification'),('reg_loss','GIoU regression'),('ctr_loss','Centerness BCE'),('final_loss','Total')]:
                ax.plot([r['epoch'] for r in rows],[r['train'][key] for r in rows],label=label)
            ax.set(xlabel='Epoch (includes warmup)',ylabel='Mean normalized loss',title=NAMES[name])
            ax.legend(ncol=2,fontsize=9)
            fig.savefig(FIGURES/f'{name}_losses.png',dpi=180)
            plt.close(fig)
        paths=sorted((RESULTS/name/'viz').glob('*.png'))[:4]
        if paths:
            fig,axs=plt.subplots(2,2,figsize=(9,6.3),constrained_layout=True)
            for ax,path in zip(axs.flat,paths):
                ax.imshow(PILImage.open(path));ax.axis('off');ax.set_title(f'Image {path.stem}',fontsize=10)
            for ax in list(axs.flat)[len(paths):]: ax.axis('off')
            fig.savefig(FIGURES/f'{name}_detections.png',dpi=160);plt.close(fig)
    paths=sorted((RESULTS/'reference_fcos'/'viz').glob('*.png'))[:4]
    if paths:
        fig,axs=plt.subplots(2,2,figsize=(9,6.3),constrained_layout=True)
        for ax,path in zip(axs.flat,paths):
            ax.imshow(PILImage.open(path));ax.axis('off');ax.set_title(f'VOC test image {path.stem}',fontsize=10)
        fig.savefig(FIGURES/'reference_fcos_detections.png',dpi=160);plt.close(fig)


def main():
    make_plots()
    metrics={name:load_json(RESULTS/name/'metrics.json') for name in NAMES}
    team=load_json(WRITEUP/'team.json')
    missing=[NAMES[name] for name in REQUIRED if not metrics[name]]
    complete=not missing and bool(team and team.get('members'))
    filename=WRITEUP/('report.pdf' if complete else 'report_draft.pdf')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleMain',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#123b49'),spaceAfter=16))
    styles.add(ParagraphStyle(name='Deck',fontSize=11,leading=16,textColor=colors.HexColor('#51616b'),spaceAfter=15))
    styles.add(ParagraphStyle(name='Body',fontSize=10,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='CaptionSmall',fontSize=8.5,leading=11,textColor=colors.HexColor('#53616b'),spaceAfter=12))
    styles.add(ParagraphStyle(name='Cell',fontSize=8.5,leading=11))
    styles['Heading1'].textColor=colors.HexColor('#123b49')
    styles['Heading1'].fontSize=15
    styles['Heading2'].fontSize=11
    story=[]
    def p(text,style='Body'):
        story.append(Paragraph(text,styles[style]))
    def heading(text): p(text,'Heading1')
    def table(rows,widths):
        converted=[[Paragraph(str(c),styles['Cell']) for c in row] for row in rows]
        t=Table(converted,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef1')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#74909d')),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dae1e5'))]))
        story.extend([t,Spacer(1,12)])
    def fig(name,caption,width=508):
        path=FIGURES/name
        if path.exists():
            w,h=PILImage.open(path).size
            story.append(Image(str(path),width=width,height=width*h/w))
            p(caption,'CaptionSmall')
    p('Deep Neural Networks for Image Classification and Object Detection','TitleMain')
    p('BMI / CS 771 - Assignment 2 | Fall 2026','Deck')
    if not complete:
        p('<b>IN-PROGRESS REPORT - NOT A COMPLETE SUBMISSION</b>')
        if missing: p('Experiments still missing final measured results: '+', '.join(missing)+'.')
        if not team: p('The team roster and actual member contributions have not yet been supplied. No authorship or solo-work approval is assumed.')
    elif team:
        p('Team: '+', '.join(xml.escape(m['name']) for m in team['members'])+'.')
    heading('1. Experimental protocol')
    p('MiniPlaces uses 100,000 training images and 10,000 validation images across 100 classes. All classifier optimization uses the training split only. VOC 2007 uses 5,011 trainval images and 4,952 test images across 20 classes. Detection fitting uses trainval only. The test split is reserved for the final evaluation of each fixed configuration. No custom classifier uses ImageNet weights; only the designated ResNet-18 comparison is fine-tuned from ImageNet. Detection backbones use ImageNet initialization, as permitted for that part.')
    p('Every run uses seed 771. Configurations, TensorBoard events, epoch-level JSON histories, final checkpoints, and evaluation logs are preserved in results/. Classification reports single-center-crop top-1 and top-5 accuracy; horizontal-flip test augmentation is disabled for a consistent comparison. Detection is evaluated with the supplied COCO-format annotations and pycocotools. The requested mAP@0.5 is COCOeval AP50 (101-point interpolation), rather than the older VOC 11-point metric.')
    p('Data augmentation follows the starter pipeline: random resized crop to 128 x 128, horizontal flip, color jitter of 0.15, and rotation up to 15 degrees; inputs use ImageNet channel normalization. Validation resizes the shorter side to 144 and takes a 128 x 128 center crop. The saved per-run configuration is authoritative for all settings.')
    p('Hardware: an Apple M3 Max (30 GPU cores, 36 GB unified memory) is used for local validation and comparison experiments; Google Colab provides a Tesla T4 for cloud training. Training uses FP32 on MPS and mixed FP16 on CUDA; evaluation uses FP32 on both devices. Timing includes data loading and, for detection inference, postprocessing and saved visualizations; the COCO evaluation stage is timed separately.')
    story.append(PageBreak())
    heading('2. Classification architectures and optimization')
    table([
      ['Model','Architecture / initialization','Schedule'],
      ['SimpleNet','Instructor-provided bottleneck CNN; no normalization or residual connections; global average pooling; random initialization.','SGD, lr 0.1, momentum 0.9, weight decay 0.0001; 1 warmup + 40 cosine epochs.'],
      ['SceneCNN','3 x 3 stride-2 stem; 4 stages with channels 32, 64, 128, 256; two residual blocks per stage, first at stride 2. Batch normalization, zero-initialized final BN in each branch, global pooling and dropout 0.2. Random initialization.','SGD, lr 0.1, momentum 0.9, weight decay 0.0001; 1 warmup + 40 cosine epochs.'],
      ['Local/global ViT','8 x 8 patches; 16 x 16 tokens; 192-dimensional embeddings; 6 blocks; 4 heads; MLP ratio 4. Local 4 x 4 windows in blocks 0, 1, 3, 4 and global attention in blocks 2, 5. Learned positions and mean pooling; random initialization.','AdamW, lr 0.001, weight decay 0.05; 1 warmup + 40 cosine epochs.'],
      ['ResNet-18','ImageNet-pretrained backbone; new 100-class linear layer; all parameters fine-tuned.','SGD, lr 0.01, momentum 0.9, weight decay 0.0001; 1 warmup + 15 cosine epochs.']
    ],[78,258,172])
    p('Batch sizes are 128 for the local SimpleNet and ResNet-18 runs and 256 for the cloud SceneCNN and ViT runs. Thus, the comparison also reflects different update counts per epoch. SceneCNN uses skip connections to improve gradient propagation, while batch normalization stabilizes activations. The narrow early stages and progressive downsampling limit computation. Zero-initializing the final batch-normalization scale makes each residual branch start near zero; dropout regularizes the final representation. This design is implemented directly rather than loading a pretrained custom CNN.')
    p('Attention projects tokens jointly to queries, keys, and values, splits channels into heads, computes softmax(QK<super>T</super>/sqrt(d)) over keys, aggregates values, and applies an output projection. Every block is Pre-LN: x + DropPath(Attention(LN(x))), followed by x + DropPath(MLP(LN(x))). Stochastic-depth probabilities increase linearly from 0 to 0.1. The supplied partition/unpartition helpers pad incomplete windows and restore the original spatial shape. Global blocks allow information exchange between windows.')
    p('At 256 tokens, a local 4 x 4 window contains 16 tokens. The quadratic attention component therefore compares 16 times fewer token pairs in a local block than in a global block; projection and MLP costs remain. Learned positional embeddings are interpolated for a different input grid. The classifier normalizes tokens, averages them spatially, and predicts 100 logits.')
    story.append(PageBreak())
    heading('3. Classification results and comparison')
    params=load_json(RESULTS/'model_parameters.json',{})
    rows=[['Model','Params (M)','Val top-1','Val top-5','Epochs','Train min']]
    for name in ['simple','cnn','vit','resnet18']:
        m=metrics[name]; h=history(name)
        rows.append([NAMES[name],f"{params[name]/1e6:.2f}" if name in params else '-',f"{m['top1']:.2f}%" if m else 'Pending',f"{m['top5']:.2f}%" if m else 'Pending',
                     str(len(h)) if h else 'Pending',f"{sum(r['train']['seconds'] for r in h)/60:.1f}" if h else 'Pending'])
    table(rows,[160,68,70,70,55,85])
    fig('classification_curves.png','Figure 1. Full-epoch training accuracy (dashed) and validation accuracy (solid). Training uses augmentation and dropout, so its accuracy is not directly interchangeable with validation accuracy.')
    if all(metrics[n] for n in ['cnn','simple','resnet18','vit']):
        cnn=metrics['cnn']['top1'];simple=metrics['simple']['top1'];res=metrics['resnet18']['top1'];vit=metrics['vit']['top1']
        p(f'SceneCNN changes top-1 accuracy by {cnn-simple:+.2f} percentage points relative to SimpleNet and {cnn-res:+.2f} points relative to pretrained ResNet-18. The ViT differs from SceneCNN by {vit-cnn:+.2f} points. These are single-seed results, so small differences should not be interpreted as statistically established improvements. Pretraining provides extra supervised information and a different starting representation, making ResNet-18 a transfer-learning reference rather than a controlled architecture-only comparison.')
    else: p('A final numerical comparison will be generated only after all four measured evaluations are present. Partial curves do not establish final accuracy or convergence.')
    story.append(PageBreak())
    heading('4. Classifier learning dynamics')
    fig('classification_loss_top5.png','Figure 2. Training cross-entropy and validation top-5 accuracy across the fixed schedules.')
    if all(metrics[n] for n in ['cnn','simple','resnet18','vit']):
        for name in ['simple','cnn','vit','resnet18']:
            h=history(name)
            if h:
                best=max(h,key=lambda r:r['val']['top1'])
                gap=h[-1]['train']['top1']-h[-1]['val']['top1']
                p(f"<b>{NAMES[name]}.</b> Final recorded training loss is {h[-1]['train']['loss']:.3f}; the strongest validation epoch is {best['epoch']} at {best['val']['top1']:.2f}% top-1. The final train-minus-validation accuracy gap is {gap:+.2f} points. All reported checkpoint evaluations use the final scheduled checkpoint; the best validation value is descriptive and is not substituted for it.")
    p('Training metrics were corrected to aggregate every mini-batch, independently of terminal print frequency. Validation metrics include all 10,000 images. Checkpoints contain the state dictionary and epoch; resumable intermediate checkpoints additionally preserve optimizer and scheduler state. Intermediate checkpoints are excluded from the submission archive.')
    heading('5. FCOS implementation')
    p('The detector retains the supplied ResNet backbone, five-level FPN, and shared convolutional prediction heads. Target assignment is vectorized over every feature location and ground-truth box within each image. Points must lie inside both the box and its stride-scaled center-sampling region, and their largest side distance must satisfy that level\'s regression range. When several boxes qualify, the smallest-area box wins. Points with no match are background.')
    p('PointGenerator stores coordinates as (y, x), so both training and inference explicitly convert them to (x, y). Targets store left, top, right, bottom distances divided by stride. The classification target uses zero-based foreground labels and an all-zero vector for background. Losses are sigmoid focal loss over all points, GIoU over matched boxes, and centerness binary cross-entropy over positive points. Centerness is the square root of the product of the min/max horizontal and vertical distance ratios. All three losses use the starter\'s exponential moving average of positive counts (momentum 0.9); empty-positive batches produce differentiable zero regression and centerness losses.')
    p('Inference computes sqrt(sigmoid(class) * sigmoid(centerness)), thresholds each class-location pair, selects the top candidates per level, multiplies distances by stride, decodes and clips boxes, removes degenerate boxes, and applies class-aware NMS across levels. Overlapping boxes of different categories remain eligible. Output labels receive the required +1 offset, and the transform rescales boxes to each original image. No external FCOS detector implementation is used.')
    story.append(PageBreak())
    heading('6. VOC detection results')
    rows=[['Detector','AP50','AP50:95','Inference seconds','Images / second']]
    for name in ['reference_fcos','fcos','swin']:
        m=metrics[name]
        rows.append([NAMES[name],f"{100*m['AP50']:.2f}%" if m else 'Pending',f"{100*m['AP']:.2f}%" if m else 'Pending',
                     f"{m['inference_seconds']:.1f}" if m else 'Pending',f"{m['images_per_second']:.1f}" if m else 'Pending'])
    table(rows,[160,70,75,100,103])
    if metrics['reference_fcos']:
        m=metrics['reference_fcos']
        p(f'The provided checkpoint achieves {100*m["AP50"]:.2f}% AP50 on {m["num_images"]:,} test images, compared with the assignment\'s approximate 60.9% reference. End-to-end evaluation takes {m["total_seconds"]:.1f} seconds, including COCO evaluation. This is an inference correctness check; these weights are explicitly identified as instructor-supplied and are not claimed as a trained model from this work.')
    p('The main trained detector uses ResNet-18 and 128-channel FPN features, batch size 32, unfrozen backbone batch normalization, and short-side scale jitter in {288, 320, 352, 384, 416, 448, 480}, with maximum long side 640. SGD uses learning rate 0.02, momentum 0.9, and weight decay 0.0001, with 3 warmup plus 12 cosine-decay epochs. Center radius is 1.5 strides. Inference thresholds are score 0.1 and NMS IoU 0.6, with 1,000 candidates per level and at most 100 detections per image.')
    fig('fcos_losses.png','Figure 3. Full-epoch FCOS loss components on VOC trainval. The test set is not used to compute training losses or to update parameters.')
    if metrics['fcos']:
        h=history('fcos');m=metrics['fcos']
        p(f'Training comprises {len(h)} epochs and {sum(r["train"]["seconds"] for r in h)/60:.1f} measured training minutes. Final test AP50 is {100*m["AP50"]:.2f}%. The score difference from the fixed reference checkpoint is {100*(m["AP50"]-metrics["reference_fcos"]["AP50"]):+.2f} percentage points. Different random seeds, hardware, precision, and loss details can affect the result; the measured run is reported without replacing it with the reference score.')
    story.append(PageBreak())
    heading('7. Qualitative detection examples')
    fig('reference_fcos_detections.png','Figure 4. Instructor checkpoint decoded by the implemented inference routine. Images are selected deterministically by image ID from the saved visualization batches. Display threshold: 0.5; AP evaluation uses the configured 0.1 threshold.')
    p('The reference examples include a tightly localized dog and separate person/horse detections. They also expose limitations: two overlapping sofa boxes survive the 0.6-IoU NMS threshold, and the crowded horse scene produces several overlapping person predictions. These examples illustrate why plausible individual boxes do not by themselves establish overall precision or recall; the full test-set AP is the quantitative measure.')
    fig('fcos_detections.png','Figure 5. The independently trained FCOS model on held-out VOC test images, with the same deterministic selection and display threshold.',width=440)
    story.append(PageBreak())
    heading('8. Bonus extensions')
    p('<b>Swin Transformer backbone.</b> A Swin-T feature adapter returns the stage outputs at strides 8, 16, and 32. Its native NHWC tensors are permuted to contiguous NCHW tensors with 192, 384, and 768 channels before the FPN. The existing FCOS heads, assignment, losses, and inference are reused unchanged. The backbone is initialized with torchvision\'s ImageNet Swin-T classification weights; no external FCOS implementation is used. Relative-position bias tables are excluded from weight decay. The fixed feasibility schedule uses AdamW at 0.0001, weight decay 0.05, batch size 8, 1 warmup plus 6 cosine epochs, and the default 256/288/320 short-side jitter with maximum long side 480.')
    fig('swin_losses.png','Figure 6. Swin-T detector training losses, if the bonus run has completed.')
    if metrics['swin']:
        p(f'The Swin detector achieves {100*metrics["swin"]["AP50"]:.2f}% AP50 on the full VOC test set. Its schedule and input resolution differ from the main ResNet-18 run, so the comparison does not isolate the backbone alone.')
    else: p('<b>Swin experiment status:</b> implementation is present; training curves and measured evaluation are still pending. This is not a completed bonus claim.')
    p('<b>COCO subset feasibility.</b> The preparation script samples 5,000 images uniformly without replacement from official train2017 and 1,000 images independently from val2017 using seed 771. It downloads only those images and keeps their official bounding-box annotations. Sparse category IDs are remapped to contiguous IDs 1..80 in both annotation files; the mapping is saved in a manifest. Model labels are 0..79 internally, and inference restores 1..80 for this remapped evaluation dataset. Crowd annotations are excluded from training targets, while pycocotools retains their evaluation behavior. The model is ResNet-18 FCOS with 128-channel FPN features, batch size 16, SGD lr 0.01, and 1 warmup plus 4 cosine epochs.')
    if metrics['coco']:
        m=metrics['coco'];h=history('coco')
        p(f'On the held-out {m["num_images"]:,}-image COCO subset, AP50:95 is {100*m["AP"]:.2f}%, AP50 is {100*m["AP50"]:.2f}%, and inference throughput is {m["images_per_second"]:.1f} images/second. Training takes {sum(r["train"]["seconds"] for r in h)/60:.1f} minutes. These subset results demonstrate feasibility and are not full COCO benchmark scores.')
    else: p('<b>COCO experiment status:</b> implementation and preparation procedure are present; measured training/evaluation results are still pending. This is not a completed bonus claim.')
    fig('coco_losses.png','Figure 7. COCO subset training losses, if available.')
    story.append(PageBreak())
    heading('9. Verification and reproducibility')
    p('The automated suite compares multihead attention values and gradients against PyTorch scaled-dot-product attention; verifies padded-window round trips, residual identity, and local-window isolation; checks all classifier outputs, gradients, and optimizer parameter coverage; checks point ordering on a rectangular grid; verifies exact decoded boxes, stride multiplication, +1 labels, class-aware NMS, clipping, and empty predictions; checks analytic focal/GIoU/centerness losses, empty-target backward passes, smallest-box assignment, center sampling, and regression ranges; and exercises the full FCOS forward/backward path and Swin-to-FPN feature shapes.')
    for label,filename2 in [('Local','local_tests.txt'),('Cloud','cloud_tests.txt')]:
        path=RESULTS/filename2
        if path.exists():
            lines=path.read_text().strip().splitlines()
            p(f'{label} test log: '+xml.escape(next((line for line in reversed(lines) if 'passed' in line or 'failed' in line),lines[-1]))+'.')
    p('Additional tools beyond the assignment dependencies are pytest (numerical regression tests), matplotlib (plots), reportlab (PDF generation), and pypdf (PDF validation). code/requirements.txt lists the runtime packages; results/environment files record exact installed versions. Python paths in submitted YAML files are relative. The archive includes only code/, writeup/, and results/, and excludes datasets, caches, environments, and intermediate epoch checkpoints.')
    p('Seeds and complete configurations support repeating the experimental procedure. Exact bitwise reproduction across devices or package versions is not guaranteed; cuDNN benchmark selection is enabled, and a resumed run does not restore the original data-loader random-number state. The saved metrics describe the actual runs rather than a guarantee of identical future scores.')
    p('Reproduce the required runs from the assignment root with <font name="Courier" size="8">python code/run_experiments.py --device cuda:0 --reference</font>. Download inputs with <font name="Courier" size="8">python code/download_datasets.py</font>. Generate plots and this report with <font name="Courier" size="8">python code/make_report.py</font>. Package only after the submission audit succeeds.')
    heading('10. Contributions')
    if team:
        for member in team['members']:
            p('<b>'+xml.escape(member['name'])+'</b>: '+xml.escape(member['contribution']))
        if team.get('solo_approval'): p('Solo-work approval: '+xml.escape(team['solo_approval']))
    else: p('Team-member names, actual contributions, and any instructor approval for solo work have not been supplied. This section must be completed with factual information before submission.')
    heading('References')
    refs=[
      'Tian et al. FCOS: A Simple and Strong Anchor-Free Object Detector. TPAMI, 2020. https://arxiv.org/abs/2006.09214',
      'Vaswani et al. Attention Is All You Need. NeurIPS, 2017. https://arxiv.org/abs/1706.03762',
      'Liu et al. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. ICCV, 2021. https://arxiv.org/abs/2103.14030',
      'MiniPlaces dataset, CSAILVision. https://github.com/CSAILVision/miniplaces',
      'COCO dataset. https://cocodataset.org/',
      'PyTorch / torchvision documentation. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html',
      'BMI / CS 771 Fall 2026 Assignment 2 release: supplied readme, code, annotations, and reference checkpoint.'
    ]
    for i,ref in enumerate(refs,1): p(f'[{i}] '+xml.escape(ref),'CaptionSmall')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d8e2e7'));canvas.line(52,42,560,42)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#53616b'))
        canvas.drawString(52,29,'BMI / CS 771 | Assignment 2'+(' | DRAFT' if not complete else ''))
        canvas.drawRightString(560,29,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(filename),pagesize=letter,rightMargin=52,leftMargin=52,topMargin=45,bottomMargin=54,
                          title='Assignment 2: Image Classification and Object Detection',author='; '.join(m['name'] for m in team['members']) if team else 'Team details pending')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(filename)


if __name__ == '__main__':
    main()
