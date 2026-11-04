# Detection component: seanchang1529

This workflow covers the required TransformerBlock review and FCOS inference,
loss, training and evaluation. It preserves Matej's classification artifacts.
No bonus experiment, merge or Canvas submission is requested by these commands.

## Start on Colab

Select a T4 GPU. Put the Canvas ZIP at
`MyDrive/BMI771_Assignment2/assignment2_release.zip`. Run this cell in the notebook:

```python
from google.colab import drive
from pathlib import Path
import os, subprocess, sys

drive.mount('/content/drive')
root = Path('/content/bmi771-detection')
if not root.exists():
    subprocess.run(['git', 'clone', '--branch', 'teammate/detection',
                    'https://github.com/matejpopovski/bmi771-assignment2.git', str(root)], check=True)
os.chdir(root)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'code/requirements.txt'], check=True)
subprocess.run([
    sys.executable, '-u', 'code/run_detection_colab.py',
    '--release', '/content/drive/MyDrive/BMI771_Assignment2/assignment2_release.zip',
    '--backup-dir', '/content/drive/MyDrive/BMI771_Assignment2/detection_artifacts',
], check=True)
```

It runs tests; extracts only course annotations/instructor weights; downloads only
VOC images; verifies trainval/test separation and image coverage; reproduces the
instructor detector; trains a fresh detector; evaluates every test image; and
writes a draft methods/results section and plots from actual artifacts.

Images stay under `/content` for speed. Epoch checkpoints, training logs and
TensorBoard events are written directly to the Drive backup's `logs/` via a
symlink. Checkpoint writes use temporary files and atomic renames. Reference
results are copied to Drive before training starts. A `finally` block copies
collected results even when a command fails. A hard runtime deletion can skip
`finally`, but already saved epoch checkpoints/logs remain on Drive.

If Colab replaces the runtime, run the same cell in a fresh session. It restores
collected results and resumes the latest valid epoch checkpoint in Drive.
Do not delete the Drive backup. Once the process finishes, verify that
`detection_artifacts/results/fcos/model_final.pth.tar`, metrics, config, logs,
history and samples exist before disconnecting. Back up/download the whole
`detection_artifacts` folder; its `logs/` supports recovery, whereas final Canvas
packaging excludes intermediate epoch checkpoints.

For live progress, inspect
`detection_artifacts/logs/det_voc_fcos_run_assignment2/training.log` in Drive.
The existing config uses batch size 32, FP16 on CUDA, and 3 warmup + 12 cosine
epochs. A T4 out-of-memory failure must be addressed with a new, documented
configuration before continuing, rather than silently changing batch size.

## Local review and direct commands

```sh
python -m pip install -r code/requirements.txt
python -m pytest code/tests -q
python code/download_datasets.py --detection-only
python code/verify_detection_data.py
python code/run_experiments.py --device cuda:0 --experiments fcos --reference
python code/make_detection_report.py
python code/verify_detection.py --recompute-ap
python code/render_detection_report.py
python code/package_detection.py
```

To evaluate only the instructor checkpoint, use
`python code/run_experiments.py --device cuda:0 --reference --experiments`.
`--reference-force` repeats evaluation even when provenance matches. Previous
metrics/configs/logs are preserved in `logs/reference_fcos_previous/` before a
fresh run. Imported metrics without `reference_run.json` are always reproduced.
The cache signature includes checkpoint/annotation/source hashes, config,
PyTorch/torchvision versions and GPU name.

Evaluation reports three explicit scopes: synchronized `model_seconds`
(forward including model transforms and decode/NMS), `inference_seconds`
(the data loop, including loading/transfers and up to 12 saved visualizations),
and `total_seconds` (including prediction serialization and COCOeval). The cold
first batch is included. Samples use the first test images in deterministic
loader order, at score >= 0.5; AP uses all test images at score threshold 0.1.

The tests review both values and gradients, including empty targets/predictions,
cross-level/class-aware NMS, nonzero GIoU gradients, padded local attention,
Pre-LN equations and per-image stochastic depth after window reassembly.
TransformerBlock's source and parameter shapes are unchanged. Coordinate any
future change with the user and re-evaluate Matej's saved ViT checkpoint.

## Report and cross-review

The completed T4 run and imported artifacts are audited in
`results/detection_review/detector_audit.json`. The trained detector scores
60.5191% AP50 and the instructor reference 60.6868% on 4,952 test images.
The local final suite has 44 passing tests; the cloud model/workflow suite has 43.
Run `verify_detection.py --recompute-ap` while the original prediction JSON files
are present to repeat the independent COCO AP check. `package_detection.py`
creates a private component ZIP with final weights and a verified SHA-256
manifest; it is not the combined Canvas submission.

`writeup/sean_detection.md` is generated from measured results and remains
explicitly incomplete until the required artifacts exist. Read and explain the
code, inspect detection examples, add qualitative analysis, and confirm actual
contributions. AI assistance must be disclosed according to the course policy.
Matej should review losses/assignment, data separation, checkpoint provenance,
performance and timing, then integrate this section into the combined PDF.

Commit code and small results only. Large weights, datasets, predictions and
intermediate checkpoints are ignored by Git. Final weights must be shared
separately and included under `results/` in the final Canvas ZIP.
