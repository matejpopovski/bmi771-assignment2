# BMI / CS 771 — Assignment 2

PyTorch image classification and FCOS object detection team workspace.

See [TEAM_PLAN.md](TEAM_PLAN.md) for the **6-point / 6-point** work split,
interfaces, deliverables, review process, and optional balanced bonus plan.

## Ownership

- **Matej:** custom CNN (2), comparison with SimpleNet/ResNet-18 (1), attention
  (1), and ViT design/training (2).
- **Sean Chang (`seanchang1529`):** local-attention Transformer block (1), FCOS
  inference (2), and FCOS loss/training (3).

The project builds on the assignment starter. AI assistance contributed to
implementation, experiment orchestration, tests, analysis, and report drafting.
Member contributions are recorded in [writeup/team.json](writeup/team.json) and the combined report.

## Setup and commands

Use Python 3.10+ and install `python -m pip install -r code/requirements.txt`.
Obtain the assignment release from Canvas for the original annotations and
instructor checkpoint, which are not republished here. Keep its `data/` and
`pretrained/` folders next to `code/`.

```sh
python code/download_datasets.py
python -m pytest code/tests -q
python code/run_experiments.py --device cuda:0 --experiments simple cnn vit resnet18
python code/run_experiments.py --device cuda:0 --experiments fcos --reference
```

[Open the classifier Colab notebook](https://colab.research.google.com/github/matejpopovski/bmi771-assignment2/blob/main/code/classification_colab.ipynb)
for a GPU setup, training, and results-download workflow. Select a T4 runtime.

Mac GPU: replace `cuda:0` with `mps`. The runner saves actual runtime settings in
`results/<experiment>/config.yaml`, resumes checkpoints, and collects final
artifacts. See [code/README.txt](code/README.txt) for evaluation and bonus commands.

## Verified classifier results

| Model | Top-1 | Top-5 |
| --- | ---: | ---: |
| SimpleNet baseline | 43.43% | 72.49% |
| SceneCNN | 48.74% | 77.60% |
| Local/global ViT | 45.12% | 74.61% |
| ImageNet ResNet-18 | 56.45% | 83.81% |

All four final checkpoints were independently re-evaluated on all 10,000
MiniPlaces validation images. Top-1 and top-5 exactly matched the primary
measurements. The local and Colab test suites each pass all 30 tests.

- [Classification report](writeup/matej_classification.pdf)
- [Artifact audit and checkpoint hashes](results/classification_audit.json)
- [Download the component ZIP and trained weights](https://github.com/matejpopovski/bmi771-assignment2/releases/tag/classification-v1)
- [Reproduction and verification instructions](code/CLASSIFICATION_README.txt)

To restore the exact weights after cloning, run
`python code/fetch_classification_weights.py`. To audit them, run
`python code/verify_classification.py`. The release component ZIP already contains
code, the classifier report, and all four classifier checkpoints/configs/logs.

The six-point classification component and its trained weights are available above.
**The classification ZIP is not the finished Canvas submission.**

## Verified detector component

| Detector | AP50 | AP50:95 | Test images | Eval batch |
| --- | ---: | ---: | ---: | ---: |
| Instructor checkpoint (evaluation only) | 60.69% | 32.80% | 4,952 | 8 |
| Independently trained FCOS | 60.52% | 32.42% | 4,952 | 32 |

Sean ran the reviewed workflow on a Colab Tesla T4. Training used only VOC2007
trainval: 3 warmup plus 12 cosine epochs, totaling 31.47 measured training-loop
minutes. The final detector's synchronized model time is 83.60 seconds over the
test set; the complete evaluation takes 158.02 seconds. Timing scopes and batch
differences are documented in the report. Both checkpoint/config pairs load
strictly; AP/AP50/AP75 are reproduced exactly from transferred predictions.
The instructor weights remain identified and separate from the trained detector.

- [Sean's component report](writeup/sean_detection.pdf) and [editable section](writeup/sean_detection.md)
- [Detector artifact audit and hashes](results/detection_review/detector_audit.json)
- [GPU workflow and reproduction](code/DETECTION_README.md)
- [Team handoff and remaining integration](writeup/TEAM_HANDOFF.md)

## Final verification and integration

The combined report is [writeup/report.pdf](writeup/report.pdf). The final suite
passes all 50 tests, including mathematical checks, complete evaluation coverage,
checkpoint/configuration loading, stale audit rejection, and archive selection.
The original component reports remain as historical artifacts; the final ZIP
includes only the combined report and team contribution metadata under writeup/.

Both handoff archives were received and hash-checked. All 144 component manifest
entries match. The trained FCOS checkpoint loads strictly, has 15 complete epochs,
and differs from the instructor checkpoint in 146 floating tensors. Independent
COCOeval recomputation from both original prediction dumps exactly matches
AP/AP50/AP75. Full inference checks and device/batch details are recorded in
[review evidence](results/matej_review/review.json) and the final report.

Preserve the saved detector evaluation batch sizes (reference 8; trained 32).
Padding and spatial GroupNorm can make predictions depend on batch composition.
Do not overwrite the primary measurements with checks made under other settings.

## Team submission

The final Canvas archive must contain only `code/`, `writeup/`, and `results/`.
Every team member submits the same finished archive. `code/package_submission.py`
checks the full team artifacts before producing `submission.zip`. Regenerate the
combined PDF with `python code/make_final_report.py`. The final ZIP remains local
because it contains the instructor-supplied reference checkpoint. Both members
should read the combined report, confirm their contributions, and submit the same ZIP.
Datasets, intermediate checkpoints, and local environments are excluded from Git.
The original assignment release remains the source for course materials and the
instructor's reference checkpoint.

No new license is asserted over instructor-provided or third-party source;
existing attribution and copyright notices are preserved.
