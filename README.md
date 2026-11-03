# BMI / CS 771 — Assignment 2

PyTorch image classification and FCOS object detection team workspace.

See [TEAM_PLAN.md](TEAM_PLAN.md) for the **6-point / 6-point** work split,
interfaces, deliverables, review process, and optional balanced bonus plan.

## Ownership

- **Matej:** custom CNN (2), comparison with SimpleNet/ResNet-18 (1), attention
  (1), and ViT design/training (2).
- **Teammate:** local-attention Transformer block (1), FCOS inference (2), and
  FCOS loss/training (3). Identity/access will be added later.

This repository includes starter code and AI-assisted work prepared before team
formation. Planned ownership is not a completed contribution statement.

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

The six-point classification component is prepared for Matej with AI assistance.
The teammate's Transformer-block review and FCOS experiments, the actual team
contribution statement, and the combined report remain to be completed and
cross-reviewed. **The classification ZIP is not the finished Canvas submission.**

## Team submission

The final Canvas archive must contain only `code/`, `writeup/`, and `results/`.
Every team member submits the same finished archive. `code/package_submission.py`
checks the full team artifacts before producing `submission.zip`.
Datasets, intermediate checkpoints, and local environments are excluded from Git.
The original assignment release remains the source for course materials and the
instructor's reference checkpoint.

No new license is asserted over instructor-provided or third-party source;
existing attribution and copyright notices are preserved.
