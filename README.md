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

Mac GPU: replace `cuda:0` with `mps`. The runner saves actual runtime settings in
`results/<experiment>/config.yaml`, resumes checkpoints, and collects final
artifacts. See [code/README.txt](code/README.txt) for evaluation and bonus commands.

## Results and submission

Measured small artifacts live under `results/`. Final trained weights will be
attached to a repository release when verified. Datasets, local environments,
intermediate checkpoints, and prediction dumps are excluded from Git.
The original assignment distribution remains the source for course materials.

The classifier report is Matej's deliverable. The combined team report remains
a draft until the teammate verifies their work and both members confirm actual
contributions. A successful test suite alone does not establish trained accuracy.

The final Canvas archive must contain only `code/`, `writeup/`, and `results/`.
Every team member submits the same finished archive. `code/package_submission.py`
automatically checks the required artifacts before producing `submission.zip`.

No new license is asserted over instructor-provided or third-party source;
existing attribution and copyright notices are preserved.
