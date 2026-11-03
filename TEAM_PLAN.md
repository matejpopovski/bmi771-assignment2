# Assignment 2: equal work plan

This is a prospective ownership plan, not a claim that either person already
performed the other's work. The repository contains starter code and AI-assisted
work prepared before the team was formed. Each owner must understand, verify,
and explain their assigned deliverables. Record actual contributions only after
review; do not infer authorship from this plan.

## Required work: 6 points each

| Rubric item | Points | Owner | Deliverable |
| --- | ---: | --- | --- |
| Custom CNN design, training, evaluation (§3.2.2) | 2 | Matej | SceneCNN implementation, config, final weights, logs, curves, top-1/top-5 |
| Compare CNN with SimpleNet and pretrained ResNet-18 (§3.2.2) | 1 | Matej | All three measured results, learning curves, discussion |
| Multi-head self-attention (§3.3.1) | 1 | Matej | `Attention` implementation and value/gradient tests |
| ViT design, training, evaluation (§3.3.2) | 2 | Matej | `SimpleViT`, config, final weights, logs, curves, analysis |
| Pre-LN local-attention Transformer block and stochastic depth (§3.3.1) | 1 | Teammate | Review/complete `TransformerBlock`, padding/window tests, explanation |
| FCOS inference (§4.2) | 2 | Teammate | Decode/NMS implementation, reference AP50 and timing, sample detections |
| FCOS multitask loss, training, evaluation (§4.3) | 3 | Teammate | Assignment/loss implementation, independently trained final detector, config/logs/curves, AP50 and visualizations |
| **Total** | **12** | **6 + 6** | |

A plain classification/detection split would be 7/5. Moving the one-point
Transformer-block task to the detection owner makes this exactly 6/6 by the
rubric. Training and review time still vary; share troubleshooting if one side
hits an infrastructure problem. Matej maintains the classifier experiment
pipeline; the teammate maintains the detector experiment pipeline.

## Interfaces and branches

- `matej/classification`: `Attention`, `SceneCNN`, `SimpleViT`, classifier configs,
  classifier training/evaluation, and the classification report/results.
- `teammate/detection`: `TransformerBlock`, detection code/configs, detector
  experiments and report/results. Only change `TransformerBlock` in the shared
  `code/cls_libs/modeling.py` unless coordinated first.
- `main`: reviewed integration. Use pull requests; no force pushes.
- Preserve existing function signatures and tensor layouts. `Attention` and
  `TransformerBlock` take/return `(batch, height, width, channels)`.
- Keep the current tested Transformer block available as shared initial
  scaffolding so classifier experiments are not blocked. The teammate owns its
  independent review, changes if needed, tests, and writeup. Do not rewrite a
  working implementation merely to manufacture commit history.

## Order of work and acceptance checks

1. Both: clone the repository, read the Canvas readme locally, install
   `code/requirements.txt`, and run `python -m pytest code/tests -q`.
2. Matej: finish the four classifier runs; check all validation counts are 10,000;
   load each final checkpoint against its saved config; produce classification
   curves and report. CNN and ViT must start from scratch.
3. Teammate: independently review the Transformer block and the FCOS scaffolding;
   verify reference AP50 near 60.9%; train only on VOC2007 trainval; evaluate all
   4,952 test images; provide complete detector artifacts and analysis.
4. Cross-review: Matej reviews the detector's data split, losses, and report;
   teammate reviews classifier metrics, attention/block interfaces, and report.
   Both run the shared tests after integration.
5. Each writes their own report section and confirms their actual contribution.
   Matej assembles the PDF; teammate independently checks the final ZIP against
   the rubric. This divides final assembly and final audit equally.
6. Both approve the same `code/`, `writeup/`, `results/` archive and each uploads
   it to Canvas. Check the live Canvas deadline; no date is assumed here.

## Optional bonuses

The core commitment above is 6 points each. If both agree to pursue bonuses,
Matej owns the COCO dataset extension/feasibility experiment (2 bonus points),
and the teammate owns the Swin-backbone experiment (2 bonus points). Existing
bonus code is preliminary scaffolding, not a completed bonus claim. A bonus
requires its own measured results and writeup.

## Artifact and authorship rules

- Commit code, small configs, metrics, histories, plots, and reports to Git.
- Put final checkpoints in a GitHub release; download them into `results/` before
  creating the Canvas ZIP. Do not commit datasets or intermediate checkpoints.
- Keep the instructor reference checkpoint identified as instructor-supplied.
- Report real measurements and actual work. AI-assisted implementation already
  exists in this repository; both members should review it and follow the
  course's applicable assistance/disclosure rules.
- Teammate identity and GitHub access will be added by Matej later. The final
  contribution statement remains incomplete until both confirm their work.
