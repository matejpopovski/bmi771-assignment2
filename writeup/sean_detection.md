# Transformer block and FCOS: methods and measured results

Prepared with AI assistance for GitHub user seanchang1529 to review and explain. This section does not assert that personal review or team cross-review has finished.

**Status:** Incomplete: a verified cloud reference run and/or independently trained detector artifacts are missing.

## TransformerBlock review

The existing block was independently checked without changing its forward behavior or parameter shapes. For input X in (batch, height, width, channels) order, its residual equations are `Y = X + DropPath(Attention(LN1(X)))` and `Z = Y + DropPath(MLP(LN2(Y)))`. Normalization precedes both learned branches (Pre-LN). A positive window size restricts attention to nonoverlapping spatial windows; zero selects global attention. Spatial padding is added after LN1, then removed before the attention residual is added. The supplied padding convention uses zero tokens without an attention mask. This was preserved for checkpoint compatibility.

Stochastic depth samples one keep/drop decision per original image and residual branch, broadcasts it over all spatial positions and channels, and divides kept branches by the keep probability. It is applied after windows are reassembled, so separate windows from one image share a decision. The two calls draw independent masks. Evaluation disables stochastic depth. The added tests compare outputs and input gradients with an explicit window-loop oracle, verify zero cross-window gradients, and check image-level masks on padded, rectangular inputs. No saved ViT checkpoint has been modified.

## FCOS inference

PointGenerator stores (y, x), whereas boxes use (x, y). At each FPN level, class and centerness logits are passed through sigmoid and combined as sqrt(p_class * p_center). Thresholding and top-K selection operate on class-location pairs, allowing multiple class hypotheses at one location. Predicted (left, top, right, bottom) distances are multiplied by stride and decoded around the location. Boxes are clipped to resized image bounds; dimensions at most 0.01 pixel are rejected. Class-aware NMS is applied across levels, followed by the per-image limit. Labels receive +1 for the supplied COCO-format category IDs, and boxes are rescaled to original image dimensions.

## FCOS assignment and multitask loss

A location must lie strictly inside a ground-truth box and its clipped center-sampling region. The region extends center_sampling_radius times the level stride from the box center. The maximum pixel distance to the box sides must fall within that level's configured regression range. The smallest-area eligible ground-truth box wins overlaps. Class targets are zero-based; unmatched locations use an all-zero class vector. Regression targets are side distances divided by stride.

The loss sums sigmoid focal classification (alpha 0.25, gamma 2) over all locations, unweighted GIoU regression over positives, and BCE-with-logits centerness over positives. The centerness target is sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)). Each loss is normalized by the starter's EMA of positive counts (initial value 100, momentum 0.9, lower bound 1). Empty-positive batches retain classification loss and differentiable zero regression/centerness losses. Assignment is vectorized over locations and boxes, with only the image dimension looped.

## Data and training protocol

Only the supplied VOC2007 trainval annotations (5,011 images) may be used for training; all 4,952 test images are reserved for evaluation. The preflight audit verifies image counts, category IDs, annotation references, image existence and disjoint filenames. The instructor detector is evaluated separately and is never loaded by train_det.py. Training initializes the ResNet-18 backbone from ImageNet; FPN and detection heads start fresh. No torchvision or other external FCOS implementation is used.

The planned configuration is code/configs/det_voc_fcos.yaml. Final training settings will be described from results/fcos/config.yaml after the run.

## Results

| Detector | AP50 | AP50:95 | Images |
| --- | ---: | ---: | ---: |
| Instructor checkpoint (evaluation only) | 60.69% | 32.80% | 4952 |
| Independently trained FCOS | Not measured | Not measured | Not measured |

The reference row above is inherited from the teammate's earlier MPS run and has not yet been independently reproduced in this work.

Reference: data-loop time 115.70 seconds, 42.80 images/second; full evaluation including COCOeval 130.28 seconds.

![reference_fcos test detections](../results/figures/reference_fcos_detections.png)

Examples use the first test images in deterministic loader order, with a display threshold of 0.5. AP uses every image and the configured score threshold 0.1; NMS IoU is 0.6. Qualitative error analysis must be added after viewing the actual trained-model examples.

## Verification, provenance and remaining review

Existing code was preliminary and AI-assisted. This work independently reviewed the assigned paths, added mathematical/behavioral tests, fixed stale reference reuse and measurement provenance, verified evaluation coverage, bounded sample rendering, and made checkpoint writes atomic. TransformerBlock and Matej's classifier artifacts were preserved. Test and package-version records are under results/detection_review/. AI assistance covered code review, tests, workflow changes and this draft; the user must verify understanding and confirm actual contributions under the course policy.

Matej should review target assignment/loss conventions, the trainval/test audit, checkpoint provenance, the real detection results and the report section before integration. The combined PDF, team contribution confirmation and final ZIP audit remain separate team steps. No merge or Canvas submission is performed by this workflow.
