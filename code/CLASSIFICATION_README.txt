Matej's classification contribution — 6 required rubric points

Scope: custom CNN (2), comparison with SimpleNet and ImageNet ResNet-18 (1),
multi-head attention (1), and ViT design/training (2).

The teammate owns TransformerBlock review/implementation (1), FCOS inference (2),
and FCOS loss/training (3). Existing shared scaffolding is included so the full
numerical/integration suite can run. Its presence does not claim completion of
those teammate-owned experiments or their contribution statement.

Final component artifacts (created only after the classifier audit passes):
- writeup/matej_classification.pdf
- results/simple, results/cnn, results/vit, results/resnet18
- results/classification_audit.json and classification_files_sha256.json
- results/figures/classification_curves.png and classification_loss_top5.png

Each experiment folder includes its resolved config, final weights, training and
evaluation logs, epoch history, TensorBoard events, and measured metrics.
Intermediate checkpoints and datasets are intentionally excluded.

From the assignment root:
  python -m pip install -r code/requirements.txt
  python code/download_datasets.py --classification-only
  python -m pytest code/tests -q
  python code/verify_classification.py
  python code/make_classification_report.py
  python code/package_classification.py

If cloning Git rather than extracting the component ZIP, restore release weights:
  python code/fetch_classification_weights.py
It checks each checkpoint's SHA-256 against the audited record. The release must
be published and the repository updated to the completed contribution first.

To evaluate from code/ on a CUDA GPU, preserving the original measurements:
  python eval_cls.py ../results/cnn/config.yaml ../results/cnn/ --device cuda:0 --metrics-output ../results/cnn/recheck_cuda.json
Use --device mps for a Mac GPU or --device cpu for CPU evaluation.
Use simple, vit, or resnet18 in place of cnn for the other models.

For a separate full-data verification record used by the artifact audit:
  python code/recheck_classification.py --device mps
Run that command from the assignment root; CUDA and CPU are also supported.

The component ZIP is NOT the finished two-person Canvas submission. Combine the
verified detector results and both members' confirmed contribution statements,
then build one shared report and the full code/writeup/results ZIP. Both students
must submit that same completed archive themselves.
