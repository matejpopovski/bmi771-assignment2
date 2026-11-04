# Sean Chang's detector component: handoff to Matej

The six required points have code review/tests, a completed cloud run and
audited artifacts. This is a technical component handoff; both students still
need to understand their work, confirm contributions and cross-review the team
submission. No merge or Canvas submission has been performed.

## Measured outcomes

| Item | Result |
| --- | --- |
| Instructor checkpoint, CUDA T4 | AP50 60.6868%; AP50:95 32.7955%; 4,952 images |
| Independently trained FCOS, CUDA T4 | AP50 60.5191%; AP50:95 32.4218%; 4,952 images |
| Training | VOC2007 trainval only; 15 total epochs; 31.47 training-loop minutes |
| Final checkpoint | Epoch 14 (zero based); 146 floating tensors differ from instructor |
| Saved events | 45 scalar records for each classification/regression/centerness stream |
| Data | 5,011 trainval + 4,952 test; all images present; zero filename overlap |

Both models use the same architecture. Reference evaluation batch size is 8;
trained-model evaluation batch size is 32. The timing results are not a
controlled architecture-speed comparison. Detailed synchronized model, data-loop
and complete evaluation timings are in the metrics and Sean's report.

## Files for integration

- `writeup/sean_detection.pdf`: formatted component report.
- `writeup/sean_detection.md` and `sean_detection_analysis.md`: editable methods,
  measured results, limitations and sample analysis.
- `results/fcos/`: trained final checkpoint/config, logs, history, metrics and 12
  saved test detections. Large weights/predictions remain ignored by Git.
- `results/reference_fcos/`: separate instructor checkpoint/provenance, fresh
  CUDA evaluation, metrics and the same 12 sample image IDs.
- `results/detection_review/detector_audit.json`: checkpoint hashes, strict loads,
  scheduled-epoch/event checks and independent AP recomputation from predictions.
- `results/figures/*fcos*.png`: loss curve and reference/trained example grids.
- `sean_detection_component.zip`: private component handoff including final
  weights; **not the complete Canvas submission**. Its manifest verifies ZIP
  contents. Prediction dumps and intermediate checkpoints are omitted.

The original completed-run archive and Drive backup retain the large prediction
JSON files used for independent AP verification. To repeat that AP calculation,
keep those JSON files available or regenerate predictions in a separate copy of
the experiment folder, preserving the original measurements.

## Work and AI assistance

Sean ran the Colab GPU workflow, supplied the Canvas release, and transferred the
completed artifacts. Codex assisted with review of pre-existing preliminary
model code, added tests, experiment/provenance fixes, artifact audits, plots and
the report section. Existing model scaffolding is not claimed as newly authored
human work. TransformerBlock behavior/parameter shapes are unchanged, and
Matej's classification code, saved metrics, logs and PDF are preserved. Classifier
weights were restored with the original audited SHA-256 checks and their artifact
audit passes; no new classifier validation inference is claimed in that audit.

Matej's component and its reported authorship remain as supplied. Each student
must confirm their actual contribution and course-policy disclosure before the
final roster is treated as approved.

## Remaining team steps

1. Sean reads the code/report and verifies understanding. In particular, explain
   Pre-LN ordering, why DropPath follows window reassembly, stride-normalized
   offsets, class-aware NMS, center sampling/smallest-box matching, and the three
   losses. The report describes each point and the tests that check it.
2. Matej reviews losses/assignment, data separation, checkpoint provenance,
   performance and timing. Sean reviews the supplied classifier component.
3. Matej integrates both report sections into `writeup/report.pdf`, updates any
   old statements that describe the detector as unfinished, and adds the
   confirmed contribution statement. Record agreed metadata in `writeup/team.json`.
4. After integration, run `python -m pytest code/tests -q` and
   `python code/package_submission.py`. Sean audits the produced ZIP, including
   final weights, report contents, relative paths, excluded datasets/intermediate
   checkpoints, and archive integrity. Keep the same verified ZIP for both people.
5. Confirm the live Canvas deadline. Merging and Canvas submission require the
   team's later instruction; each student submits the same completed ZIP.

No bonus results are claimed. The instructor checkpoint is course-supplied and
kept separate; only the team's independently trained weights belong in any
public trained-weight release.
