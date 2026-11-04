BMI / CS 771 Assignment 2

Run commands from the assignment root unless stated otherwise.

Install: python -m pip install -r code/requirements.txt
Download required datasets: python code/download_datasets.py
Tests: python -m pytest code/tests -q
Required training/evaluation: python code/run_experiments.py --device cuda:0 --reference
Local Mac comparison runs: python code/run_experiments.py --device mps --experiments simple resnet18
Bonus COCO preparation (run from code/): python prepare_coco.py
Bonus training: python code/run_experiments.py --device cuda:0 --experiments swin coco
Final report: python code/make_final_report.py
Audit/package (required assignment): python code/package_submission.py

All model implementations are in cls_libs/modeling.py and det_libs/modeling.py.
The baseline is selectable as model_type: SIMPLENET; CNN selects SceneCNN.
Evaluation defaults to a single center crop. Optional --flip enables flip averaging
but is excluded from the primary comparison. Never train on MiniPlaces val or VOC test.

Results are copied after each successful run. Intermediate epoch checkpoints are
retained only in logs/ to permit restart and are never included by the packager.
A completed run is skipped on restart. Incomplete runs resume from the latest
saved epoch checkpoint. To repeat a run from scratch, invoke train_cls.py or
train_det.py directly from code/ with a new --output suffix. Seeds are fixed,
but exact bitwise replay across hardware or resumed data loaders is not promised.

To evaluate a submitted classifier from code/:
python eval_cls.py ../results/cnn/config.yaml ../results/cnn/
For a detector: python eval_det.py ../results/fcos/config.yaml ../results/fcos/ -v
Change the YAML devices entry to the available device if needed. Evaluation
constructs the architecture without downloading initialization weights, then
loads the complete submitted checkpoint. Dataset files are still required.

The instructor checkpoint in results/reference_fcos is clearly marked as supplied
reference material. It is not the independently trained detector.

The final report generator reads measured metrics, logs and verification evidence.
It requires matching checkpoint hashes and completed prediction audits. The final
team contributions are in writeup/team.json. The older make_report.py and
component report generators are retained as development tools; use
make_final_report.py to regenerate the submitted team report.

Independent detector verification: python code/verify_detection.py --recompute-ap
--output results/matej_review/detector_audit.json (one line). This check requires
the original prediction JSON files from the private handoff, plus the original
data/test.json and pretrained/det_voc_res18.pth.tar. The large prediction files
are omitted from submission.zip; new inference regenerates them. Evaluate in a
separate result folder to preserve the submitted measurements, and keep the
saved evaluation batch size: 8 for the instructor and 32 for the trained FCOS.
Padding and GroupNorm can make detector predictions depend on batch composition.

Optional COCO experiments use 5,000 official train2017 and 1,000 official val2017
images, sampled with seed 771. Sparse COCO category IDs are remapped consistently
in both annotation JSON files. This is a feasibility subset, not a full COCO score.

Exact package versions and evaluation device are recorded under results/.
