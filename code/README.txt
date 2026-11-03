BMI / CS 771 Assignment 2

Run commands from the assignment root unless stated otherwise.

Install: python -m pip install -r code/requirements.txt
Download required datasets: python code/download_datasets.py
Tests: python -m pytest code/tests -q
Required training/evaluation: python code/run_experiments.py --device cuda:0 --reference
Local Mac comparison runs: python code/run_experiments.py --device mps --experiments simple resnet18
Bonus COCO preparation (run from code/): python prepare_coco.py
Bonus training: python code/run_experiments.py --device cuda:0 --experiments swin coco
Report: python code/make_report.py
Audit/package (including both bonuses): python code/package_submission.py --bonus

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

The report generator reads measured metrics and logs; it does not synthesize
results. It writes report_draft.pdf until the required experiment metrics and
factual team metadata are available. Team metadata format (writeup/team.json):
{"members": [{"name": "actual name", "contribution": "actual contribution"}]}
For approved solo work, add "solo_approval" with the factual approval description.

Optional COCO experiments use 5,000 official train2017 and 1,000 official val2017
images, sampled with seed 771. Sparse COCO category IDs are remapped consistently
in both annotation JSON files. This is a feasibility subset, not a full COCO score.

Exact package versions and evaluation device are recorded under results/.
