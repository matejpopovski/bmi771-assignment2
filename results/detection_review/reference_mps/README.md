# Additional local reference validation

The instructor-supplied checkpoint was independently evaluated on all 4,952 VOC2007 test images using the reviewed code. This is a local MPS correctness check, not a trained detector and not the required CUDA training run.

- AP50: 60.68622887780991% (matches the previously checked-in reference AP50 exactly).
- AP50:95: 32.79533955988392%.
- Synchronized model forward, including model transforms and decoding/NMS: 145.49 seconds.
- Complete data loop, including loading/transfers and 12 sample PNGs: 156.33 seconds.
- Complete evaluation, including serialization and COCOeval: 176.34 seconds.
- PyTorch: 2.15.0; device: MPS; batch size: 8. The cold first batch is included.

Sample observations at display threshold 0.5: image 000001 has overlapping person boxes and no displayed detection for the occluded dog; image 000002 has a train detection; image 000003 has no displayed detections despite furniture behind glass; image 000004 includes several cars. These examples show successes and limitations and do not substitute for full-test AP. Display omissions do not imply that no lower-confidence prediction exists.

The checkpoint is a copy of the Canvas instructor checkpoint (hash in ../review.json), remains ignored by Git, and is kept separate from the independently trained model. Predictions are also ignored by Git. CUDA measurements and final trained-detector results belong under results/reference_fcos and results/fcos.
