## Qualitative analysis

The four displayed images are a fixed selection, rather than examples chosen for
success. Image 000001 illustrates occlusion and threshold effects: the trained
model displays one person box (score 0.7033), while the reference displays two
overlapping person boxes (0.6878 and 0.5277). Both produce dog candidates below
the display threshold: the highest dog score is 0.4523 for the trained detector
and 0.4997 for the reference. Thus, an absent displayed dog box does not mean
the prediction is absent from AP evaluation.

In image 000002, the reference displays a train at score 0.5425. The trained
detector's highest train score is 0.4893, narrowly below the 0.5 display cutoff.
It is still included among predictions scored by the AP evaluator at threshold
0.1. Image 000003 shows furniture behind glass, and neither model displays a
box above 0.5; occlusion, reflections and small objects remain challenging.
In image 000004, both models localize multiple cars. The trained detector
displays six car boxes and the reference seven, including overlapping hypotheses
among the distant vehicles. A relatively permissive NMS IoU of 0.6 can retain
boxes whose overlap does not exceed that threshold.

These examples show that similar aggregate AP50 can coexist with different
confidence calibration and individual detections. They cannot establish a
general improvement or degradation by themselves. The full-test result is
60.52% AP50 for the trained detector versus 60.69% for the instructor reference,
a difference of -0.17 percentage points in this single-seed experiment.
