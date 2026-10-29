# Reducing track fragmentation in offline camera-only 3D MOT

Evaluation report. All numbers below come from `results/synthetic_results.md`, which
`python run_synthetic.py` regenerates. The nuScenes pipeline (`run_nuscenes.py`) is complete
and checked against the official evaluator (§5). **Real nuScenes and PAI–Foundation4D
numbers are still missing**, because they need the dataset and a frozen detection file (§7, §8).

## 1. Problem and success criterion

The task is offline 3D multi-object tracking from **frozen camera-only detections**. The main
goal is fewer fragmented trajectories without adding identity switches.

The pre-registered success criterion, measured on held-out val against a baseline tuned the
same way:

* **FRAG** at least 10% lower;
* **AMOTA** not lower;
* **IDS** not higher.

Primary metrics are AMOTA, FRAG and IDS from the nuScenes devkit. Secondary metrics are:

* HOTA and IDF1 from TrackEval;
* IDs/GT: distinct predicted ids per matched GT object (1.0 means unfragmented);
* tracking runtime per scene.

## 2. Baseline

The baseline mirrors the Foundation4D offline tracker (arXiv:2610.30340, Sec. 5.3):

* constant-velocity Kalman filter on BEV position, with detector velocities as measurements;
* Hungarian matching on center distance with class-specific gates;
* a longer track lifetime to relink short gaps;
* RTS smoothing.

It runs in the same code as ours (`trackfrag/tracker.py`, `baseline_config()`), so the two
differ only in the switches listed in §3. Two baseline versions are reported:

* paper-style defaults;
* tuned on train with the same budget and objective as ours.

The tuned baseline is the reference for the success check.

## 3. Method

The pipeline has three stages.

**Pass 1: online association with a short lifetime.** Its aim is to keep fragments pure, so
it avoids wrong merges.
* **Depth-aware Mahalanobis association.** Monocular and multi-view camera depth error is
  much larger than lateral error, grows with range and lies along the viewing ray. The
  measurement covariance is therefore anisotropic: σ_depth = 0.25 + 0.03·r and
  σ_lat = 0.10 + 0.006·r, rotated onto the ego-to-object ray. Association minimises the
  Gaussian negative log-likelihood under a χ² gate. A far-away car that "slides" in depth
  still matches; a near object a metre off sideways does not.
* **Two-stage, ByteTrack-style association.** High-score detections (≥ 0.3) match first.
  Low-score detections (0.1–0.3) can then extend tracks that already exist but cannot start
  new ones. This fills gaps caused by temporary score drops, typically from occlusion or
  truncation.

**Pass 2: global tracklet stitching** (replaces lifetime relinking).
* A single Hungarian assignment over all (tracklet end → tracklet start) pairs with
  0 < gap ≤ 3 s and the same class.
* Cost: a bidirectional Mahalanobis distance. The end state is predicted forward and the
  start state backward, each with covariance propagated through the gap. Velocity
  consistency, a size-ratio check and a linear gap penalty are added.
* Because the assignment is global, two fragments cannot claim the same continuation, which
  is the usual way greedy relinking creates identity switches.

**Output cleanup.**
* RTS smoothing over each merged track, interpolating the missing frames.
* Median box size, interpolated z and yaw, and a 180° heading-flip fix.
* Duplicate-track suppression: a weaker track is dropped if it shadows a stronger one for
  ≥ 60% of its life.
* Track-level rescoring and pruning: scores are blended with the track mean
  (`0.5·s + 0.5·mean`), and short or low-confidence tracks are dropped.

Future frames are used only through smoothing and stitching, which the offline setting
allows. No appearance features are used; that variant is deliberately left separate (§8).

## 4. Protocol

* **Data (synthetic proxy).** Real detections are not yet available, so development and the
  full protocol ran on camera-like synthetic scenes (`trackfrag/synthetic.py`). Each scene
  is 40 frames at 2 Hz with a moving ego vehicle and cars, trucks, pedestrians and bicycles.
  The simulated detector has:
  * range-dependent depth error along the ray, with AR(1) temporal correlation;
  * occlusion from the ego viewpoint, and burst misses;
  * score correlated with quality;
  * 6% heading flips;
  * car/truck confusion;
  * duplicates along the ray, clutter and short-lived ghosts.

  The point of this proxy is to stress exactly the failure modes behind camera-only
  fragmentation.
* **Splits.**
  * Train: 60 scenes, used only for tuning.
  * Val: 150 held-out scenes with disjoint seeds, the same size as nuScenes val, reported
    once.
  * Difficulty sweep: a further 50 scenes per level.
* **Tuning.**
  * Both trackers get a small grid on train and are selected by AMOTA.
  * Baseline grid (12 configs): detection threshold × lifetime × relink gap.
  * Ours (6 configs): detection threshold × stitching gap.
  * The track-score threshold for HOTA and IDF1 is also chosen on train.
* **Metrics.**
  * AMOTA, AMOTP, MOTA, IDS and FRAG come from the unmodified nuScenes devkit
    `TrackingEvaluation` (`tracking_nips_2019`: 40 recall thresholds, IDS and FRAG at the
    best-MOTA threshold, summed over classes).
  * HOTA, AssA and IDF1 come from TrackEval with 2 m BEV center-distance similarity,
    averaged over classes.
  * Our own metric code (`trackfrag/metrics.py`) is verified equal to TrackEval and drives
    the paired bootstrap over scenes (1000 resamples).
* **Runtime.** Tracking only, on one CPU core, in a dedicated single-process pass.

## 5. Checking the evaluation path

* **Sanity checks.** GT scored against itself gives AMOTA 1.0, IDS 0 and FRAG 0 on the
  devkit, and HOTA 100. GT with scrambled ids gives DetA 100 with lower AssA and HOTA, as
  expected.
* **nuScenes path end to end.** `tests/make_fake_nuscenes.py` writes a database in nuScenes
  format, with the real table schema and real val scene names, from the 150 synthetic val
  scenes. It also writes a detection file in nuScenes format. `run_nuscenes.py eval
  --official` then tracks, writes a submission and scores it twice: once with our evaluator
  and once with the unmodified `TrackingEval.main()`. The two agree exactly:

| Tracker (untuned defaults) | Our path AMOTA / IDS / FRAG | Official `TrackingEval` AMOTA / IDS / FRAG |
|---|---|---|
| Baseline | 0.5266 / 2649 / 1575 | 0.5266 / 2649 / 1575 |
| Ours | 0.5688 / 568 / 799 | 0.5688 / 568 / 799 |

The baseline row also equals the "paper defaults" row in §6.1, computed without nuScenes
files. So the conversion to and from the submission format (box sizes, quaternions,
timestamps, ids) loses nothing.

This check found one bug, now fixed: TrackEval averaged over all 7 nuScenes classes even when
a class had no GT, which pulled HOTA down. Such classes are now skipped, as the devkit does.

## 6. Results (synthetic, held-out val, 150 scenes)

### 6.1 Main comparison

| Tracker | AMOTA ↑ | FRAG ↓ | IDS ↓ | MOTA ↑ | HOTA ↑ | AssA ↑ | IDF1 ↑ | IDs/GT ↓ | ms/scene |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline, paper defaults | 0.527 | 1575 | 2649 | 0.474 | 41.5 | 45.2 | 59.1 | 1.959 | 39 |
| Baseline, tuned on train | 0.546 | 1229 | 2103 | 0.470 | 43.3 | 49.2 | 62.1 | 1.757 | 58 |
| **Ours**, tuned on train | **0.600** | **870** | **780** | **0.567** | **50.4** | **57.6** | **74.4** | **1.239** | 45 |

**Success criterion against the tuned baseline:**

* FRAG −29.2% ✅ (target ≥ 10%);
* AMOTA 0.546 → 0.600 ✅;
* IDS 2103 → 780 (−63%) ✅.

So the method reduces fragmentation and identity switches together, which shows that the
fewer fragments do not come from wrongly merging tracks. A typical GT object is covered by
1.24 predicted ids instead of 1.76.

Paired bootstrap over val scenes, ours − tuned baseline (95% CI):

* HOTA +7.06 [+6.59, +7.52];
* IDF1 +11.73 [+10.93, +12.51];
* IDs/GT −0.516 [−0.554, −0.474].

None of the intervals comes close to zero.

### 6.2 Cumulative ablation (components added to the tuned baseline)

| Step | AMOTA | FRAG | IDS | HOTA | IDF1 |
|---|---:|---:|---:|---:|---:|
| Baseline (tuned) | 0.546 | 1229 | 2103 | 43.3 | 62.1 |
| + depth-aware Mahalanobis association | 0.590 | 1059 | 1363 | 46.2 | 64.3 |
| + two-stage low-score association | 0.580 | 1128 | 1262 | 48.1 | 68.9 |
| + global bidirectional stitching | 0.593 | 856 | 793 | 48.8 | 71.3 |
| + dedup, pruning, rescoring, yaw fix (= ours) | 0.600 | 870 | 780 | 50.4 | 74.4 |

### 6.3 Leave-one-out ablation (components removed from ours)

| Removed | AMOTA | FRAG | IDS | HOTA | IDF1 |
|---|---:|---:|---:|---:|---:|
| nothing (ours) | 0.600 | 870 | 780 | 50.4 | 74.4 |
| depth-aware noise (isotropic) | 0.593 | 941 | 852 | 47.5 | 73.2 |
| Mahalanobis (planar distance) | 0.583 | 971 | 1096 | 49.2 | 72.4 |
| two-stage association | 0.606 | 924 | 839 | 48.3 | 72.2 |
| global stitching (baseline relink) | 0.593 | 1099 | 1209 | 50.2 | 73.0 |
| any second pass | 0.568 | 1259 | 1574 | 49.3 | 70.2 |
| RTS smoothing / interpolation | 0.523 | 2455 | 900 | 41.1 | 70.5 |
| duplicate suppression | 0.598 | 869 | 813 | 50.3 | 74.2 |
| pruning & rescoring | 0.595 | 849 | 753 | 48.9 | 71.5 |

What the ablations show:

* **Global stitching drives the main goal.** Removing it brings back 26% more FRAG and 55%
  more IDS. Removing the whole second pass is worse still (FRAG 1259, IDS 1574).
* **Depth-aware Mahalanobis association gives most of the AMOTA gain** (+0.044 when added
  first) and also cuts IDS by 35%. Isotropic noise or a planar distance each lose part of
  that.
* **Two-stage association is a trade-off.** It improves FRAG, IDS, HOTA (+2.1) and IDF1
  (+2.2) but costs a little AMOTA (0.606 without it, 0.600 with it). Low-score boxes added
  to tracks lower the track-mean score that AMOTA ranks by. It is kept because it serves the
  primary goal (FRAG and IDS) and still passes the AMOTA check. If AMOTA alone mattered,
  turning it off would be the right setting.
* **RTS smoothing and interpolation are essential**: without them, FRAG nearly triples.
  This matters for comparing against the paper too: their tracker already smooths, so
  the baseline here includes it as well.
* **The cleanup steps are small.** They mainly lift MOTA, HOTA and IDF1 by removing false
  tracks, and leave the identity metrics roughly neutral.

### 6.4 Robustness to detector quality

The detector's noise, misses and clutter are scaled together. Each level uses 50 fresh
scenes; the configs were tuned once at difficulty 1.0.

| Difficulty | AMOTA base → ours | FRAG base → ours | IDS base → ours | HOTA base → ours |
|---:|---:|---:|---:|---:|
| 0.5 | 0.727 → 0.724 | 337 → 255 | 418 → 150 | 64.0 → 69.3 |
| 1.0 | 0.531 → 0.586 | 399 → 304 | 676 → 267 | 43.4 → 50.5 |
| 1.5 | 0.265 → 0.399 | 541 → 385 | 920 → 372 | 29.6 → 37.1 |
| 2.0 | 0.129 → 0.231 | 596 → 381 | 917 → 345 | 21.4 → 27.4 |

* FRAG and IDS drop at every level.
* The AMOTA gain grows as detections get worse. At the easiest level AMOTA is tied (−0.003),
  which is the only cell where the AMOTA check would not strictly pass.
* The method helps most where camera-only detectors struggle: far range, occlusion and
  noisy depth.

### 6.5 Runtime

Ours runs at 45 ms per 40-frame scene on one CPU core (the tuned baseline takes 58 ms) in
pure numpy/scipy. Global stitching is one Hungarian solve per class per scene and is
negligible. Ours is not slower than the baseline because pruning leaves fewer boxes to smooth.

## 7. Plan for the real-data run: official camera-only detections (Mapillary / MonoDIS)

No StreamPETR or GeoUP results file is public; StreamPETR releases only checkpoints. The first
real run therefore uses the frozen camera-only detections nuScenes itself publishes for
train/val/test (Mapillary's MonoDIS). The protocol is adapted in four ways:

1. **No detector velocities.** All MonoDIS velocities are exactly 0 (checked on all 6,019 val
   samples). Treating them as measurements would be wrong, so *both* trackers run
   position-only: velocity measurements get no weight and new tracks start from a 10 m/s
   velocity prior. The baseline therefore no longer exactly mirrors Foundation4D, which uses
   detector velocities.

   On 30 synthetic val scenes with velocities zeroed:

   | | with det. velocities | zero vel. fed as measurements | position-only |
   |---|---|---|---|
   | Baseline AMOTA / FRAG / IDS | 0.503 / 325 / 538 | 0.258 / 797 / 1700 | 0.440 / 406 / 954 |
   | Ours AMOTA / FRAG / IDS | 0.558 / 211 / 193 | 0.355 / 536 / 1045 | 0.551 / 243 / 212 |

   Losing velocities hurts the baseline much more than ours. **This setting favours stitching**,
   and the gap it measures is larger than with a velocity-capable detector.
2. **Tuning on held-out val scenes.** MonoDIS was trained on nuScenes train, so its train
   detections are optimistic and thresholds tuned there would be too permissive on val.
   Both trackers are tuned on the first 25 val scenes (sorted by name) and reported on the
   other 125. The report must say this split was used.
3. **Framing.** MonoDIS is a weak, older detector, and absolute AMOTA will be low. The
   difficulty sweep (§6.4) shows gains shrink as detections improve (AMOTA tied at the
   easiest level). A MonoDIS win does **not** show the method helps on Foundation4D-class
   detectors; that still needs a StreamPETR run. The relative change, recall, MT and ML
   (now printed by `run_nuscenes.py`) matter more than the raw values here.
4. **A public baseline tracker (pending).** To answer the "straw-man baseline" objection,
   AB3DMOT or SimpleTrack will be run on the same detection file. The tuned baseline should
   score at least as well as they do.

## 8. Limitations and open items

1. **The results are synthetic.** The simulator models the known camera failure modes but is
   not a real detector. The real numbers come from:

   ```
   run_nuscenes.py tune --split train --scenes 100
   run_nuscenes.py eval --split val --official
   ```

   This needs nuScenes metadata (`v1.0-trainval_meta`; no sensor data needed) and **one**
   frozen detection file (StreamPETR or GeoUP) for val and a train slice. If no public
   results json exists, those detections must be made once with the released checkpoint,
   which needs a GPU (e.g. Kaggle or Colab).
2. **PAI–Foundation4D is not evaluated.** The second evaluation needs data access and an
   adapter for its format. The tracker only consumes `Frame` objects (boxes, velocities,
   scores, classes, ego position), so the adapter is a loader, not a change to the method.
3. **Tuning on train is optimistic.** On nuScenes, a detector trained on train produces
   better detections on train than on val, so thresholds tuned there can be too permissive
   for val. Both trackers share this bias, so the comparison stays fair. Absolute numbers
   may still drop.
4. **The noise model is hand-set.** σ_depth and σ_lat are fixed priors, not fitted. On real
   data, fit them to the train-slice residuals between detections and GT; the code takes
   them as config.
5. **Software environment.** The devkit pins packages older than Python 3.12. A small shim
   (`trackfrag/official.py`) restores removed aliases (`collections.Iterable`,
   `inspect.getargspec`, `np.float`) without changing any metric code. §5 shows our scores
   match the unmodified `TrackingEval.main()` exactly.
6. **No appearance features.** By design, an appearance-embedding variant (frozen public
   re-ID or detector features in the stitching cost) would be reported separately. It is
   not implemented yet. It would most likely help long-gap stitching, where kinematics alone
   is ambiguous.
7. **The two-stage AMOTA cost** (§6.3) is real. Whether to keep two-stage association
   should be decided again on real nuScenes train, where the tuning grid can include it.

## 9. Reproducing

See `README.md`. In short:

```bash
pip install -r requirements.txt
python run_synthetic.py                       # full protocol, ~65 min on 8 workers → results/
python tests/make_fake_nuscenes.py /tmp/fake  # optional: end-to-end check of the nuScenes path
python run_nuscenes.py tune --dataroot /data/nuscenes --dets dets_train.json --split train --scenes 100
python run_nuscenes.py eval --dataroot /data/nuscenes --dets dets_val.json --split val \
    --config results/nusc/selected_baseline.json --config results/nusc/selected_ours.json --official
```
