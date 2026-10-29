# Synthetic benchmark results

Train 60 scenes (seeds 0..), tuning only. Held-out val 150 scenes (seeds 10000..). Sweep 50 scenes/level (seeds 20000..). 40 frames @ 2 Hz. AMOTA/FRAG/IDS/AMOTP/MOTA: nuScenes devkit (tracking_nips_2019; FRAG/IDS at the best-MOTA threshold, summed over classes). HOTA/AssA/IDF1: TrackEval, 2 m center distance, class average, track-score threshold tuned on train (baseline 0.3, ours 0.2). IDs/GT: distinct predicted ids per matched GT object. ms/scene: tracking only, one CPU core; main-table rows timed in a dedicated single-process pass, ablation rows inside the shared worker pool (indicative only).

## Success criterion (val, vs tuned baseline)

| Check | Pass | Value |
|---|:---:|---:|
| FRAG reduction >= 10% | ✅ | 29.2% |
| AMOTA not lower | ✅ | 0.546 -> 0.600 |
| IDS not higher | ✅ | 2103 -> 780 |

## Main comparison (held-out val)

| Tracker | AMOTA ↑ | FRAG ↓ | IDS ↓ | AMOTP ↓ | MOTA ↑ | HOTA ↑ | AssA ↑ | IDF1 ↑ | IDs/GT ↓ | ms/scene |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline, paper defaults (Foundation4D-style) | 0.527 | 1575 | 2649 | 1.044 | 0.474 | 41.5 | 45.2 | 59.1 | 1.959 | 39 |
| Baseline, tuned on train | 0.546 | 1229 | 2103 | 0.913 | 0.470 | 43.3 | 49.2 | 62.1 | 1.757 | 58 |
| **Ours**, tuned on train | 0.600 | 870 | 780 | 0.846 | 0.567 | 50.4 | 57.6 | 74.4 | 1.239 | 45 |

## Paired bootstrap over val scenes, ours − tuned baseline (1000 resamples)

| Metric | Δ mean | 95% CI |
|---|---:|---:|
| HOTA | +7.062 | [+6.593, +7.517] |
| IDF1 | +11.726 | [+10.934, +12.512] |
| IDs/GT | -0.516 | [-0.554, -0.474] |

## Cumulative ablation (val): add one component at a time to the tuned baseline

| Tracker | AMOTA ↑ | FRAG ↓ | IDS ↓ | AMOTP ↓ | MOTA ↑ | HOTA ↑ | AssA ↑ | IDF1 ↑ | IDs/GT ↓ | ms/scene |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (tuned) | 0.546 | 1229 | 2103 | 0.913 | 0.470 | 43.3 | 49.2 | 62.1 | 1.757 | 58 |
| + depth-aware Mahalanobis association | 0.590 | 1059 | 1363 | 0.833 | 0.516 | 46.2 | 55.3 | 64.3 | 1.507 | 201 |
| + two-stage low-score association | 0.580 | 1128 | 1262 | 0.911 | 0.531 | 48.1 | 55.4 | 68.9 | 1.449 | 122 |
| + global bidirectional stitching | 0.593 | 856 | 793 | 0.849 | 0.538 | 48.8 | 57.4 | 71.3 | 1.269 | 154 |
| + duplicate suppression, pruning, rescoring, yaw fix (= ours) | 0.600 | 870 | 780 | 0.846 | 0.567 | 50.4 | 57.6 | 74.4 | 1.239 | 45 |

## Leave-one-out ablation (val): remove one component from ours

| Tracker | AMOTA ↑ | FRAG ↓ | IDS ↓ | AMOTP ↓ | MOTA ↑ | HOTA ↑ | AssA ↑ | IDF1 ↑ | IDs/GT ↓ | ms/scene |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours (full) | 0.600 | 870 | 780 | 0.846 | 0.567 | 50.4 | 57.6 | 74.4 | 1.239 | 45 |
| − depth-aware noise (isotropic) | 0.593 | 941 | 852 | 0.894 | 0.552 | 47.5 | 54.5 | 73.2 | 1.272 | 164 |
| − Mahalanobis (planar distance) | 0.583 | 971 | 1096 | 0.876 | 0.549 | 49.2 | 55.7 | 72.4 | 1.334 | 183 |
| − two-stage association | 0.606 | 924 | 839 | 0.902 | 0.581 | 48.3 | 55.0 | 72.2 | 1.277 | 160 |
| − global stitching (baseline relink) | 0.593 | 1099 | 1209 | 0.891 | 0.566 | 50.2 | 56.4 | 73.0 | 1.357 | 134 |
| − any second pass | 0.568 | 1259 | 1574 | 0.951 | 0.548 | 49.3 | 54.2 | 70.2 | 1.473 | 96 |
| − RTS smoothing / interpolation | 0.523 | 2455 | 900 | 1.013 | 0.481 | 41.1 | 45.8 | 70.5 | 1.296 | 128 |
| − duplicate suppression | 0.598 | 869 | 813 | 0.847 | 0.563 | 50.3 | 57.5 | 74.2 | 1.254 | 157 |
| − pruning & rescoring | 0.595 | 849 | 753 | 0.849 | 0.542 | 48.9 | 57.5 | 71.5 | 1.253 | 178 |

## Robustness: detector difficulty sweep (noise, misses, clutter scaled)

| Difficulty | AMOTA base | AMOTA ours | FRAG base | FRAG ours | IDS base | IDS ours | HOTA base | HOTA ours |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.727 | 0.724 | 337 | 255 | 418 | 150 | 64.0 | 69.3 |
| 1.0 | 0.531 | 0.586 | 399 | 304 | 676 | 267 | 43.4 | 50.5 |
| 1.5 | 0.265 | 0.399 | 541 | 385 | 920 | 372 | 29.6 | 37.1 |
| 2.0 | 0.129 | 0.231 | 596 | 381 | 917 | 345 | 21.4 | 27.4 |

## Tuning on train (AMOTA)

```json
{
 "baseline": [
  [
   {
    "det_thr": 0.1,
    "max_age_s": 0.8,
    "relink_gap_s": 1.67
   },
   0.5465069588269313
  ],
  [
   {
    "det_thr": 0.1,
    "max_age_s": 0.8,
    "relink_gap_s": 3.0
   },
   0.5610727393065065
  ],
  [
   {
    "det_thr": 0.1,
    "max_age_s": 1.5,
    "relink_gap_s": 1.67
   },
   0.551633548300953
  ],
  [
   {
    "det_thr": 0.1,
    "max_age_s": 1.5,
    "relink_gap_s": 3.0
   },
   0.5564705279041409
  ],
  [
   {
    "det_thr": 0.2,
    "max_age_s": 0.8,
    "relink_gap_s": 1.67
   },
   0.5308490827818617
  ],
  [
   {
    "det_thr": 0.2,
    "max_age_s": 0.8,
    "relink_gap_s": 3.0
   },
   0.5610099669443686
  ],
  [
   {
    "det_thr": 0.2,
    "max_age_s": 1.5,
    "relink_gap_s": 1.67
   },
   0.5394015078877442
  ],
  [
   {
    "det_thr": 0.2,
    "max_age_s": 1.5,
    "relink_gap_s": 3.0
   },
   0.5571917289481266
  ],
  [
   {
    "det_thr": 0.3,
    "max_age_s": 0.8,
    "relink_gap_s": 1.67
   },
   0.5085690485371683
  ],
  [
   {
    "det_thr": 0.3,
    "max_age_s": 0.8,
    "relink_gap_s": 3.0
   },
   0.5463767438020014
  ],
  [
   {
    "det_thr": 0.3,
    "max_age_s": 1.5,
    "relink_gap_s": 1.67
   },
   0.5233957543968631
  ],
  [
   {
    "det_thr": 0.3,
    "max_age_s": 1.5,
    "relink_gap_s": 3.0
   },
   0.5435761519726976
  ]
 ],
 "ours": [
  [
   {
    "det_thr": 0.3,
    "relink_gap_s": 3.0
   },
   0.6134256994487853
  ],
  [
   {
    "det_thr": 0.3,
    "relink_gap_s": 5.0
   },
   0.5817652058692048
  ],
  [
   {
    "det_thr": 0.3,
    "relink_gap_s": 8.0
   },
   0.5187451832121683
  ],
  [
   {
    "det_thr": 0.4,
    "relink_gap_s": 3.0
   },
   0.5902316374124656
  ],
  [
   {
    "det_thr": 0.4,
    "relink_gap_s": 5.0
   },
   0.5773332463921835
  ],
  [
   {
    "det_thr": 0.4,
    "relink_gap_s": 8.0
   },
   0.5362944288822634
  ]
 ],
 "hota_thr": {
  "baseline": 0.3,
  "ours": 0.2
 }
}
```

## Selected configurations

```json
{
 "baseline": {
  "name": "baseline",
  "det_thr": 0.1,
  "two_stage": false,
  "low_thr": 0.1,
  "assoc": "euclid",
  "gate_scale": 1.0,
  "chi2_gate": 16.0,
  "max_age_s": 0.8,
  "noise": {
   "depth_sd0": 0.25,
   "depth_sd_k": 0.03,
   "lat_sd0": 0.1,
   "lat_sd_k": 0.006,
   "vel_sd": 1.0,
   "iso_sd": 1.0,
   "depth_aware": false
  },
  "relink": "lifetime",
  "relink_gap_s": 3.0,
  "stitch_chi2": 12.0,
  "stitch_vel_w": 0.5,
  "stitch_gap_w": 0.3,
  "smooth": true,
  "yaw_fix": false,
  "dedup": false,
  "dedup_frac": 0.6,
  "min_hits": 1,
  "prune_score": 0.0,
  "rescore_alpha": 1.0,
  "out_thr": 0.0
 },
 "ours": {
  "name": "ours",
  "det_thr": 0.3,
  "two_stage": true,
  "low_thr": 0.1,
  "assoc": "nll",
  "gate_scale": 1.5,
  "chi2_gate": 16.0,
  "max_age_s": 1.0,
  "noise": {
   "depth_sd0": 0.25,
   "depth_sd_k": 0.03,
   "lat_sd0": 0.1,
   "lat_sd_k": 0.006,
   "vel_sd": 1.0,
   "iso_sd": 1.0,
   "depth_aware": true
  },
  "relink": "global",
  "relink_gap_s": 3.0,
  "stitch_chi2": 12.0,
  "stitch_vel_w": 0.5,
  "stitch_gap_w": 0.3,
  "smooth": true,
  "yaw_fix": true,
  "dedup": true,
  "dedup_frac": 0.6,
  "min_hits": 3,
  "prune_score": 0.45,
  "rescore_alpha": 0.5,
  "out_thr": 0.25
 }
}
```
