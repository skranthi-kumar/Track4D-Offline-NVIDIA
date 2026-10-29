# trackfrag: fewer fragmented tracks from frozen camera-only 3D detections

An offline 3D multi-object tracker that extends the Foundation4D offline tracker
(arXiv:2610.30340, Sec. 5.3) to reduce track fragmentation without adding identity switches.
The detector is never touched: every method consumes the same detection file.

## Layout

| Path | What it is |
|---|---|
| `trackfrag/tracker.py` | The whole pipeline. One config switches between the baseline and ours, so every ablation is a single flag. |
| `trackfrag/core.py` | Frame/track types, constant-velocity Kalman filter, RTS smoother, depth-aware camera noise model. |
| `trackfrag/official.py` | Official metric code: nuScenes devkit (AMOTA, FRAG, IDS…) and TrackEval (HOTA, IDF1). |
| `trackfrag/metrics.py` | Our own HOTA / IDF1 / CLEAR / IDs-per-GT. Verified equal to TrackEval; used for the bootstrap. |
| `trackfrag/synthetic.py` | Camera-like synthetic scenes for development before real detections exist. |
| `run_synthetic.py` | Full protocol on synthetic data: tune on train → report on held-out val → ablations → sweep. |
| `run_nuscenes.py` | Same protocol on nuScenes with a real detection file (`tune` on a train slice, `eval` on val). |
| `REPORT.md` | Method, protocol, results, limitations. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The nuScenes devkit pins old dependencies. The pins in `requirements.txt` were verified on
Python 3.12. `trackfrag/official.py` restores a few aliases that motmetrics/TrackEval still
import (`collections.Iterable`, `inspect.getargspec`, `np.float`). It changes no metric code and
does nothing on Python ≤ 3.9. For submission-grade numbers, also run `--official`, which calls
`TrackingEval.main()` unchanged.

## Reproduce

```bash
python run_synthetic.py            # full synthetic protocol, ~65 min on 8 cores → results/synthetic_results.md
python run_synthetic.py --quick    # smoke test, ~3 min
```

nuScenes (needs the dataset plus detection files from one frozen detector, e.g. StreamPETR or GeoUP):

```bash
# 1) tune both trackers on a train slice; selects by AMOTA and writes results/nusc/selected_*.json
python run_nuscenes.py tune --dataroot /data/nuscenes --dets streampetr_train.json --split train --scenes 100
# 2) report once on the 150 val scenes
python run_nuscenes.py eval --dataroot /data/nuscenes --dets streampetr_val.json --split val \
    --config results/nusc/selected_baseline.json --config results/nusc/selected_ours.json --official
```

Detection files use the standard nuScenes detection submission format
(`{"results": {sample_token: [{translation, size, rotation, velocity, detection_name, detection_score}, …]}}`).
Seven tracking classes are used. Ego poses come from the dataset, as in standard nuScenes
camera-tracking practice.

## Method in one paragraph

Pass 1 is online Kalman tracking with a short lifetime. It uses **depth-aware Mahalanobis
association**: camera depth error grows with range and points along the viewing ray, so the
measurement covariance is anisotropic. A **ByteTrack-style second stage** recovers low-score
detections for established tracks. Pass 2 reconnects fragments with **one global assignment**
over tracklet ends and starts. The cost is a bidirectional Mahalanobis distance with propagated
covariances, plus velocity consistency, a size check and a gap penalty. This replaces the
baseline's longer-lifetime relink. Finally, RTS smoothing interpolates the gaps. **Duplicate-track
suppression**, track-level rescoring/pruning and a heading-flip fix clean up the output.
The baseline is the same code with these switches off. See `REPORT.md`.
