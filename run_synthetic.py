"""Synthetic benchmark following the project protocol, for development before real detections exist.

Protocol
  * train split (60 scenes) is used for every tuning decision; val split (150 scenes) is reported once.
  * primary metrics:   AMOTA, FRAG, IDS        - nuScenes devkit TrackingEvaluation (official code)
  * secondary metrics: HOTA, IDF1 (TrackEval), predicted tracks per GT object, runtime
  * tuning objective:  AMOTA on train. HOTA/IDF1 need a box set, so each method's track-score
                       threshold for them is also chosen on train.
  * success:           FRAG -10% or better, AMOTA not lower, IDS not higher (vs tuned baseline).

    python run_synthetic.py            # ~20 min on 8 cores
    python run_synthetic.py --quick    # ~3 min smoke test
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # small matrices; BLAS threads only oversubscribe the worker pool

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from trackfrag.core import NoiseModel
from trackfrag.metrics import combine, evaluate_sequence
from trackfrag.official import devkit_metrics, frames_to_tracks, threshold_tracks, trackeval_metrics
from trackfrag.synthetic import make_split
from trackfrag.tracker import baseline_config, ours_config, track_scene

HOTA_THRS = (0.2, 0.3, 0.4, 0.5, 0.6)


@lru_cache(maxsize=8)
def split(start, n, difficulty=1.0):
    return make_split(range(start, start + n), difficulty=difficulty)


def job(args):
    """Track one split with one config and score it. Runs in a worker process."""
    cfg, start, n, difficulty, hota_thr, with_devkit = args
    dets, gts = split(start, n, difficulty)
    cfg = replace(cfg, out_thr=0.0)  # AMOTA sweeps the score threshold itself
    t = time.perf_counter()
    out = [track_scene(s, cfg) for s in dets]
    sec = time.perf_counter() - t
    res = {"runtime_ms_per_frame": 1000 * sec / sum(len(s) for s in dets), "runtime_ms_per_scene": 1000 * sec / len(dets)}
    if with_devkit:
        dk = devkit_metrics(frames_to_tracks(gts, gt=True), frames_to_tracks(out, gt=False))
        res.update({k: dk[k] for k in ("amota", "amotp", "mota", "recall", "ids", "frag", "mt", "ml")})
    # threshold-dependent metrics, for every candidate threshold (cheap)
    res["by_thr"] = {}
    for thr in ([hota_thr] if hota_thr is not None else HOTA_THRS):
        th = threshold_tracks(out, thr)
        te = trackeval_metrics(gts, th)
        stats = [evaluate_sequence(g, p) for g, p in zip(gts, th)]
        own = combine(stats)
        res["by_thr"][thr] = {"HOTA": te["HOTA"], "DetA": te["DetA"], "AssA": te["AssA"], "IDF1": te["IDF1"],
                              "IDs/GT": own["IDs/GT"], "Pred_tracks": own["Pred_tracks"], "GT_traj": own["GT_traj"],
                              "per_class": te["per_class"], "stats": stats}
    return res


def run_all(jobs, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(job, jobs))


def bootstrap(sa, sb, n=1000, seed=0):
    """Paired bootstrap over scenes of (b - a) on the threshold-dependent metrics (own implementation,
    verified equal to TrackEval for HOTA/IDF1)."""
    rng = np.random.default_rng(seed)
    keys = ("HOTA", "IDF1", "IDs/GT")
    d = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(sa), len(sa))
        a, b = combine([sa[i] for i in idx]), combine([sb[i] for i in idx])
        for k in keys:
            d[k].append(b[k] - a[k])
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in d.items()}


def row(name, r, thr):
    b = r["by_thr"][thr]
    return (f"| {name} | {r['amota']:.3f} | {int(r['frag'])} | {int(r['ids'])} | {r['amotp']:.3f} | {r['mota']:.3f} | "
            f"{b['HOTA']:.1f} | {b['AssA']:.1f} | {b['IDF1']:.1f} | {b['IDs/GT']:.3f} | {r['runtime_ms_per_scene']:.0f} |\n")


HEAD = ("| Tracker | AMOTA ↑ | FRAG ↓ | IDS ↓ | AMOTP ↓ | MOTA ↑ | HOTA ↑ | AssA ↑ | IDF1 ↑ | IDs/GT ↓ | ms/scene |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    n_tr, n_val, n_sw = (12, 24, 12) if a.quick else (60, 150, 50)
    TR, VAL, SW = 0, 10_000, 20_000
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()

    # ---- 1. tuning on train (objective: AMOTA)
    bgrid = [baseline_config(det_thr=d, max_age_s=m, relink_gap_s=g)
             for d, m, g in itertools.product([0.1, 0.2, 0.3], [0.8, 1.5], [1.67, 3.0])]
    ogrid = [ours_config(det_thr=d, relink_gap_s=g) for d, g in itertools.product([0.3, 0.4], [3.0, 5.0, 8.0])]
    res = run_all([(c, TR, n_tr, 1.0, None, True) for c in bgrid + ogrid], a.workers)
    pick = lambda cs, rs: max(zip(cs, rs), key=lambda x: x[1]["amota"])
    base, rb = pick(bgrid, res[:len(bgrid)])
    ours, ro = pick(ogrid, res[len(bgrid):])
    thr_b = max(HOTA_THRS, key=lambda t: rb["by_thr"][t]["HOTA"])
    thr_o = max(HOTA_THRS, key=lambda t: ro["by_thr"][t]["HOTA"])
    tuning = {"baseline": [(dict(det_thr=c.det_thr, max_age_s=c.max_age_s, relink_gap_s=c.relink_gap_s), r["amota"])
                           for c, r in zip(bgrid, res[:len(bgrid)])],
              "ours": [(dict(det_thr=c.det_thr, relink_gap_s=c.relink_gap_s), r["amota"])
                       for c, r in zip(ogrid, res[len(bgrid):])],
              "hota_thr": {"baseline": thr_b, "ours": thr_o}}
    print(f"[{time.time()-t0:.0f}s] tuned: baseline det_thr={base.det_thr} max_age={base.max_age_s} gap={base.relink_gap_s} "
          f"(AMOTA {rb['amota']:.3f}); ours det_thr={ours.det_thr} gap={ours.relink_gap_s} (AMOTA {ro['amota']:.3f})")

    # ---- 2. held-out val: main comparison + ablations
    paper = baseline_config(name="paper defaults", det_thr=0.2)
    cum = [("+ depth-aware Mahalanobis association",
            replace(base, assoc="nll", noise=NoiseModel(depth_aware=True), gate_scale=1.5))]
    cum.append(("+ two-stage low-score association", replace(cum[-1][1], two_stage=True, low_thr=ours.low_thr, det_thr=ours.det_thr)))
    cum.append(("+ global bidirectional stitching", replace(cum[-1][1], relink="global", relink_gap_s=ours.relink_gap_s, max_age_s=ours.max_age_s)))
    loo = [("− depth-aware noise (isotropic)", replace(ours, noise=NoiseModel(depth_aware=False))),
           ("− Mahalanobis (planar distance)", replace(ours, assoc="euclid", gate_scale=1.0)),
           ("− two-stage association", replace(ours, two_stage=False)),
           ("− global stitching (baseline relink)", replace(ours, relink="lifetime", relink_gap_s=base.relink_gap_s)),
           ("− any second pass", replace(ours, relink="none")),
           ("− RTS smoothing / interpolation", replace(ours, smooth=False)),
           ("− duplicate suppression", replace(ours, dedup=False)),
           ("− pruning & rescoring", replace(ours, min_hits=1, prune_score=0.0, rescore_alpha=1.0))]
    vjobs = ([(paper, VAL, n_val, 1.0, thr_b, True), (base, VAL, n_val, 1.0, thr_b, True), (ours, VAL, n_val, 1.0, thr_o, True)]
             + [(c, VAL, n_val, 1.0, thr_o, True) for _, c in cum] + [(c, VAL, n_val, 1.0, thr_o, True) for _, c in loo]
             + [(c, SW, n_sw, d, t, True) for d in (0.5, 1.0, 1.5, 2.0) for c, t in ((base, thr_b), (ours, thr_o))])
    vr = run_all(vjobs, a.workers)
    r_paper, r_base, r_ours = vr[:3]
    r_cum = vr[3:3 + len(cum)]
    r_loo = vr[3 + len(cum):3 + len(cum) + len(loo)]
    r_sw = vr[3 + len(cum) + len(loo):]
    print(f"[{time.time()-t0:.0f}s] val done")

    # clean single-process timing (the pool above shares cores, so its timings are only indicative)
    dets_val = split(VAL, n_val)[0]
    for r, c in ((r_paper, paper), (r_base, base), (r_ours, ours)):
        t = time.perf_counter()
        for s in dets_val:
            track_scene(s, replace(c, out_thr=0.0))
        sec = time.perf_counter() - t
        r["runtime_ms_per_scene"] = 1000 * sec / len(dets_val)
        r["runtime_ms_per_frame"] = 1000 * sec / sum(len(s) for s in dets_val)

    # ---- 3. success criterion and significance
    frag_red = 1 - r_ours["frag"] / max(r_base["frag"], 1)
    checks = {"FRAG reduction >= 10%": (frag_red >= 0.10, f"{100*frag_red:.1f}%"),
              "AMOTA not lower": (r_ours["amota"] >= r_base["amota"], f"{r_base['amota']:.3f} -> {r_ours['amota']:.3f}"),
              "IDS not higher": (r_ours["ids"] <= r_base["ids"], f"{int(r_base['ids'])} -> {int(r_ours['ids'])}")}
    ci = bootstrap(r_base["by_thr"][thr_b]["stats"], r_ours["by_thr"][thr_o]["stats"])

    # ---- report
    md = ["# Synthetic benchmark results\n\n",
          f"Train {n_tr} scenes (seeds {TR}..), tuning only. Held-out val {n_val} scenes (seeds {VAL}..). "
          f"Sweep {n_sw} scenes/level (seeds {SW}..). 40 frames @ 2 Hz. AMOTA/FRAG/IDS/AMOTP/MOTA: nuScenes devkit "
          f"(tracking_nips_2019; FRAG/IDS at the best-MOTA threshold, summed over classes). HOTA/AssA/IDF1: TrackEval, "
          f"2 m center distance, class average, track-score threshold tuned on train "
          f"(baseline {thr_b}, ours {thr_o}). IDs/GT: distinct predicted ids per matched GT object. "
          f"ms/scene: tracking only, one CPU core; main-table rows timed in a dedicated single-process pass, "
          f"ablation rows inside the shared worker pool (indicative only).\n\n",
          "## Success criterion (val, vs tuned baseline)\n\n| Check | Pass | Value |\n|---|:---:|---:|\n",
          "".join(f"| {k} | {'✅' if ok else '❌'} | {v} |\n" for k, (ok, v) in checks.items()),
          "\n## Main comparison (held-out val)\n\n", HEAD,
          row("Baseline, paper defaults (Foundation4D-style)", r_paper, thr_b),
          row("Baseline, tuned on train", r_base, thr_b), row("**Ours**, tuned on train", r_ours, thr_o),
          "\n## Paired bootstrap over val scenes, ours − tuned baseline (1000 resamples)\n\n"
          "| Metric | Δ mean | 95% CI |\n|---|---:|---:|\n",
          "".join(f"| {k} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |\n" for k, (m, lo, hi) in ci.items()),
          "\n## Cumulative ablation (val): add one component at a time to the tuned baseline\n\n", HEAD,
          row("Baseline (tuned)", r_base, thr_b), "".join(row(n, r, thr_o) for (n, _), r in zip(cum, r_cum)),
          row("+ duplicate suppression, pruning, rescoring, yaw fix (= ours)", r_ours, thr_o),
          "\n## Leave-one-out ablation (val): remove one component from ours\n\n", HEAD, row("Ours (full)", r_ours, thr_o),
          "".join(row(n, r, thr_o) for (n, _), r in zip(loo, r_loo)),
          "\n## Robustness: detector difficulty sweep (noise, misses, clutter scaled)\n\n"
          "| Difficulty | AMOTA base | AMOTA ours | FRAG base | FRAG ours | IDS base | IDS ours | HOTA base | HOTA ours |\n"
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
          "".join(f"| {d} | {b['amota']:.3f} | {o['amota']:.3f} | {int(b['frag'])} | {int(o['frag'])} | {int(b['ids'])} | "
                  f"{int(o['ids'])} | {b['by_thr'][thr_b]['HOTA']:.1f} | {o['by_thr'][thr_o]['HOTA']:.1f} |\n"
                  for d, b, o in zip((0.5, 1.0, 1.5, 2.0), r_sw[0::2], r_sw[1::2])),
          "\n## Tuning on train (AMOTA)\n\n```json\n" + json.dumps(tuning, indent=1) + "\n```\n",
          "\n## Selected configurations\n\n```json\n" + json.dumps({"baseline": asdict(base), "ours": asdict(ours)}, indent=1, default=str) + "\n```\n"]
    (out / "synthetic_results.md").write_text("".join(md))
    strip = lambda r: {k: v for k, v in r.items() if k != "by_thr"} | {
        "by_thr": {t: {k: v for k, v in b.items() if k != "stats"} for t, b in r["by_thr"].items()}}
    (out / "synthetic_results.json").write_text(json.dumps({
        "checks": {k: [bool(ok), v] for k, (ok, v) in checks.items()}, "bootstrap": ci, "tuning": tuning,
        "val": {"paper": strip(r_paper), "baseline": strip(r_base), "ours": strip(r_ours)},
        "ablation_cumulative": {n: strip(r) for (n, _), r in zip(cum, r_cum)},
        "ablation_leave_one_out": {n: strip(r) for (n, _), r in zip(loo, r_loo)},
        "sweep": [{"difficulty": d, "baseline": strip(b), "ours": strip(o)} for d, b, o in zip((0.5, 1.0, 1.5, 2.0), r_sw[0::2], r_sw[1::2])],
    }, indent=1, default=float))
    print("".join(md[:8]))
    print(f"[{time.time()-t0:.0f}s] wrote {out/'synthetic_results.md'}")


if __name__ == "__main__":
    main()
