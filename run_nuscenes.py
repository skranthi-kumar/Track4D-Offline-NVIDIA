"""Track frozen camera-only nuScenes detections and evaluate them.

Input: a detection results file in the standard nuScenes submission format (e.g. produced by the
frozen StreamPETR or GeoUP checkpoint), one file per split, used unchanged by every method.

  # 1. tune on a train slice (AMOTA objective); writes <out>/selected_<tracker>.json
  python run_nuscenes.py tune --dataroot /data/nuscenes --dets dets_train.json --split train --scenes 100

  # 2. report once on the 150 val scenes with the selected configs
  python run_nuscenes.py eval --dataroot /data/nuscenes --dets dets_val.json --split val \
      --config results/nusc/selected_baseline.json --config results/nusc/selected_ours.json --official

Metrics: AMOTA/AMOTP/MOTA/IDS/FRAG from the devkit (identical code path to TrackingEval;
``--official`` additionally runs TrackingEval.main() on the full split for the canonical numbers),
HOTA/IDF1 from TrackEval on the devkit-filtered tracks, IDs/GT, runtime per scene.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # tiny matrices: multithreaded BLAS only adds overhead

import argparse
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from trackfrag.core import Frame, NoiseModel
from trackfrag.metrics import combine, evaluate_sequence
from trackfrag.official import (TRACKING_NAMES, devkit_metrics, quat_to_yaw, threshold_tracks,
                                tracks_to_frames, trackeval_metrics, yaw_to_quat)
from trackfrag.tracker import TrackerConfig, baseline_config, ours_config, track_scene

HOTA_THRS = (0.2, 0.3, 0.4, 0.5, 0.6)


def load_scenes(nusc, split, n_scenes, det_path):
    """Detections grouped into scenes of time-ordered keyframes, in the global frame."""
    from nuscenes.utils.splits import create_splits_scenes
    names = sorted(create_splits_scenes()[split])
    if n_scenes:
        names = names[:n_scenes]
    by_name = {s["name"]: s for s in nusc.scene}
    dets = json.load(open(det_path))["results"]
    scenes = []
    for name in names:
        scene, tok = [], by_name[name]["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = np.array(nusc.get("ego_pose", sd["ego_pose_token"])["translation"][:2])
            boxes = [b for b in dets.get(tok, []) if b["detection_name"] in TRACKING_NAMES]
            if boxes:
                # nuScenes size is (w, l, h); internal boxes are (x, y, z, l, w, h, yaw)
                scene.append(Frame(
                    sample["timestamp"] * 1e-6,
                    np.array([[*b["translation"], b["size"][1], b["size"][0], b["size"][2], quat_to_yaw(b["rotation"])] for b in boxes]),
                    np.array([b["velocity"][:2] for b in boxes], float),
                    np.array([b["detection_score"] for b in boxes], float),
                    np.array([b["detection_name"] for b in boxes], dtype=object), ego, token=tok))
            else:
                scene.append(Frame.empty(sample["timestamp"] * 1e-6, ego, tok))
            tok = sample["next"]
        scenes.append(scene)
    return names, scenes


def to_submission(names, outputs):
    results = {}
    for name, scene in zip(names, outputs):
        for fr in scene:
            results[fr.token] = [{
                "sample_token": fr.token, "translation": [float(v) for v in b[:3]],
                "size": [float(b[4]), float(b[3]), float(b[5])], "rotation": yaw_to_quat(b[6]),
                "velocity": [float(v) for v in fr.vel[i]], "tracking_id": f"{name}_{fr.ids[i]}",
                "tracking_name": str(fr.cls[i]), "tracking_score": float(fr.score[i])}
                for i, b in enumerate(fr.boxes)]
    meta = {"use_camera": True, "use_lidar": False, "use_radar": False, "use_map": False, "use_external": False}
    return {"meta": meta, "results": results}


class Evaluator:
    """Loads and filters GT once for the chosen scenes, then scores any submission dict."""

    def __init__(self, nusc, split, names):
        from nuscenes.eval.common.config import config_factory
        from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
        from nuscenes.eval.tracking.data_classes import TrackingBox
        from nuscenes.eval.tracking.loaders import create_tracks
        self.nusc, self.split, self.cfg = nusc, split, config_factory("tracking_nips_2019")
        self._mods = (add_center_dist, filter_eval_boxes, create_tracks, TrackingBox)
        gt = load_gt(nusc, split, TrackingBox, verbose=False)
        keep = {self._scene_token(n) for n in names}
        tokens = [t for t in gt.sample_tokens if nusc.get("sample", t)["scene_token"] in keep]
        gt = self._subset(gt, tokens)
        gt = filter_eval_boxes(nusc, add_center_dist(nusc, gt), self.cfg.class_range, verbose=False)
        self.tokens = tokens
        self.keys = [self._scene_token(n) for n in names]
        self.tracks_gt = {k: v for k, v in create_tracks(gt, nusc, split, gt=True).items() if k in keep}

    def _scene_token(self, name):
        return next(s["token"] for s in self.nusc.scene if s["name"] == name)

    @staticmethod
    def _subset(boxes, tokens):
        from nuscenes.eval.common.data_classes import EvalBoxes
        out = EvalBoxes()
        for t in tokens:
            out.add_boxes(t, boxes[t])
        return out

    def __call__(self, submission, hota_thrs=HOTA_THRS):
        from nuscenes.eval.common.data_classes import EvalBoxes
        add_center_dist, filter_eval_boxes, create_tracks, TrackingBox = self._mods
        sub = {t: submission["results"].get(t, []) for t in self.tokens}
        pred = EvalBoxes.deserialize(sub, TrackingBox)
        pred = filter_eval_boxes(self.nusc, add_center_dist(self.nusc, pred), self.cfg.class_range, verbose=False)
        tracks_pred = create_tracks(pred, self.nusc, self.split, gt=False)
        tracks_pred = {k: tracks_pred[k] for k in self.keys}
        res = devkit_metrics(self.tracks_gt, tracks_pred)
        gts = tracks_to_frames(self.tracks_gt, self.keys)
        prs = tracks_to_frames(tracks_pred, self.keys)
        res["by_thr"] = {}
        for thr in hota_thrs:
            th = threshold_tracks(prs, thr)
            te = trackeval_metrics(gts, th, classes=TRACKING_NAMES, range_filter=False)
            own = combine([evaluate_sequence(g, p, range_filter=False) for g, p in zip(gts, th)])
            res["by_thr"][thr] = {k: te[k] for k in ("HOTA", "DetA", "AssA", "IDF1")} | {
                "IDs/GT": own["IDs/GT"], "per_class": te["per_class"]}
        return res


def run_tracker(cfg, scenes):
    cfg = replace(cfg, out_thr=0.0)
    t = time.perf_counter()
    out = [track_scene(s, cfg) for s in scenes]
    return out, 1000 * (time.perf_counter() - t) / len(scenes)


def load_config(path):
    d = json.load(open(path))
    d["noise"] = NoiseModel(**d["noise"])
    return TrackerConfig(**{k: v for k, v in d.items() if k != "hota_thr"}), d.get("hota_thr", 0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "eval"])
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--dets", required=True, help="nuScenes detection results json (frozen detector)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenes", type=int, default=0, help="use the first N scenes of the split (0 = all)")
    ap.add_argument("--config", action="append", default=[], help="selected config json(s) for eval")
    ap.add_argument("--official", action="store_true", help="also run TrackingEval.main() (needs the full split)")
    ap.add_argument("--out", default="results/nusc")
    a = ap.parse_args()
    from nuscenes import NuScenes
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    names, scenes = load_scenes(nusc, a.split, a.scenes, a.dets)
    ev = Evaluator(nusc, a.split, names)
    print(f"{len(scenes)} scenes, {sum(len(f) for s in scenes for f in s)} detections")

    if a.mode == "tune":
        grids = {"baseline": [baseline_config(det_thr=d, max_age_s=m, relink_gap_s=g)
                              for d, m, g in itertools.product([0.1, 0.2, 0.3], [0.8, 1.5], [1.67, 3.0])],
                 "ours": [ours_config(det_thr=d, relink_gap_s=g) for d, g in itertools.product([0.3, 0.4], [3.0, 5.0, 8.0])]}
        log = {}
        for name, grid in grids.items():
            best = None
            for cfg in grid:
                outputs, ms = run_tracker(cfg, scenes)
                r = ev(to_submission(names, outputs))
                log.setdefault(name, []).append({**{k: getattr(cfg, k) for k in ("det_thr", "max_age_s", "relink_gap_s")},
                                                 "amota": r["amota"], "frag": r["frag"], "ids": r["ids"]})
                print(name, log[name][-1])
                if best is None or r["amota"] > best[1]["amota"]:
                    best = (cfg, r)
            cfg, r = best
            thr = max(HOTA_THRS, key=lambda t: r["by_thr"][t]["HOTA"])
            (out / f"selected_{name}.json").write_text(json.dumps(asdict(cfg) | {"hota_thr": thr}, indent=1))
        (out / "tuning_log.json").write_text(json.dumps(log, indent=1))
        return

    rows = {}
    for path in a.config:
        cfg, thr = load_config(path)
        outputs, ms = run_tracker(cfg, scenes)
        sub = to_submission(names, outputs)
        sub_path = out / f"tracking_{cfg.name}_{a.split}.json"
        sub_path.write_text(json.dumps(sub))
        r = ev(sub, hota_thrs=(thr,))
        r["ms_per_scene"] = ms
        if a.official:
            from nuscenes.eval.common.config import config_factory
            from nuscenes.eval.tracking.evaluate import TrackingEval
            off = TrackingEval(config_factory("tracking_nips_2019"), str(sub_path), a.split, str(out / f"eval_{cfg.name}"),
                               a.version, a.dataroot, verbose=False).main(render_curves=False)
            r["official"] = {k: off[k] for k in ("amota", "amotp", "ids", "frag", "mota", "recall")}
        b = r["by_thr"][thr]
        rows[cfg.name] = r
        print(f"{cfg.name}: AMOTA {r['amota']:.3f} FRAG {int(r['frag'])} IDS {int(r['ids'])} | "
              f"HOTA {b['HOTA']:.1f} IDF1 {b['IDF1']:.1f} IDs/GT {b['IDs/GT']:.3f} | {ms:.0f} ms/scene")
    if "baseline" in rows and "ours" in rows:
        rb, ro = rows["baseline"], rows["ours"]
        red = 1 - ro["frag"] / max(rb["frag"], 1)
        print(f"success check: FRAG -{100*red:.1f}% ({'pass' if red >= 0.1 else 'FAIL'}), "
              f"AMOTA {rb['amota']:.3f}->{ro['amota']:.3f} ({'pass' if ro['amota'] >= rb['amota'] else 'FAIL'}), "
              f"IDS {int(rb['ids'])}->{int(ro['ids'])} ({'pass' if ro['ids'] <= rb['ids'] else 'FAIL'})")
    (out / f"eval_{a.split}.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
