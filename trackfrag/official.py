"""Official metric code applied to tracker output held as lists of Frame.

* AMOTA / AMOTP / MOTA / IDS / FRAG / MT / ML: nuScenes devkit ``TrackingEvaluation``, with the
  same per-class protocol as ``TrackingEval.evaluate`` (track-mean scores, interpolation,
  40 recall thresholds, IDS/FRAG at the best-MOTA threshold, summed over classes).
* HOTA / DetA / AssA / IDF1: TrackEval's ``HOTA`` and ``Identity`` metrics fed with BEV
  center-distance similarity (s = max(0, 1 - d / 2 m), class-aware), combined over sequences.

For nuScenes data the GT tracks come from the devkit loaders (range / num-points filtered);
for synthetic data they are built from the generator's ground truth after the same range filter.
"""
from collections import defaultdict

import numpy as np

from .core import CLASS_RANGE, Frame
from .metrics import _filter, _in_range

TRACKING_NAMES = ["bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"]


def _py312_compat():
    """motmetrics 1.1.3 (pinned by the nuScenes devkit) predates Python 3.10. Restore the removed
    aliases it imports; no metric code is changed. On Python <= 3.9 this is a no-op."""
    import collections
    import collections.abc
    import inspect
    for n in ("Iterable", "Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, n):
            setattr(collections, n, getattr(collections.abc, n))
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    # TrackEval (master) still uses the numpy aliases removed in numpy 1.24.
    for n, t in (("float", float), ("int", int), ("bool", bool)):
        if n not in np.__dict__:
            setattr(np, n, t)


_py312_compat()


# ----------------------------------------------------------------------------- conversions
def yaw_to_quat(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def quat_to_yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def frames_to_tracks(scenes, gt, range_filter=True):
    """{scene_key: {timestamp_us: [TrackingBox]}} in the form create_tracks() returns."""
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.data_classes import TrackingBox
    from nuscenes.eval.tracking.loaders import interpolate_tracks
    config_factory("tracking_nips_2019")  # registers the devkit's TRACKING_NAMES
    tracks = {}
    for si, scene in enumerate(scenes):
        st = defaultdict(list)
        for k, fr in enumerate(scene):
            ts = int(round(fr.t * 1e6))
            st[ts] = []
            if range_filter:
                fr = _filter(fr, _in_range(fr))
            for i in range(len(fr)):
                b = fr.boxes[i]
                ego = fr.ego if fr.ego is not None else np.zeros(2)
                st[ts].append(TrackingBox(
                    sample_token=fr.token or f"{si}_{k}", translation=tuple(map(float, b[:3])),
                    size=tuple(map(float, np.maximum(b[3:6], 1e-3))), rotation=tuple(yaw_to_quat(b[6])),
                    velocity=tuple(map(float, fr.vel[i])),
                    ego_translation=(float(b[0] - ego[0]), float(b[1] - ego[1]), 0.0),
                    num_pts=1 if gt else -1, tracking_id=str(fr.ids[i]), tracking_name=str(fr.cls[i]),
                    tracking_score=float(fr.score[i]) if not gt else -1.0))
        if not gt:  # same as create_tracks(): every box gets its track's mean score
            sc = defaultdict(list)
            for boxes in st.values():
                for bx in boxes:
                    sc[bx.tracking_id].append(bx.tracking_score)
            mean = {t: float(np.mean(v)) for t, v in sc.items()}
            for boxes in st.values():
                for bx in boxes:
                    bx.tracking_score = mean[bx.tracking_id]
        st = interpolate_tracks(st)
        tracks[si] = defaultdict(list, sorted(st.items(), key=lambda kv: kv[0]))
    return tracks


def tracks_to_frames(tracks, keys=None):
    """Inverse of frames_to_tracks (used for nuScenes GT/pred loaded by the devkit)."""
    scenes = []
    for key in keys if keys is not None else tracks:
        scene = []
        for ts, boxes in sorted(tracks[key].items()):
            if not boxes:
                scene.append(Frame.empty(ts * 1e-6, with_ids=True))
                continue
            scene.append(Frame(
                ts * 1e-6,
                np.array([[*b.translation, *b.size, quat_to_yaw(b.rotation)] for b in boxes]),
                np.array([b.velocity[:2] for b in boxes], float),
                np.array([b.tracking_score for b in boxes], float),
                np.array([b.tracking_name for b in boxes], dtype=object), None,
                np.array([hash(b.tracking_id) & 0x7FFFFFFF for b in boxes], np.int64), boxes[0].sample_token))
        scenes.append(scene)
    return scenes


# ----------------------------------------------------------------------------- nuScenes devkit
def devkit_metrics(tracks_gt, tracks_pred, classes=None, verbose=False):
    """AMOTA etc. exactly as TrackingEval.evaluate() computes them. Returns the serialized dict."""
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.algo import TrackingEvaluation
    from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP
    from nuscenes.eval.tracking.data_classes import TrackingMetricData, TrackingMetrics
    cfg = config_factory("tracking_nips_2019")
    metrics = TrackingMetrics(cfg)
    for cname in classes or cfg.class_names:
        n_gt = sum(1 for s in tracks_gt.values() for bs in s.values() for b in bs if b.tracking_name == cname)
        if n_gt == 0:
            continue  # class absent from this data; the devkit would report nan and skip it too
        md = TrackingEvaluation(tracks_gt, tracks_pred, cname, cfg.dist_fcn_callable, cfg.dist_th_tp,
                                cfg.min_recall, num_thresholds=TrackingMetricData.nelem,
                                metric_worst=cfg.metric_worst, verbose=verbose).accumulate()
        if not np.all(np.isnan(md.mota)):
            best = np.nanargmax(md.mota)
            for m in MOT_METRIC_MAP.values():
                if m:
                    metrics.add_label_metric(m, cname, md.get_metric(m)[best])
        for m, src in AVG_METRIC_MAP.items():
            v = np.array(md.get_metric(src))
            if np.all(np.isnan(v)):
                val = np.nan
            else:
                v[np.isnan(v)] = cfg.metric_worst[m]
                val = float(np.nanmean(v))
            metrics.add_label_metric(m, cname, val)
    s = metrics.serialize()
    return {k: s[k] for k in ("amota", "amotp", "mota", "motp", "recall", "ids", "frag", "mt", "ml",
                              "tp", "fp", "fn", "tid", "lgd") if k in s} | {"per_class": s["label_metrics"]}


# ----------------------------------------------------------------------------- TrackEval
def _trackeval_data(gt_scene, pr_scene, cname, thr=2.0):
    gt_ids, pr_ids, sims = [], [], []
    for g, p in zip(gt_scene, pr_scene):
        gm, pm = g.cls == cname, p.cls == cname
        gi, pi = g.ids[gm], p.ids[pm]
        if gm.any() and pm.any():
            d = np.linalg.norm(g.boxes[gm][:, None, :2] - p.boxes[pm][None, :, :2], axis=2)
            s = np.clip(1 - d / thr, 0, 1)
        else:
            s = np.zeros((gm.sum(), pm.sum()))
        gt_ids.append(gi); pr_ids.append(pi); sims.append(s)
    ug = np.unique(np.concatenate(gt_ids + [np.zeros(0, np.int64)]))
    up = np.unique(np.concatenate(pr_ids + [np.zeros(0, np.int64)]))
    gmap = {v: i for i, v in enumerate(ug)}; pmap = {v: i for i, v in enumerate(up)}
    return dict(num_timesteps=len(gt_scene), num_gt_ids=len(ug), num_tracker_ids=len(up),
                num_gt_dets=int(sum(len(x) for x in gt_ids)), num_tracker_dets=int(sum(len(x) for x in pr_ids)),
                gt_ids=[np.array([gmap[v] for v in x], int) for x in gt_ids],
                tracker_ids=[np.array([pmap[v] for v in x], int) for x in pr_ids],
                similarity_scores=sims)


def trackeval_metrics(gt_scenes, pr_scenes, classes=None, range_filter=True):
    """HOTA / DetA / AssA / IDF1 with TrackEval. Per class combined over sequences, then the
    class average (nuScenes / Foundation4D convention) and the pooled 'all classes' result."""
    from trackeval.metrics import HOTA, Identity
    hota, ident = HOTA(), Identity({"THRESHOLD": 0.0 + 1e-9, "PRINT_CONFIG": False})
    if range_filter:
        gt_scenes = [[_filter(f, _in_range(f)) for f in s] for s in gt_scenes]
        pr_scenes = [[_filter(f, _in_range(f)) for f in s] for s in pr_scenes]
    present = {c for s in gt_scenes for f in s for c in f.cls}
    # classes without GT are skipped, as the devkit does (they would average in as 0)
    classes = [c for c in classes if c in present] if classes else sorted(present)
    per_class = {}
    for c in classes:
        hs, is_ = {}, {}
        for i, (g, p) in enumerate(zip(gt_scenes, pr_scenes)):
            data = _trackeval_data(g, p, c)
            hs[i], is_[i] = hota.eval_sequence(data), ident.eval_sequence(data)
        h, idn = hota.combine_sequences(hs), ident.combine_sequences(is_)
        per_class[c] = {"HOTA": 100 * float(np.mean(h["HOTA"])), "DetA": 100 * float(np.mean(h["DetA"])),
                        "AssA": 100 * float(np.mean(h["AssA"])), "IDF1": 100 * float(idn["IDF1"])}
    avg = {k: float(np.mean([v[k] for v in per_class.values()])) for k in ("HOTA", "DetA", "AssA", "IDF1")}
    return avg | {"per_class": per_class}


def threshold_tracks(scenes, thr):
    """Keep whole tracks whose mean box score reaches ``thr`` (for threshold-dependent metrics)."""
    out = []
    for scene in scenes:
        sc = defaultdict(list)
        for f in scene:
            for i, s in zip(f.ids, f.score):
                sc[i].append(s)
        keep = {i for i, v in sc.items() if np.mean(v) >= thr}
        out.append([_filter(f, np.array([i in keep for i in f.ids], bool)) for f in scene])
    return out
