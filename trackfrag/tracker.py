"""Offline 3D multi-object tracker for frozen camera-only detections.

One configurable pipeline covers both the baseline and the proposed tracker, so every
ablation differs from the baseline in exactly the switches it names:

  pass 1  online Kalman tracking with a short lifetime     (both)
  pass 2  reconnect fragments                               baseline: greedy lifetime relink
                                                            ours:     global bidirectional stitching
  final   RTS smoothing, gap interpolation, median size     (both)
          duplicate suppression, track rescoring, pruning   (ours)

The baseline reproduces the offline tracker of Foundation4D (arXiv:2610.30340, Sec. 5.3 and
Tab. 10): constant-velocity KF on planar position + detector velocity, Hungarian matching on
planar distance under a gate, a short-lifetime pass, a second pass with a longer lifetime,
gap interpolation, RTS smoothing, unwrapped yaw and median cuboid dimensions.
"""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import (CLASS_ACCEL, CLASS_GATE, Frame, Meas, NoiseModel, Tracklet,
                   cv_model, kf_update, rts_smooth, wrap_angle)

BIG = 1e9


@dataclass
class TrackerConfig:
    name: str = "baseline"
    # --- pass 1: online association
    det_thr: float = 0.3          # detections below this are ignored (stage-1 threshold)
    two_stage: bool = False       # ByteTrack-style second stage on low-score detections
    low_thr: float = 0.1
    assoc: str = "euclid"         # "euclid" (planar distance) or "nll" (Mahalanobis negative log-likelihood)
    gate_scale: float = 1.0       # multiplies CLASS_GATE
    chi2_gate: float = 16.0       # Mahalanobis gate for assoc="nll" (2 dof)
    max_age_s: float = 0.8        # pass-1 track lifetime without detections
    noise: NoiseModel = field(default_factory=lambda: NoiseModel(depth_aware=False))
    # --- pass 2: fragment reconnection
    relink: str = "lifetime"      # "lifetime" (baseline), "global" (ours) or "none"
    relink_gap_s: float = 1.67    # longest gap that can be bridged
    stitch_chi2: float = 12.0
    stitch_vel_w: float = 0.5     # weight of the velocity-consistency term
    stitch_gap_w: float = 0.3     # cost per second of gap (prefers short bridges)
    # --- final trajectory refinement
    smooth: bool = True           # RTS smoothing + interpolation of gaps
    yaw_fix: bool = False         # undo 180-degree heading flips using the smoothed velocity
    dedup: bool = False           # drop tracks that shadow a stronger track (ray duplicates)
    dedup_frac: float = 0.6
    min_hits: int = 1             # prune tracks with fewer detections ...
    prune_score: float = 0.0      # ... unless their best score reaches this
    rescore_alpha: float = 1.0    # box score = a * det score + (1 - a) * track mean score
    out_thr: float = 0.0          # output boxes below this score are dropped


def baseline_config(**kw):
    return replace(TrackerConfig(name="baseline"), **kw)


def ours_config(**kw):
    cfg = TrackerConfig(
        name="ours", det_thr=0.35, two_stage=True, low_thr=0.1, assoc="nll", gate_scale=1.5,
        max_age_s=1.0, noise=NoiseModel(depth_aware=True), relink="global", relink_gap_s=5.0,
        yaw_fix=True, dedup=True, min_hits=3, prune_score=0.45, rescore_alpha=0.5, out_thr=0.25)
    return replace(cfg, **kw)


# ----------------------------------------------------------------------------- pass 1
class _Live:
    __slots__ = ("trk", "x", "P", "t", "last_t", "hits")

    def __init__(self, cls, m, t):
        self.trk = Tracklet(cls, [m])
        self.x, self.P, self.t, self.last_t, self.hits = m.z.copy(), m.R.copy(), t, t, 1


def _cost_matrix(live, xy, R, cls, cfg):
    """(M, N) association cost; BIG where gated out."""
    pos = np.array([tr.x[:2] for tr in live])
    y = xy[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(y, axis=2)
    tcls = np.array([tr.trk.cls for tr in live])
    gate = np.array([CLASS_GATE.get(c, 2.0) for c in tcls])[:, None] * cfg.gate_scale
    valid = (tcls[:, None] == cls[None, :]) & (dist < gate)
    if cfg.assoc == "euclid":
        cost = dist
    else:
        P = np.array([tr.P[:2, :2] for tr in live])
        S = P[:, None] + R[None, :, :2, :2]
        a, b, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
        det = a * d - b * b
        d2 = (d * y[..., 0] ** 2 - 2 * b * y[..., 0] * y[..., 1] + a * y[..., 1] ** 2) / det
        cost = d2 + np.log(det)
        valid &= d2 < cfg.chi2_gate
    return np.where(valid, cost, BIG)


def _match(cost):
    if cost.size == 0:
        return []
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if cost[i, j] < BIG]


def online_pass(scene, cfg):
    """Pass 1: frame-by-frame Kalman tracking. Returns tracklets (lists of assigned detections)."""
    live, done = [], []
    for k, fr in enumerate(scene):
        for tr in live:
            F, Q = cv_model(fr.t - tr.t, CLASS_ACCEL.get(tr.trk.cls, 2.0))
            tr.x, tr.P, tr.t = F @ tr.x, F @ tr.P @ F.T + Q, fr.t
        lo = cfg.low_thr if cfg.two_stage else cfg.det_thr
        keep = np.flatnonzero(fr.score >= lo)
        xy = fr.boxes[keep, :2]
        R = cfg.noise.meas_cov(xy, fr.ego)
        cls = fr.cls[keep]
        high = np.flatnonzero(fr.score[keep] >= cfg.det_thr)
        low = np.flatnonzero(fr.score[keep] < cfg.det_thr)

        def apply(pairs, trk_idx, det_idx):
            for i, j in pairs:
                tr, d = live[trk_idx[i]], det_idx[j]
                g = keep[d]
                z = np.r_[fr.boxes[g, :2], fr.vel[g]]
                tr.x, tr.P = kf_update(tr.x, tr.P, z, R[d])
                tr.trk.meas.append(Meas(k, z, R[d], fr.boxes[g].copy(), float(fr.score[g])))
                tr.last_t, tr.hits = fr.t, tr.hits + 1

        all_t = np.arange(len(live))
        pairs = _match(_cost_matrix(live, xy[high], R[high], cls[high], cfg)) if len(live) and len(high) else []
        apply(pairs, all_t, high)
        matched_t = {i for i, _ in pairs}
        matched_d = {high[j] for _, j in pairs}
        if cfg.two_stage and len(low):
            rest = np.array([i for i in all_t if i not in matched_t and live[i].hits >= 2], dtype=int)
            if len(rest):
                sub = [live[i] for i in rest]
                pairs2 = _match(_cost_matrix(sub, xy[low], R[low], cls[low], cfg))
                apply(pairs2, rest, low)
        for j in high:
            if j not in matched_d:
                g = keep[j]
                z = np.r_[fr.boxes[g, :2], fr.vel[g]]
                live.append(_Live(cls[j], Meas(k, z, R[j], fr.boxes[g].copy(), float(fr.score[g])), fr.t))
        alive = []
        for tr in live:
            (alive if fr.t - tr.last_t <= cfg.max_age_s + 1e-6 else done).append(tr)
        live = alive
    return [tr.trk for tr in done + live]


# ----------------------------------------------------------------------------- pass 2
def _endpoints(trk, times):
    ks, xs, Ps = rts_smooth(trk.meas, times, CLASS_ACCEL.get(trk.cls, 2.0))
    return xs[0], Ps[0], xs[-1], Ps[-1]


def _size(trk):
    return np.median(np.array([m.box[3:6] for m in trk.meas]), axis=0)


def relink(tracklets, times, cfg):
    """Pass 2: join tracklet A (ending) to tracklet B (starting later) by one global assignment.

    baseline ("lifetime"): A's smoothed end is extrapolated to B's start; cost is planar distance
        under the class gate, for gaps up to ``relink_gap_s``.
    ours ("global"): symmetric Mahalanobis distance of A-forward vs B-start and B-backward vs
        A-end with propagated covariances, plus velocity consistency, a size check and a gap
        penalty. Gaps up to ``relink_gap_s``.
    """
    n = len(tracklets)
    if n < 2 or cfg.relink == "none":
        return tracklets
    ends = [_endpoints(t, times) for t in tracklets]
    sizes = [_size(t) for t in tracklets]
    C = np.full((n, n), BIG)
    for a, A in enumerate(tracklets):
        sA, PsA, eA, PeA = ends[a]
        acc = CLASS_ACCEL.get(A.cls, 2.0)
        for b, B in enumerate(tracklets):
            if a == b or A.cls != B.cls or B.k0 <= A.k1:
                continue
            dt = times[B.k0] - times[A.k1]
            if dt > cfg.relink_gap_s + 1e-6:
                continue
            sB, PsB, _, _ = ends[b]
            F, Q = cv_model(dt, acc)
            fwd = F @ eA
            if cfg.relink == "lifetime":
                d = np.linalg.norm(fwd[:2] - sB[:2])
                if d < CLASS_GATE.get(A.cls, 2.0):
                    C[a, b] = d
                continue
            ratio = np.max(np.maximum(sizes[a], sizes[b]) / np.maximum(np.minimum(sizes[a], sizes[b]), 1e-3))
            if ratio > 1.6:
                continue
            Fb, _ = cv_model(-dt, acc)
            bwd = Fb @ sB
            Pf = F @ PeA @ F.T + Q
            Pb = Fb @ PsB @ Fb.T + Q
            yf, yb = sB[:2] - fwd[:2], eA[:2] - bwd[:2]
            d2f = yf @ np.linalg.solve(Pf[:2, :2] + PsB[:2, :2], yf)
            d2b = yb @ np.linalg.solve(Pb[:2, :2] + PeA[:2, :2], yb)
            d2 = 0.5 * (d2f + d2b)
            dv = sB[2:] - eA[2:]
            d2v = dv @ np.linalg.solve(PeA[2:, 2:] + PsB[2:, 2:] + Q[2:, 2:], dv)
            if d2 < cfg.stitch_chi2 and min(np.linalg.norm(yf), np.linalg.norm(yb)) < \
                    CLASS_GATE.get(A.cls, 2.0) * cfg.gate_scale * (1 + 0.25 * dt):
                C[a, b] = d2 + cfg.stitch_vel_w * d2v + cfg.stitch_gap_w * dt
    nxt = dict(_match(C))
    has_prev = set(nxt.values())
    merged = []
    for a in range(n):
        if a in has_prev:
            continue
        trk = Tracklet(tracklets[a].cls, list(tracklets[a].meas))
        while a in nxt:
            a = nxt[a]
            trk.meas.extend(tracklets[a].meas)
        merged.append(trk)
    return merged


# ----------------------------------------------------------------------------- final stage
def _render(trk, scene, times, cfg):
    """Per-frame boxes of one trajectory: dict k -> (box(7), vel(2), score, measured)."""
    size = _size(trk)
    mk = np.array([m.k for m in trk.meas])
    yaw = np.unwrap(np.array([m.box[6] for m in trk.meas]))
    zs = np.array([m.box[2] for m in trk.meas])
    sc = np.array([m.score for m in trk.meas])
    out = {}
    if not cfg.smooth:
        for m in trk.meas:
            out[m.k] = (np.r_[m.box[:3], size, m.box[6]], m.z[2:], m.score, True)
        return out
    ks, xs, _ = rts_smooth(trk.meas, times, CLASS_ACCEL.get(trk.cls, 2.0))
    if cfg.yaw_fix:
        # Camera detectors often flip heading by pi; align measured yaw to the motion direction.
        for i, k in enumerate(mk):
            v = xs[k - ks[0], 2:]
            if np.linalg.norm(v) > 2.0:
                head = np.arctan2(v[1], v[0])
                if abs(wrap_angle(yaw[i] - head)) > np.pi / 2:
                    yaw[i] += np.pi
        yaw = np.unwrap(yaw)
    zi = np.interp(ks, mk, zs)
    yi = np.interp(ks, mk, yaw)
    si = np.interp(ks, mk, sc)
    measured = set(mk.tolist())
    for i, k in enumerate(ks):
        out[int(k)] = (np.r_[xs[i, :2], zi[i], size, wrap_angle(yi[i])], xs[i, 2:], float(si[i]), int(k) in measured)
    return out


def _dedup(rendered, tracklets, cfg):
    """Drop a track whose boxes sit on top of a stronger track for most of its life."""
    order = np.argsort([-sum(m.score for m in t.meas) for t in tracklets])
    kept = []
    for i in order:
        ri, ci = rendered[i], tracklets[i].cls
        dup = False
        for j in kept:
            if tracklets[j].cls != ci:
                continue
            common = set(ri) & set(rendered[j])
            if len(common) < cfg.dedup_frac * len(ri):
                continue
            d = np.median([np.linalg.norm(ri[k][0][:2] - rendered[j][k][0][:2]) for k in common])
            if d < 0.5 * CLASS_GATE.get(ci, 2.0):
                dup = True
                break
        if not dup:
            kept.append(i)
    return sorted(kept)


def track_scene(scene, cfg):
    """Run the full offline pipeline on one scene (list of Frame). Returns a list of Frame with ids."""
    times = np.array([f.t for f in scene])
    trks = online_pass(scene, cfg)
    trks = relink(trks, times, cfg)
    trks = [t for t in trks if len(t.meas) >= cfg.min_hits or max(m.score for m in t.meas) >= cfg.prune_score]
    rendered = [_render(t, scene, times, cfg) for t in trks]
    idx = _dedup(rendered, trks, cfg) if cfg.dedup else range(len(trks))
    per_frame = [[] for _ in scene]
    for tid, i in enumerate(idx):
        tmean = trks[i].mean_score()
        for k, (box, vel, s, _) in rendered[i].items():
            s = cfg.rescore_alpha * s + (1 - cfg.rescore_alpha) * tmean
            if s >= cfg.out_thr:
                per_frame[k].append((tid, box, vel, s, trks[i].cls))
    out = []
    for fr, items in zip(scene, per_frame):
        if not items:
            out.append(Frame.empty(fr.t, fr.ego, fr.token, with_ids=True))
            continue
        ids, boxes, vels, scores, clss = zip(*items)
        out.append(Frame(fr.t, np.array(boxes), np.array(vels), np.array(scores),
                         np.array(clss, dtype=object), fr.ego, np.array(ids), fr.token))
    return out
