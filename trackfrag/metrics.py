"""Tracking metrics with BEV center-distance matching (as in nuScenes and Foundation4D Sec. 7.2.2).

* HOTA / DetA / AssA / LocA - TrackEval algorithm, similarity s = max(0, 1 - d / 2 m), class-aware.
* IDF1 / IDP / IDR          - global identity matching, d < 2 m.
* CLEAR MOT                 - MOTA, MOTP, IDS, FRAG, MT, ML (d < 2 m, matches kept across frames).
* Fragmentation extras      - IDs/GT: mean number of distinct predicted ids that cover one
                              ground-truth object (1.0 is perfect); Frag/GT.

Sequences are combined like TrackEval: counts are summed, association scores are TP-weighted.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import CLASS_RANGE

ALPHAS = np.arange(0.05, 0.99, 0.05)
DIST_THR = 2.0


def _in_range(fr):
    if fr.ego is None or len(fr) == 0:
        return np.ones(len(fr), bool)
    r = np.linalg.norm(fr.boxes[:, :2] - fr.ego[None, :2], axis=1)
    return r <= np.array([CLASS_RANGE.get(c, 50) for c in fr.cls])


def _dist(g, p):
    d = np.linalg.norm(g.boxes[:, None, :2] - p.boxes[None, :, :2], axis=2)
    d[g.cls[:, None] != p.cls[None, :]] = np.inf
    return d


def _filter(fr, m):
    from .core import Frame
    return Frame(fr.t, fr.boxes[m], fr.vel[m], fr.score[m], fr.cls[m], fr.ego, fr.ids[m], fr.token)


def evaluate_sequence(gt, pr, thr=DIST_THR, range_filter=True):
    """gt, pr: lists of Frame (same length) with ids. Returns raw per-sequence statistics."""
    if range_filter:
        gt = [_filter(g, _in_range(g)) for g in gt]
        pr = [_filter(p, _in_range(p)) for p in pr]
    gids = np.unique(np.concatenate([g.ids for g in gt] + [np.zeros(0, int)]))
    pids = np.unique(np.concatenate([p.ids for p in pr] + [np.zeros(0, int)]))
    gmap = {v: i for i, v in enumerate(gids)}
    pmap = {v: i for i, v in enumerate(pids)}
    nG, nP = len(gids), len(pids)
    frames = []
    for g, p in zip(gt, pr):
        gi = np.array([gmap[v] for v in g.ids], int)
        pi = np.array([pmap[v] for v in p.ids], int)
        d = _dist(g, p) if len(g) and len(p) else np.zeros((len(g), len(p)))
        frames.append((gi, pi, d))

    # ---------------- HOTA
    A = len(ALPHAS)
    pmc = np.zeros((nG, nP)); gcnt = np.zeros(nG); pcnt = np.zeros(nP)
    for gi, pi, d in frames:
        gcnt[gi] += 1; pcnt[pi] += 1
        if len(gi) and len(pi):
            S = np.clip(1 - d / thr, 0, 1)
            den = S.sum(0)[None] + S.sum(1)[:, None] - S
            pmc[gi[:, None], pi[None]] += np.where(den > 0, S / np.maximum(den, 1e-12), 0)
    ga = pmc / np.maximum(gcnt[:, None] + pcnt[None] - pmc, 1e-12)
    TP = np.zeros(A); FN = np.zeros(A); FP = np.zeros(A); LOC = np.zeros(A)
    mc = np.zeros((A, nG, nP))
    for gi, pi, d in frames:
        if not len(gi) or not len(pi):
            FN += len(gi); FP += len(pi)
            continue
        S = np.clip(1 - d / thr, 0, 1)
        r, c = linear_sum_assignment(-(ga[gi[:, None], pi[None]] * S))
        for a, al in enumerate(ALPHAS):
            ok = S[r, c] >= al - 1e-10
            n = ok.sum()
            TP[a] += n; FN[a] += len(gi) - n; FP[a] += len(pi) - n
            LOC[a] += S[r[ok], c[ok]].sum()
            np.add.at(mc[a], (gi[r[ok]], pi[c[ok]]), 1)
    AssA = np.zeros(A); AssRe = np.zeros(A); AssPr = np.zeros(A)
    for a in range(A):
        m = mc[a]
        AssA[a] = (m * m / np.maximum(gcnt[:, None] + pcnt[None] - m, 1)).sum() / max(TP[a], 1)
        AssRe[a] = (m * m / np.maximum(gcnt[:, None], 1)).sum() / max(TP[a], 1)
        AssPr[a] = (m * m / np.maximum(pcnt[None], 1)).sum() / max(TP[a], 1)

    # ---------------- IDF1
    co = np.zeros((nG, nP))
    for gi, pi, d in frames:
        if len(gi) and len(pi):
            np.add.at(co, (np.repeat(gi, len(pi)), np.tile(pi, len(gi))), (d < thr).ravel())
    if nG and nP:
        r, c = linear_sum_assignment(-co)
        idtp = co[r, c].sum()
    else:
        idtp = 0.0

    # ---------------- CLEAR MOT + fragmentation
    prev = {}          # gt id -> pred id matched in the previous frame it was matched
    last = {}          # gt id -> last matched pred id (for IDS)
    status = {g: [] for g in range(nG)}
    ids_per_gt = {g: set() for g in range(nG)}
    tp = fn = fp = ids = 0
    dsum = 0.0
    for gi, pi, d in frames:
        matched = {}
        if len(gi) and len(pi):
            score = np.where(d < thr, thr - d, -1.0)
            for a in range(len(gi)):
                b = np.flatnonzero(pi == prev.get(gi[a], -1))
                if len(b) and d[a, b[0]] < thr:
                    score[a, b[0]] += 1000.0
            r, c = linear_sum_assignment(-score)
            for a, b in zip(r, c):
                if d[a, b] < thr:
                    matched[gi[a]] = (pi[b], d[a, b])
        for g in gi:
            if g in matched:
                p, dd = matched[g]
                if g in last and last[g] != p:
                    ids += 1
                last[g] = prev[g] = p
                ids_per_gt[g].add(p)
                tp += 1; dsum += dd
                status[g].append(1)
            else:
                fn += 1
                status[g].append(0)
        fp += len(pi) - len(matched)
    frag = mt = ml = 0
    for s in status.values():
        s = np.array(s)
        if not len(s):
            continue
        on = np.flatnonzero(s)
        if len(on):
            # number of times tracking resumes after an interruption
            frag += int(np.sum(np.diff(on) > 1))
        ratio = s.mean()
        mt += ratio >= 0.8; ml += ratio <= 0.2
    covered = [len(v) for v in ids_per_gt.values() if v]
    return dict(TP=TP, FN=FN, FP=FP, LOC=LOC, AssA=AssA, AssRe=AssRe, AssPr=AssPr,
                idtp=idtp, n_gt=gcnt.sum(), n_pr=pcnt.sum(),
                tp=tp, fn=fn, fp=fp, ids=ids, frag=frag, mt=mt, ml=ml, dsum=dsum,
                n_traj=sum(1 for s in status.values() if len(s)),
                ids_per_gt_sum=sum(covered), n_covered=len(covered), n_pred_tracks=nP)


def combine(stats):
    """Combine per-sequence statistics into final metrics (percent where usual)."""
    S = lambda k: sum(s[k] for s in stats)
    TP, FN, FP = S("TP"), S("FN"), S("FP")
    w = lambda k: sum(s[k] * s["TP"] for s in stats) / np.maximum(TP, 1)
    AssA, AssRe, AssPr = w("AssA"), w("AssRe"), w("AssPr")
    DetA = TP / np.maximum(TP + FN + FP, 1)
    HOTA = np.sqrt(DetA * AssA)
    n_gt, n_pr, idtp = S("n_gt"), S("n_pr"), S("idtp")
    tp, fn, fp, ids = S("tp"), S("fn"), S("fp"), S("ids")
    gt_total = tp + fn
    return {
        "HOTA": 100 * HOTA.mean(), "DetA": 100 * DetA.mean(), "AssA": 100 * AssA.mean(),
        "DetRe": 100 * (TP / np.maximum(TP + FN, 1)).mean(),
        "DetPr": 100 * (TP / np.maximum(TP + FP, 1)).mean(),
        "AssRe": 100 * AssRe.mean(), "AssPr": 100 * AssPr.mean(),
        "LocA": 100 * (S("LOC") / np.maximum(TP, 1)).mean(),
        "IDF1": 100 * 2 * idtp / max(n_gt + n_pr, 1),
        "IDR": 100 * idtp / max(n_gt, 1), "IDP": 100 * idtp / max(n_pr, 1),
        "MOTA": 100 * (1 - (fn + fp + ids) / max(gt_total, 1)),
        "MOTP_m": S("dsum") / max(tp, 1),
        "IDS": int(ids), "FRAG": int(S("frag")),
        "MT": int(S("mt")), "ML": int(S("ml")), "GT_traj": int(S("n_traj")),
        "IDs/GT": S("ids_per_gt_sum") / max(S("n_covered"), 1),
        "Frag/GT": S("frag") / max(S("n_traj"), 1),
        "Pred_tracks": int(S("n_pred_tracks")),
        "FP": int(fp), "FN": int(fn),
    }


def evaluate(gt_scenes, pr_scenes, thr=DIST_THR, range_filter=True):
    return combine([evaluate_sequence(g, p, thr, range_filter) for g, p in zip(gt_scenes, pr_scenes)])
