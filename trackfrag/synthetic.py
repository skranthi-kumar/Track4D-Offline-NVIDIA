"""Synthetic driving scenes with a camera-like 3D detector, for development without dataset access.

It reproduces the failure modes that fragment camera-only tracks:
  * depth error that grows with range, lies along the viewing ray and is correlated over time
  * occlusion by nearer objects, computed from the ego viewpoint
  * temporally correlated missed detections (bursts) and fluctuating confidence
  * noisy predicted velocities, 180-degree heading flips, car/truck confusion
  * duplicate boxes displaced along the camera ray, clutter and short-lived ghost objects

This is a proxy, not a substitute for nuScenes: final claims must come from real detections.
"""
import numpy as np

from .core import CLASS_RANGE, Frame

CLASSES = {  # size (l, w, h), speed range m/s, P(static), base detection quality, spawn weight
    "car":        dict(size=(4.6, 1.9, 1.7), speed=(3, 13), p_static=0.4, q=2.6, w=0.55),
    "truck":      dict(size=(8.0, 2.6, 3.2), speed=(3, 11), p_static=0.3, q=2.0, w=0.10),
    "pedestrian": dict(size=(0.7, 0.7, 1.75), speed=(0.6, 1.8), p_static=0.2, q=1.8, w=0.28),
    "bicycle":    dict(size=(1.8, 0.6, 1.3), speed=(2, 6), p_static=0.2, q=1.3, w=0.07),
}
NAMES = list(CLASSES)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_scene(seed, n_frames=40, dt=0.5, difficulty=1.0):
    """Returns (detections, ground_truth): two lists of Frame. ``difficulty`` scales noise and misses."""
    rng = np.random.default_rng(seed)
    T = n_frames
    # ---------------- ego trajectory
    ego = np.zeros((T, 2)); ev = rng.uniform(0, 10); eh = 0.0; ew = rng.normal(0, 0.03)
    for t in range(1, T):
        eh += ew * dt
        ego[t] = ego[t - 1] + ev * dt * np.array([np.cos(eh), np.sin(eh)])
    # ---------------- objects
    n_obj = rng.integers(20, 45)
    objs = []
    span = ego[-1, 0] - ego[0, 0]
    for i in range(n_obj):
        c = NAMES[rng.choice(len(NAMES), p=[CLASSES[n]["w"] for n in NAMES])]
        p = CLASSES[c]
        x = rng.uniform(-40, span + 50); y = rng.uniform(-30, 30)
        if c in ("car", "truck"):
            y = rng.choice([-7.0, -3.5, 3.5, 7.0, -12, 12]) + rng.normal(0, 0.3)
            h0 = 0.0 if y < 0 else np.pi
        else:
            h0 = rng.uniform(-np.pi, np.pi)
        static = rng.random() < p["p_static"]
        sp = 0.0 if static else rng.uniform(*p["speed"])
        size = np.array(p["size"]) * rng.uniform(0.85, 1.15, 3)
        pos = np.zeros((T, 2)); head = np.zeros(T); vel = np.zeros((T, 2))
        pos[0] = (x, y); hd = h0; yr = rng.normal(0, 0.05) if not static else 0.0
        stop_go = rng.random() < 0.25
        for t in range(T):
            if t:
                if c == "pedestrian":
                    hd += rng.normal(0, 0.15)
                else:
                    hd += yr * dt
                if stop_go and not static:
                    sp = np.clip(sp + rng.normal(0, 1.5) * dt, 0, p["speed"][1])
                v = sp * np.array([np.cos(hd), np.sin(hd)])
                pos[t] = pos[t - 1] + v * dt
            vel[t] = sp * np.array([np.cos(hd), np.sin(hd)])
            head[t] = hd
        objs.append(dict(cls=c, size=size, pos=pos, head=head, vel=vel, z=size[2] / 2 + rng.normal(0, 0.1)))

    # ---------------- per-frame rendering
    det_frames, gt_frames = [], []
    st = {i: dict(eb=0.0, el=0.0, q=0.0, fail=False) for i in range(len(objs))}
    ghosts = []
    for t in range(T):
        e = ego[t]
        rel = np.array([o["pos"][t] - e for o in objs])
        r = np.linalg.norm(rel, axis=1)
        bearing = np.arctan2(rel[:, 1], rel[:, 0])
        half = np.array([0.5 * np.hypot(*o["size"][:2]) * 0.8 for o in objs]) / np.maximum(r, 0.5)
        # occlusion: fraction of each object's angular extent covered by nearer objects
        occ = np.zeros(len(objs))
        for i in range(len(objs)):
            if r[i] > 60:
                continue
            samples = bearing[i] + np.linspace(-half[i], half[i], 9)
            near = np.flatnonzero((r < r[i] - 1.0) & (r < 60))
            if len(near):
                diff = np.abs((samples[:, None] - bearing[None, near] + np.pi) % (2 * np.pi) - np.pi)
                occ[i] = np.mean(np.any(diff < half[None, near], axis=1))
        gb, gv, gc, gid = [], [], [], []
        db, dv, ds, dc = [], [], [], []
        for i, o in enumerate(objs):
            c = o["cls"]
            if r[i] <= CLASS_RANGE[c] and occ[i] < 0.9:
                gb.append(np.r_[o["pos"][t], o["z"], o["size"], o["head"][t]]); gv.append(o["vel"][t])
                gc.append(c); gid.append(i)
            if r[i] > 55 or r[i] < 1.0:
                continue
            s = st[i]
            # temporally-correlated detector state
            s["fail"] = rng.random() < (0.7 if s["fail"] else 0.04 * difficulty)
            s["q"] = 0.7 * s["q"] + np.sqrt(1 - 0.49) * rng.normal(0, 0.6)
            q = CLASSES[c]["q"] - 0.025 * r[i] - 2.2 * occ[i] * difficulty - 3 * (occ[i] > 0.85) + s["q"]
            if s["fail"]:
                q -= 3.0
            score = _sigmoid(q - 1.0 + rng.normal(0, 0.4))
            u = rel[i] / max(r[i], 1e-3); n = np.array([-u[1], u[0]])
            sd_d = (0.12 + 0.035 * r[i]) * difficulty
            sd_l = (0.06 + 0.005 * r[i]) * difficulty
            s["eb"] = 0.6 * s["eb"] + 0.8 * rng.normal(0, sd_d)
            s["el"] = 0.5 * s["el"] + 0.85 * rng.normal(0, sd_l)
            xy = o["pos"][t] + u * s["eb"] + n * s["el"]
            v = o["vel"][t] + rng.normal(0, (0.4 + 0.02 * r[i]) * difficulty, 2)
            if rng.random() < 0.05:
                v += rng.normal(0, 3.0, 2)
            yaw = o["head"][t] + rng.normal(0, 0.1) + (np.pi if rng.random() < 0.06 else 0)
            size = o["size"] * rng.normal(1, 0.05, 3)
            cc = c
            if c in ("car", "truck") and rng.random() < 0.03:
                cc = "truck" if c == "car" else "car"
            if score >= 0.05:
                db.append(np.r_[xy, o["z"] + rng.normal(0, 0.15), size, yaw]); dv.append(v); ds.append(score); dc.append(cc)
                if rng.random() < 0.08 * difficulty:  # duplicate along the camera ray
                    xy2 = xy + u * rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
                    db.append(np.r_[xy2, o["z"], size, yaw]); dv.append(v); ds.append(score * rng.uniform(0.3, 0.7)); dc.append(cc)
        # clutter
        for _ in range(rng.poisson(6 * difficulty)):
            ang = rng.uniform(-np.pi, np.pi); rr = rng.uniform(5, 55)
            c = NAMES[rng.choice(len(NAMES), p=[CLASSES[n]["w"] for n in NAMES])]
            db.append(np.r_[e + rr * np.array([np.cos(ang), np.sin(ang)]), 0.8, CLASSES[c]["size"], rng.uniform(-np.pi, np.pi)])
            dv.append(rng.normal(0, 1, 2)); ds.append(rng.beta(1.2, 6)); dc.append(c)
        # short-lived ghosts (persistent false positives)
        if rng.random() < 0.3 * difficulty:
            ang = rng.uniform(-np.pi, np.pi); rr = rng.uniform(8, 45)
            ghosts.append(dict(xy=e + rr * np.array([np.cos(ang), np.sin(ang)]), left=rng.integers(2, 6),
                               cls=NAMES[rng.integers(len(NAMES))], s=rng.uniform(0.25, 0.55)))
        for g in ghosts:
            if g["left"] > 0:
                db.append(np.r_[g["xy"] + rng.normal(0, 0.4, 2), 0.8, CLASSES[g["cls"]]["size"], 0.0])
                dv.append(rng.normal(0, 0.5, 2)); ds.append(g["s"] * rng.uniform(0.7, 1.1)); dc.append(g["cls"])
                g["left"] -= 1
        to_arr = lambda a, d: np.array(a) if a else np.zeros((0, d))
        det_frames.append(Frame(t * dt, to_arr(db, 7), to_arr(dv, 2), np.array(ds), np.array(dc, dtype=object), e.copy()))
        gt_frames.append(Frame(t * dt, to_arr(gb, 7), to_arr(gv, 2), np.ones(len(gb)), np.array(gc, dtype=object),
                               e.copy(), np.array(gid, dtype=np.int64)))
    return det_frames, gt_frames


def make_split(seeds, **kw):
    dets, gts = zip(*(make_scene(s, **kw) for s in seeds))
    return list(dets), list(gts)
