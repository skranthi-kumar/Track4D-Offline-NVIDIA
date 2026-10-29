"""Shared data types, Kalman filter / RTS smoother and per-class constants."""
from dataclasses import dataclass, field

import numpy as np

# Classes evaluated by the nuScenes tracking benchmark, plus their evaluation ranges (m).
CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50,
               "pedestrian": 40, "motorcycle": 40, "bicycle": 40}
# Planar association gates (m), as used by CenterPoint / Foundation4D-style trackers.
CLASS_GATE = {"car": 4.0, "truck": 4.0, "bus": 5.5, "trailer": 3.0,
              "pedestrian": 1.5, "motorcycle": 4.0, "bicycle": 3.0}
# White-noise acceleration std (m/s^2) of the constant-velocity model.
CLASS_ACCEL = {"car": 3.0, "truck": 2.0, "bus": 2.0, "trailer": 1.5,
               "pedestrian": 1.5, "motorcycle": 3.5, "bicycle": 2.5}


@dataclass
class Frame:
    """Boxes of one timestamp in a scene (detections, tracker output or ground truth).

    All coordinates are in a common world frame. ``ids`` is None for raw detections.
    """
    t: float                      # seconds
    boxes: np.ndarray             # (N, 7) x, y, z, l, w, h, yaw
    vel: np.ndarray               # (N, 2) vx, vy
    score: np.ndarray             # (N,)
    cls: np.ndarray               # (N,) class names
    ego: np.ndarray = None        # (2,) ego xy; enables the depth-aware noise model
    ids: np.ndarray = None        # (N,) track / ground-truth ids
    token: str = None             # e.g. nuScenes sample token

    def __len__(self):
        return len(self.score)

    @staticmethod
    def empty(t, ego=None, token=None, with_ids=False):
        return Frame(t, np.zeros((0, 7)), np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=object),
                     ego, np.zeros(0, dtype=np.int64) if with_ids else None, token)


@dataclass
class NoiseModel:
    """Measurement noise of a camera-only detector.

    Camera depth error grows with range and lies along the viewing ray, so the position
    covariance is anisotropic: ``sd_depth`` along the ego->object ray, ``sd_lat`` across it.
    With ``depth_aware=False`` an isotropic ``iso_sd`` is used (the usual planar-distance assumption).
    With ``use_vel=False`` detector velocities are ignored (single-frame detectors such as MonoDIS
    report none) and the filter estimates motion from positions alone.
    """
    depth_sd0: float = 0.25
    depth_sd_k: float = 0.03
    lat_sd0: float = 0.10
    lat_sd_k: float = 0.006
    vel_sd: float = 1.0
    iso_sd: float = 1.0
    depth_aware: bool = True
    use_vel: bool = True

    def pos_cov(self, xy, ego):
        """(N, 2, 2) position covariances for detections at ``xy`` (N, 2)."""
        n = len(xy)
        if not self.depth_aware or ego is None:
            return np.tile(np.eye(2) * self.iso_sd ** 2, (n, 1, 1))
        d = xy - ego[None, :2]
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-3)
        u = d / r[:, None]
        sd_d = self.depth_sd0 + self.depth_sd_k * r
        sd_l = self.lat_sd0 + self.lat_sd_k * r
        uu = u[:, :, None] * u[:, None, :]
        return sd_d[:, None, None] ** 2 * uu + sd_l[:, None, None] ** 2 * (np.eye(2)[None] - uu)

    def meas_cov(self, xy, ego):
        """(N, 4, 4) covariances of [x, y, vx, vy] measurements."""
        R = np.zeros((len(xy), 4, 4))
        R[:, :2, :2] = self.pos_cov(xy, ego)
        R[:, 2, 2] = R[:, 3, 3] = self.vel_sd ** 2 if self.use_vel else NO_VEL_VAR
        return R


# Velocity variance of a measurement that carries no velocity, and the prior std (m/s) a new
# track starts with. The prior is wider than any real velocity noise, so with detector velocities
# it never binds and the filter is unchanged.
NO_VEL_VAR = 1e8
VEL_PRIOR_SD = 10.0


def init_cov(R):
    """Initial state covariance of a track started from a measurement with covariance R."""
    P = R.copy()
    P[2, 2] = min(P[2, 2], VEL_PRIOR_SD ** 2)
    P[3, 3] = min(P[3, 3], VEL_PRIOR_SD ** 2)
    return P


def cv_model(dt, accel_sd):
    """Constant-velocity transition and process noise for state [x, y, vx, vy]."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
    return F, accel_sd ** 2 * G @ G.T


def kf_update(x, P, z, R):
    S = P + R
    K = P @ np.linalg.inv(S)
    return x + K @ (z - x), (np.eye(4) - K) @ P


@dataclass
class Meas:
    """One detection assigned to a track."""
    k: int                # frame index in the scene
    z: np.ndarray         # [x, y, vx, vy]
    R: np.ndarray         # (4, 4)
    box: np.ndarray       # (7,) full box as detected
    score: float


@dataclass
class Tracklet:
    cls: str
    meas: list = field(default_factory=list)

    @property
    def k0(self):
        return self.meas[0].k

    @property
    def k1(self):
        return self.meas[-1].k

    def mean_score(self):
        return float(np.mean([m.score for m in self.meas]))


def rts_smooth(meas, times, accel_sd):
    """Kalman filter + Rauch-Tung-Striebel smoother over every frame from the first to the last
    measurement. Frames without a measurement are pure predictions, so this also interpolates gaps.

    Returns (ks, xs, Ps): frame indices, smoothed states (n, 4), covariances (n, 4, 4).
    """
    by_k = {m.k: m for m in meas}
    ks = np.arange(meas[0].k, meas[-1].k + 1)
    n = len(ks)
    xf = np.zeros((n, 4)); Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    Fs = [None] * n
    x, P = meas[0].z.copy(), init_cov(meas[0].R)
    for i, k in enumerate(ks):
        if i > 0:
            F, Q = cv_model(times[k] - times[ks[i - 1]], accel_sd)
            x, P = F @ x, F @ P @ F.T + Q
            Fs[i] = F
        xp[i], Pp[i] = x, P
        if k in by_k and i > 0:
            x, P = kf_update(x, P, by_k[k].z, by_k[k].R)
        xf[i], Pf[i] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for i in range(n - 2, -1, -1):
        C = Pf[i] @ Fs[i + 1].T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    return ks, xs, Ps


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
