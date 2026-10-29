"""Write a minimal nuScenes-format database plus a detection file from synthetic scenes.

This exercises run_nuscenes.py and the unmodified devkit TrackingEval end to end without the
real dataset: real val/train scene names, real table schema, synthetic content.

    python tests/make_fake_nuscenes.py /tmp/fake_nusc --val 150 --train 12
    python run_nuscenes.py eval --dataroot /tmp/fake_nusc --dets /tmp/fake_nusc/dets_val.json ...
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackfrag.official import yaw_to_quat  # noqa: E402
from trackfrag.synthetic import make_scene  # noqa: E402

CATEGORY = {"car": "vehicle.car", "truck": "vehicle.truck", "pedestrian": "human.pedestrian.adult",
            "bicycle": "vehicle.bicycle"}
tok = lambda: uuid.uuid4().hex


def build(root, splits):
    from nuscenes.utils.splits import create_splits_scenes
    names = create_splits_scenes()
    T = {k: [] for k in ("category", "attribute", "visibility", "instance", "sensor", "calibrated_sensor", "ego_pose",
                         "log", "scene", "sample", "sample_data", "sample_annotation", "map")}
    cat_tok = {}
    for c, full in CATEGORY.items():
        cat_tok[c] = tok()
        T["category"].append({"token": cat_tok[c], "name": full, "description": ""})
    T["visibility"].append({"token": "4", "level": "v80-100", "description": ""})
    sensor = {"token": tok(), "channel": "LIDAR_TOP", "modality": "lidar"}
    calib = {"token": tok(), "sensor_token": sensor["token"], "translation": [0, 0, 0], "rotation": [1, 0, 0, 0],
             "camera_intrinsic": []}
    T["sensor"].append(sensor); T["calibrated_sensor"].append(calib)
    log = {"token": tok(), "logfile": "fake", "vehicle": "fake", "date_captured": "2026-01-01", "location": "boston-seaport"}
    T["log"].append(log)
    (root / "maps").mkdir(parents=True, exist_ok=True)
    import matplotlib.image
    matplotlib.image.imsave(root / "maps" / "fake.png", np.zeros((4, 4)), cmap="gray")
    T["map"].append({"token": tok(), "log_tokens": [log["token"]], "category": "semantic_prior", "filename": "maps/fake.png"})
    dets = {}
    for split, seed0, n in splits:
        dets[split] = {}
        for i, name in enumerate(sorted(names[split])[:n]):
            det, gt = make_scene(seed0 + i)
            scene = {"token": tok(), "log_token": log["token"], "nbr_samples": len(det), "name": name, "description": ""}
            sample_toks = [tok() for _ in det]
            instances = {}
            last_ann = {}
            for k, (fd, fg) in enumerate(zip(det, gt)):
                ts = 1_600_000_000_000_000 + i * 100_000_000 + int(round(fd.t * 1e6))
                T["sample"].append({"token": sample_toks[k], "timestamp": ts, "scene_token": scene["token"],
                                    "prev": sample_toks[k - 1] if k else "", "next": sample_toks[k + 1] if k + 1 < len(det) else ""})
                pose = {"token": tok(), "timestamp": ts, "rotation": [1, 0, 0, 0], "translation": [*fd.ego, 0.0]}
                T["ego_pose"].append(pose)
                T["sample_data"].append({"token": tok(), "sample_token": sample_toks[k], "ego_pose_token": pose["token"],
                                         "calibrated_sensor_token": calib["token"], "timestamp": ts, "fileformat": "pcd",
                                         "is_key_frame": True, "height": 0, "width": 0, "filename": "", "prev": "", "next": ""})
                for j in range(len(fg)):
                    gid = int(fg.ids[j]); c = fg.cls[j]; b = fg.boxes[j]
                    if gid not in instances:
                        instances[gid] = {"token": tok(), "category_token": cat_tok[c], "nbr_annotations": 0,
                                          "first_annotation_token": "", "last_annotation_token": ""}
                    inst = instances[gid]
                    ann = {"token": tok(), "sample_token": sample_toks[k], "instance_token": inst["token"],
                           "visibility_token": "4", "attribute_tokens": [], "translation": [float(b[0]), float(b[1]), float(b[2])],
                           "size": [float(b[4]), float(b[3]), float(b[5])], "rotation": yaw_to_quat(b[6]),
                           "prev": last_ann.get(gid, ""), "next": "", "num_lidar_pts": 5, "num_radar_pts": 0}
                    inst["nbr_annotations"] += 1
                    inst["first_annotation_token"] = inst["first_annotation_token"] or ann["token"]
                    inst["last_annotation_token"] = ann["token"]
                    last_ann[gid] = ann["token"]
                    T["sample_annotation"].append(ann)
                dets[split][sample_toks[k]] = [{
                    "sample_token": sample_toks[k], "translation": [float(v) for v in fd.boxes[j, :3]],
                    "size": [float(fd.boxes[j, 4]), float(fd.boxes[j, 3]), float(fd.boxes[j, 5])],
                    "rotation": yaw_to_quat(fd.boxes[j, 6]), "velocity": [float(v) for v in fd.vel[j]],
                    "detection_name": str(fd.cls[j]), "detection_score": float(fd.score[j]), "attribute_name": ""}
                    for j in range(len(fd))]
            scene["first_sample_token"], scene["last_sample_token"] = sample_toks[0], sample_toks[-1]
            T["scene"].append(scene)
            T["instance"].extend(instances.values())
    # forward links of annotations
    by_tok = {a["token"]: a for a in T["sample_annotation"]}
    for a in T["sample_annotation"]:
        if a["prev"]:
            by_tok[a["prev"]]["next"] = a["token"]
    tables = root / "v1.0-trainval"
    tables.mkdir(parents=True, exist_ok=True)
    for k, v in T.items():
        (tables / f"{k}.json").write_text(json.dumps(v))
    meta = {"use_camera": True, "use_lidar": False, "use_radar": False, "use_map": False, "use_external": False}
    for split, d in dets.items():
        (root / f"dets_{split}.json").write_text(json.dumps({"meta": meta, "results": d}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--train", type=int, default=12)
    a = ap.parse_args()
    # same seeds as run_synthetic.py: val 10000.., train 0..
    build(Path(a.root), [("val", 10_000, a.val), ("train", 0, a.train)])
    print("wrote", a.root)
