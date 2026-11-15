"""Track A — batched 12-config sweep for 3D-anchored tracklet vote.

Runs all (w_geom, w_velo) configs in a single process. Loads pred_h5 once and
caches calibration. For each session we also cache:
  - per-frame 3D centroid per identity
  - per (cam, frame, identity) the 2D projection (computed once)
  - per (cam, slot, frame) the 2D tracklet centroid
…so the 12 configs only differ in the scoring step.

For each config, writes:
  outputs/PAF_3d_kalman/luc3d_l1_tracklet_3d_results/wg<wg>_wv<wv>/<session>.json

Progress is appended to outputs/PAF_3d_kalman/track_a_progress.log.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import orjson
    def _dumps(obj) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)
    _USE_ORJSON = True
except ImportError:
    def _dumps(obj) -> bytes:
        return json.dumps(obj).encode()
    _USE_ORJSON = False

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from inject_phantoms import parse_calibration_toml  # noqa: E402

BENCH_CAMERAS = ("back", "backL", "mid", "midL", "top", "topL")

W_GEOMS = (0.5, 1.0, 2.0, 5.0)
W_VELOS = (0.0, 0.5, 1.0)
SIGMA_GEOM = 50.0
SIGMA_VELO = 20.0
TRACKLET_GAP = 30

INPUT_DIR = REPO / "outputs" / "PAF_3d_kalman" / "luc3d_l1_only_results"
OUT_ROOT = REPO / "outputs" / "PAF_3d_kalman" / "luc3d_l1_tracklet_3d_results"
PRED_H5_DIR = REPO / "outputs" / "predictions_h5s"
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
PROGRESS_LOG = REPO / "outputs" / "PAF_3d_kalman" / "track_a_progress.log"


def log(msg: str):
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def project_points_batch(pts3d: np.ndarray, cam: dict) -> np.ndarray:
    """Project (N, 3) 3D points to (N, 2) 2D. NaN rows stay NaN."""
    import cv2
    out = np.full((pts3d.shape[0], 2), np.nan, dtype=np.float64)
    valid = ~np.isnan(pts3d).any(axis=1)
    if not valid.any():
        return out
    p3 = pts3d[valid].reshape(-1, 1, 3).astype(np.float64)
    p2, _ = cv2.projectPoints(p3, cam["rotation"], cam["translation"],
                               cam["matrix"], cam["distortions"])
    out[valid] = p2.reshape(-1, 2)
    return out


def session_3d_centroids(d: dict) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """Return (centroids 3D (n_frames, n_id, 3), frame list, id_idx map)."""
    frames = d["frames"]
    n = len(frames)
    # Collect all identities
    all_ids = set()
    for fr in frames:
        for t in fr.get("targets3d", []) or []:
            if t.get("identityId") is not None:
                all_ids.add(int(t["identityId"]))
    all_ids = sorted(all_ids)
    id_idx = {ident: i for i, ident in enumerate(all_ids)}
    centroids = np.full((n, len(all_ids), 3), np.nan, dtype=np.float64)
    frame_nums = []
    for fi, fr in enumerate(frames):
        frame_nums.append(int(fr["frame"]))
        for t in fr.get("targets3d", []) or []:
            ident = t.get("identityId")
            if ident is None:
                continue
            pts = t.get("points3d")
            if not pts:
                continue
            arr = np.array([(p if p is not None else [np.nan, np.nan, np.nan]) for p in pts],
                           dtype=np.float64)
            valid = ~np.isnan(arr).any(axis=1)
            if not valid.any():
                continue
            centroids[fi, id_idx[int(ident)]] = arr[valid].mean(axis=0)
    return centroids, frame_nums, id_idx


def tracklet_2d_centroids_for_cam(pred_tracks_session: np.ndarray) -> np.ndarray:
    """Compute (n_frames, n_slots, 2) of 2D centroids for a session's camera.

    pred_tracks_session: (n_frames, n_slots, n_nodes, 2)
    Returns NaN where there are no valid keypoints in that (frame, slot).
    """
    # Per (frame, slot), mean over non-NaN nodes.
    kp = pred_tracks_session
    valid = ~np.isnan(kp).any(axis=-1)   # (frames, slots, nodes)
    s = np.where(valid[..., None], kp, 0).sum(axis=2)   # (frames, slots, 2)
    c = valid.sum(axis=2)                                # (frames, slots)
    with np.errstate(invalid="ignore"):
        centroid = s / c[..., None]
    centroid[c == 0] = np.nan
    return centroid


def score_tracklet(trk: list[dict],
                   identity_proj_2d: dict[int, np.ndarray],  # id -> (n_frames, 2) projections for THIS cam
                   tracklet_2d: np.ndarray,                  # (n_frames, 2) for THIS (cam, slot)
                   frames_arr: np.ndarray,
                   tracklet_index_map: dict[int, int],
                   sigma2_geom: float,
                   sigma2_velo: float,
                   w_geom: float, w_velo: float) -> int:
    """Return the best identity id for this tracklet, given precomputed projections."""
    # Per-frame indices in the global frames array.
    n = len(trk)
    if n == 0:
        return None
    trk_frame_idx = np.array([tracklet_index_map[e["frame"]] for e in trk], dtype=np.int64)

    # 2D centroid of the tracklet at each step.
    trk_2d = tracklet_2d[trk_frame_idx]   # (n, 2)

    # 2D velocity of the tracklet (between successive frames in the tracklet).
    trk_frames = frames_arr[trk_frame_idx]
    if n > 1:
        dframes = np.diff(trk_frames).astype(np.float64)
        dframes[dframes == 0] = 1.0
        trk_vel = (trk_2d[1:] - trk_2d[:-1]) / dframes[:, None]
    else:
        trk_vel = np.empty((0, 2))

    vote_counts = Counter(e["id"] for e in trk)
    modal_id, _ = vote_counts.most_common(1)[0]

    best_id = modal_id
    best_score = -math.inf

    for ident, proj_full in identity_proj_2d.items():
        vc = vote_counts.get(ident, 0)
        proj = proj_full[trk_frame_idx]   # (n, 2)

        # Spatial fit: mean Euclidean distance where both are valid.
        both = ~(np.isnan(proj).any(axis=1) | np.isnan(trk_2d).any(axis=1))
        if both.any():
            diffs = proj[both] - trk_2d[both]
            d = np.sqrt((diffs * diffs).sum(axis=1))
            mean_d = float(d.mean())
            spatial_fit = math.exp(-(mean_d * mean_d) / sigma2_geom)
        else:
            spatial_fit = 0.0

        # Velocity match.
        if n >= 2:
            # ident's velocity between consecutive in-tracklet frames.
            id_vel = (proj[1:] - proj[:-1]) / dframes[:, None]
            valid_v = ~(np.isnan(id_vel).any(axis=1) | np.isnan(trk_vel).any(axis=1))
            if valid_v.any():
                vd = np.sqrt(((id_vel[valid_v] - trk_vel[valid_v]) ** 2).sum(axis=1))
                mean_vd = float(vd.mean())
                velocity_match = math.exp(-(mean_vd * mean_vd) / sigma2_velo)
            else:
                velocity_match = 0.0
        else:
            velocity_match = 0.0

        score = vc + w_geom * spatial_fit + w_velo * velocity_match
        if score > best_score:
            best_score = score
            best_id = ident

    return best_id


def _precompute_tracklet_metrics(
    trk: list[dict],
    identity_proj_2d: dict[int, np.ndarray],
    tracklet_2d_pf: np.ndarray,
    frame_to_arr_idx: dict[int, int],
) -> dict:
    """Return per-tracklet precomputed data: vote_counts, modal_id,
    per-identity (mean_d, mean_vd). These are config-independent."""
    n = len(trk)
    trk_frame_idx = np.array([frame_to_arr_idx[e["frame"]] for e in trk], dtype=np.int64)
    trk_2d = tracklet_2d_pf[trk_frame_idx]   # (n, 2)

    # Tracklet 2D velocity between successive frames in the tracklet.
    if n > 1:
        trk_frames_abs = np.array([e["frame"] for e in trk], dtype=np.float64)
        dframes = np.diff(trk_frames_abs)
        dframes[dframes == 0] = 1.0
        trk_vel = (trk_2d[1:] - trk_2d[:-1]) / dframes[:, None]
    else:
        dframes = None
        trk_vel = None

    vote_counts = Counter(e["id"] for e in trk)
    modal_id, _ = vote_counts.most_common(1)[0]

    per_id_metrics: dict[int, tuple[int, float | None, float | None]] = {}
    for ident, proj_full in identity_proj_2d.items():
        vc = vote_counts.get(ident, 0)
        proj = proj_full[trk_frame_idx]

        both = ~(np.isnan(proj).any(axis=1) | np.isnan(trk_2d).any(axis=1))
        if both.any():
            diffs = proj[both] - trk_2d[both]
            d = np.sqrt((diffs * diffs).sum(axis=1))
            mean_d = float(d.mean())
        else:
            mean_d = None

        if n >= 2 and trk_vel is not None:
            id_vel = (proj[1:] - proj[:-1]) / dframes[:, None]
            valid_v = ~(np.isnan(id_vel).any(axis=1) | np.isnan(trk_vel).any(axis=1))
            if valid_v.any():
                vd = np.sqrt(((id_vel[valid_v] - trk_vel[valid_v]) ** 2).sum(axis=1))
                mean_vd = float(vd.mean())
            else:
                mean_vd = None
        else:
            mean_vd = None

        per_id_metrics[ident] = (vc, mean_d, mean_vd)

    return {"modal_id": modal_id, "per_id_metrics": per_id_metrics}


def _pick_best_id(per_id_metrics: dict[int, tuple[int, float | None, float | None]],
                  modal_id: int,
                  sigma2_geom: float, sigma2_velo: float,
                  w_geom: float, w_velo: float) -> int:
    """O(n_identities) scoring using precomputed (mean_d, mean_vd)."""
    best_id = modal_id
    best_score = -math.inf
    for ident, (vc, mean_d, mean_vd) in per_id_metrics.items():
        sf = math.exp(-(mean_d * mean_d) / sigma2_geom) if mean_d is not None else 0.0
        vm = math.exp(-(mean_vd * mean_vd) / sigma2_velo) if mean_vd is not None else 0.0
        score = vc + w_geom * sf + w_velo * vm
        if score > best_score:
            best_score = score
            best_id = ident
    return best_id


def process_session_all_configs(json_path: Path,
                                sess_idx: int,
                                cam_by_name: dict[str, dict],
                                pred_tracks_by_cam_session: dict[str, np.ndarray],
                                out_dirs_by_cfg: dict[tuple[float, float], Path],
                                sigma2_geom: float, sigma2_velo: float,
                                configs: list[tuple[float, float]]) -> dict:
    raw = json_path.read_bytes()
    d = orjson.loads(raw) if _USE_ORJSON else json.loads(raw)

    # Per-session precomputation:
    centroids_3d, frame_nums, id_idx = session_3d_centroids(d)
    all_ids = sorted(id_idx, key=lambda x: id_idx[x])  # original identity ids in order
    frames_arr = np.array(frame_nums, dtype=np.int64)
    frame_to_arr_idx = {int(f): i for i, f in enumerate(frame_nums)}
    n_frames_orig = len(d["frames"])

    # For each camera: precompute identity projections and tracklet 2D centroids.
    cam_proj: dict[str, dict[int, np.ndarray]] = {}
    cam_trk_2d: dict[str, np.ndarray] = {}
    for cam_name, cam_calib in cam_by_name.items():
        if cam_name not in pred_tracks_by_cam_session:
            continue
        proj_per_id: dict[int, np.ndarray] = {}
        for ident, idx in id_idx.items():
            pts3d_seq = centroids_3d[:, idx, :]
            proj_per_id[ident] = project_points_batch(pts3d_seq, cam_calib)
        cam_proj[cam_name] = proj_per_id

        kp = pred_tracks_by_cam_session[cam_name]
        kp_centroid = tracklet_2d_centroids_for_cam(kp)
        cam_trk_2d[cam_name] = kp_centroid

    # Build timelines per (cam, slot).
    timelines: dict[str, list[dict]] = defaultdict(list)
    for pf_idx, fr in enumerate(d["frames"]):
        for slot_idx, (key, ident) in enumerate(fr["assignments"]):
            timelines[key].append({"frame": fr["frame"], "id": int(ident),
                                    "pf_idx": pf_idx, "slot_idx": slot_idx})
    for k in timelines:
        timelines[k].sort(key=lambda e: e["frame"])

    stats_by_cfg = {cfg: {"tracklets": 0, "changed_to_nonmodal": 0, "kept_modal": 0,
                          "frames_changed": 0}
                    for cfg in configs}

    # Pre-split tracklets per (cam, slot).
    tracklets_by_key: dict[str, list[list[dict]]] = {}
    for key, entries in timelines.items():
        tracklets = []
        cur = []
        for e in entries:
            if not cur or e["frame"] - cur[-1]["frame"] <= TRACKLET_GAP:
                cur.append(e)
            else:
                tracklets.append(cur); cur = [e]
        if cur:
            tracklets.append(cur)
        tracklets_by_key[key] = tracklets

    # Precompute per-(key, tracklet_idx) metrics: vote_counts, mean_d, mean_vd, modal_id.
    # tracklets_metrics[key] = list of dicts (one per tracklet).
    tracklets_metrics: dict[str, list[dict]] = {}
    for key, tracklets in tracklets_by_key.items():
        cam_name, slot_str = key.rsplit(":", 1)
        slot = int(slot_str)
        if cam_name not in cam_proj or cam_name not in cam_trk_2d:
            # No 3D evidence available — use modal only.
            tracklets_metrics[key] = []
            for trk in tracklets:
                ids = [e["id"] for e in trk]
                modal_id = Counter(ids).most_common(1)[0][0]
                tracklets_metrics[key].append({
                    "modal_id": modal_id,
                    "per_id_metrics": {modal_id: (len(trk), None, None)},
                })
            continue

        identity_proj_2d = cam_proj[cam_name]
        trk_2d_all = cam_trk_2d[cam_name][:, slot, :]
        tracklet_2d_pf = np.full((n_frames_orig, 2), np.nan, dtype=np.float64)
        n_kp_frames = trk_2d_all.shape[0]
        valid_frames_mask = frames_arr < n_kp_frames
        if valid_frames_mask.any():
            valid_idx = np.where(valid_frames_mask)[0]
            tracklet_2d_pf[valid_idx] = trk_2d_all[frames_arr[valid_idx]]

        per_trk_metrics = []
        for trk in tracklets:
            per_trk_metrics.append(
                _precompute_tracklet_metrics(trk, identity_proj_2d, tracklet_2d_pf,
                                              frame_to_arr_idx)
            )
        tracklets_metrics[key] = per_trk_metrics

    # For each config, compute best-id and serialize output.
    # We share the base dict d but swap out assignments per config.
    # Cache the serialized "static" prefix? Simplest: just write fresh d for each config.

    # Build the new_frames structure once with the original assignments; we'll mutate.
    # For performance: keep the original `d` intact, and per config rebuild only the
    # assignments lists.

    # Map from (key, trk_idx) → (best_id per config) so we can apply changes.
    # Actually, easier: per config, iterate tracklets and write changes into a per-pf list.

    for (w_geom, w_velo) in configs:
        cfg_stats = stats_by_cfg[(w_geom, w_velo)]

        # Start with a fresh copy of assignments per frame.
        new_assignments = [[list(a) for a in fr["assignments"]] for fr in d["frames"]]

        for key, tracklets in tracklets_by_key.items():
            per_trk_metrics = tracklets_metrics[key]
            for trk, m in zip(tracklets, per_trk_metrics):
                cfg_stats["tracklets"] += 1
                modal_id = m["modal_id"]
                pim = m["per_id_metrics"]
                if len(pim) == 1:
                    # Only modal known (no calibration available); use it as-is.
                    best_id = modal_id
                else:
                    best_id = _pick_best_id(pim, modal_id,
                                             sigma2_geom, sigma2_velo,
                                             w_geom, w_velo)
                if best_id == modal_id:
                    cfg_stats["kept_modal"] += 1
                else:
                    cfg_stats["changed_to_nonmodal"] += 1
                for e in trk:
                    if e["id"] != best_id:
                        new_assignments[e["pf_idx"]][e["slot_idx"]] = [key, best_id]
                        cfg_stats["frames_changed"] += 1

        # Write output JSON for this config.
        out_dir = out_dirs_by_cfg[(w_geom, w_velo)]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / json_path.name
        # Mutate frames in place (the orig d is no longer needed by other configs
        # — but assignments will be overwritten next iter).
        for i, fr in enumerate(d["frames"]):
            fr["assignments"] = new_assignments[i]
        d["tracklet3dStats"] = {"w_geom": w_geom, "w_velo": w_velo, **cfg_stats}
        out_path.write_bytes(_dumps(d))

    return stats_by_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip configs whose output dir already has all 74 JSONs")
    args = ap.parse_args()

    log(f"[batch_sweep] BEGIN input={args.input_dir} (n_configs={len(W_GEOMS)*len(W_VELOS)})")

    configs = [(wg, wv) for wg in W_GEOMS for wv in W_VELOS]
    out_dirs_by_cfg: dict[tuple[float, float], Path] = {}
    configs_to_run = []
    for (wg, wv) in configs:
        cfg_id = f"wg{wg}_wv{wv}"
        out_dir = OUT_ROOT / cfg_id
        out_dirs_by_cfg[(wg, wv)] = out_dir
        if args.skip_existing and out_dir.exists():
            existing = sorted(out_dir.glob("*.json"))
            if len(existing) >= 74:
                log(f"[batch_sweep] SKIP {cfg_id} (already has {len(existing)} JSONs)")
                continue
        configs_to_run.append((wg, wv))

    if not configs_to_run:
        log("[batch_sweep] nothing to do")
        return

    log(f"[batch_sweep] configs to run: {len(configs_to_run)}")

    # Load all pred_h5 once.
    log("[batch_sweep] loading pred_h5 ...")
    pred_h5_full: dict[str, np.ndarray] = {}
    for cam in BENCH_CAMERAS:
        p = PRED_H5_DIR / f"{cam}_predictions.h5"
        if not p.exists():
            log(f"  WARN: missing {p}")
            continue
        with h5py.File(p, "r") as f:
            pred_h5_full[cam] = f["tracks"][...]
        log(f"  loaded {cam}: shape={pred_h5_full[cam].shape}")

    master = pd.read_csv(MASTER, sep="\t", index_col=0)
    master["session"] = master["session"].astype(str)
    sess_to_row = {row["session"]: (idx, row) for idx, row in master.iterrows()}

    json_files = sorted(Path(args.input_dir).glob("*.json"))
    if args.limit:
        json_files = json_files[: args.limit]

    sigma2_geom = SIGMA_GEOM * SIGMA_GEOM
    sigma2_velo = SIGMA_VELO * SIGMA_VELO

    calib_cache: dict[str, dict[str, dict]] = {}

    total_stats = {cfg: {"tracklets": 0, "changed_to_nonmodal": 0, "kept_modal": 0,
                         "frames_changed": 0} for cfg in configs_to_run}

    for jp in tqdm(json_files, desc="sessions"):
        sess_id = jp.stem
        if sess_id not in sess_to_row:
            log(f"  skip {sess_id} (not in master sheet)")
            continue
        sess_idx, row = sess_to_row[sess_id]
        calib_path = str(row["calibration_toml"])
        if calib_path not in calib_cache:
            cams = parse_calibration_toml(Path(calib_path).read_text())
            calib_cache[calib_path] = {c["name"]: c for c in cams}
        cam_by_name = calib_cache[calib_path]

        pred_tracks_by_cam_session = {
            cam: pred_h5_full[cam][int(sess_idx)] for cam in pred_h5_full
        }

        s = process_session_all_configs(
            jp, int(sess_idx), cam_by_name,
            pred_tracks_by_cam_session, out_dirs_by_cfg,
            sigma2_geom, sigma2_velo, configs_to_run,
        )
        for cfg, st in s.items():
            for k, v in st.items():
                total_stats[cfg][k] += v

    # Summary log.
    log("[batch_sweep] DONE — per-config totals:")
    for cfg in configs_to_run:
        wg, wv = cfg
        st = total_stats[cfg]
        log(f"  wg={wg} wv={wv}: tracklets={st['tracklets']} "
            f"changed_to_nonmodal={st['changed_to_nonmodal']} "
            f"frames_changed={st['frames_changed']}")


if __name__ == "__main__":
    main()
