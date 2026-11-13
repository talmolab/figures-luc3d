"""Metric (millimetre) accuracy of each calibration, from the ChArUco board itself.

Reprojection error is in pixels and is blind to a global scale error. The board's
geometry is known exactly (square_length in spec.json, 24.0 mm), so every pair of
corners seen in the same frame has a true 3D distance. For each calibration this
script triangulates the board corners frame by frame and compares the measured
distances with the true ones:

  * adjacent corners (one square apart, 24.0 mm): error in mm, median |e|, signed
    median (a scale bias shows up here) and p95 |e|;
  * long spans (true distance >= 150 mm, up to the 259.6 mm board diagonal):
    relative error in %, which is where a scale error grows.

Same detections, loaders, camera-name mapping and aniposelib triangulation as
fig7_calib_score.py, and the same rule that a point which fails to triangulate
under ANY calibration is dropped for ALL of them, so every calibration is scored
on identical points. The board measures the calibration on the board, not the
accuracy of animal keypoints.

    python3 fig7_calib_board_mm.py <dataset_dir> <detections.pkl|session.json> -- <label>=<toml> ...

Writes <dataset_dir>/board_mm.json.
"""
import sys, os, json, pickle, base64, re

import numpy as np
from aniposelib.cameras import CameraGroup

d = sys.argv[1]
src = sys.argv[2]
cut = sys.argv.index('--')
CALIBS = [a.split('=', 1) for a in sys.argv[cut + 1:]]
spec = json.load(open(os.path.join(d, 'spec.json')))
NAMES = [c[0] for c in spec['cameras']]
B = spec['board']
NX, NY = B['board_x'] - 1, B['board_y'] - 1          # inner-corner grid
NPTS = NX * NY
SQ = float(B['square_length'])
# true planar position of every corner id (ChArUco ids run along x first)
GRID = np.array([[(i % NX) * SQ, (i // NX) * SQ] for i in range(NPTS)])
TRUE = np.linalg.norm(GRID[:, None] - GRID[None], axis=2)          # NPTS x NPTS
IU = np.triu_indices(NPTS, 1)
ADJ = np.isclose(TRUE[IU], SQ)
LONG = TRUE[IU] >= 150.0


def short(name):
    m = [r for r in NAMES if name == r or re.search(rf'(^|[-_ ]){re.escape(r)}([-_ .]|$)', name)]
    if len(m) != 1:
        raise SystemExit(f'camera name {name!r} maps to {m} of {NAMES}')
    return m[0]


def load_anipose(path):
    a = pickle.load(open(path, 'rb'))
    assert a['names'] == NAMES, (a['names'], NAMES)
    dets = {}
    for ci, rows in enumerate(a['rows']):
        for r in rows:
            ids = r['ids'].ravel().tolist()
            dets[(ci, int(r['framenum'][1]))] = dict(zip(ids, r['corners'].reshape(-1, 2)))
    return dets


def load_calibrat3(path):
    sess = json.load(open(path))
    vnames = sess['detections']['viewNames']
    order = [[short(v) for v in vnames].index(n) for n in NAMES]
    b64 = lambda s, dt: np.frombuffer(base64.b64decode(s), dtype=dt)
    dets = {}
    for fr in sess['detections']['frames']:
        for ci, vi in enumerate(order):
            v = fr['views'][vi]
            if v:
                ids = b64(v['ids'], np.int32).tolist()
                dets[(ci, int(fr['frame']))] = dict(zip(ids, b64(v['corners'], np.float32).reshape(-1, 2)))
    return dets


def load_group(path):
    cg = CameraGroup.load(path)
    cams = {}
    for c in cg.cameras:
        s = short(c.get_name()); c.set_name(s); cams[s] = c
    cg.cameras = [cams[r] for r in NAMES]
    return cg


dets = load_anipose(src) if src.endswith('.pkl') else load_calibrat3(src)
frames = sorted({f for _, f in dets})
fidx = {f: i for i, f in enumerate(frames)}
imgp = np.full((len(NAMES), len(frames), NPTS, 2), np.nan)
for (ci, f), pts in dets.items():
    for i, p in pts.items():
        imgp[ci, fidx[f], i] = p
flat = imgp.reshape(len(NAMES), -1, 2)
seen2 = np.sum(~np.isnan(flat[..., 0]), axis=0) >= 2
print(f'{len(frames)} frames, {int(seen2.sum())} corner observations seen by >= 2 cameras')

P3, keep = {}, seen2.copy()
for label, path in CALIBS:
    cg = load_group(path)
    p = np.full((flat.shape[1], 3), np.nan)
    p[seen2] = cg.triangulate(flat[:, seen2])
    err = np.full((len(NAMES), flat.shape[1]), np.nan)
    err[:, seen2] = np.linalg.norm(cg.reprojection_error(p[seen2], flat[:, seen2], mean=False), axis=2)
    worst = np.nanmax(np.where(np.isfinite(err), err, -np.inf), axis=0)
    keep &= np.isfinite(p).all(axis=1) & (worst < 1000)
    P3[label] = p
print(f'common valid points: {int(keep.sum())} / {int(seen2.sum())}')

out = dict(dataset=os.path.basename(os.path.abspath(d)), detections=os.path.basename(src),
           square_mm=SQ, board=B, frames=len(frames), points=int(keep.sum()), calibrations={})
for label, p in P3.items():
    p = np.where(keep[:, None], p, np.nan).reshape(len(frames), NPTS, 3)
    adj, rel = [], []
    for f in range(len(frames)):
        q = p[f]
        ok = np.isfinite(q[:, 0])
        if ok.sum() < 2:
            continue
        dist = np.linalg.norm(q[:, None] - q[None], axis=2)[IU]
        both = ok[IU[0]] & ok[IU[1]]
        adj.append(dist[both & ADJ] - SQ)
        rel.append(100.0 * (dist[both & LONG] - TRUE[IU][both & LONG]) / TRUE[IU][both & LONG])
    adj, rel = np.concatenate(adj), np.concatenate(rel)
    r = dict(adjacent_n=int(adj.size),
             adjacent_median_abs_mm=float(np.median(np.abs(adj))),
             adjacent_signed_median_mm=float(np.median(adj)),
             adjacent_p95_abs_mm=float(np.percentile(np.abs(adj), 95)),
             long_n=int(rel.size),
             long_median_abs_pct=float(np.median(np.abs(rel))),
             long_signed_median_pct=float(np.median(rel)),
             long_p95_abs_pct=float(np.percentile(np.abs(rel), 95)))
    out['calibrations'][label] = r
    print(f'{label:28s} adjacent (24 mm): median |e| {r["adjacent_median_abs_mm"]:.3f} mm, '
          f'signed {r["adjacent_signed_median_mm"]:+.3f} mm, p95 {r["adjacent_p95_abs_mm"]:.3f} mm (n={r["adjacent_n"]}) | '
          f'spans >=150 mm: median |e| {r["long_median_abs_pct"]:.3f} %, signed {r["long_signed_median_pct"]:+.3f} %, '
          f'p95 {r["long_p95_abs_pct"]:.3f} % (n={r["long_n"]})', flush=True)

json.dump(out, open(os.path.join(d, 'board_mm.json'), 'w'), indent=1)
print(f'wrote {d}/board_mm.json')
