#!/usr/bin/env python
"""Supplementary Movie 3: every SLAP-2M (animals x obstacle_rating) condition.

A ~2 min tour of the 14 (animals, obstacle_rating) combinations that exist in
slap_2m/master_sheet.xlsx, rendered in enrichment_scene.py's cage aesthetic
(itself cage_scene.py's). Each combination plays that combo's representative
session (enrichment_scene.COMBOS) from its representative frame onward with the
REAL tracked 3D poses (aligned_points3d.h5) at SPEED x real time while the camera
orbits the cage. Sessions are 30 fps except the 13 recorded on 2022-10-19 (60 fps);
step_for() picks the data step per session so every segment plays at the same speed.

Reuse, not copies: the cage, animals, lights, Cycles setup and wall-opening
logic come from cage_scene.py; the session table, the enrichment-object
geometry, greedy placement and framing come from enrichment_scene.py.

HONESTY RULES baked into the composite:
  * Enrichment objects (rating > 0) are SYNTHETIC placements -- no session
    recorded object positions. v1/v2 captioned every enriched 3D segment with
    "Enrichment objects illustrative (positions not recorded)"; v3 drops that
    caption (OBJECT_CAVEAT = "") and the paper carries the caveat. Placement here
    clears the animals' trajectory over the WHOLE segment (not one frame as in
    the stills), so tile and movie arrangements can differ in position.
  * Cut-ins to real video only for the four rating-0 sessions (empty arena, so
    3D and camera agree). The Blender camera flies onto the real camera's pose
    (calibration.toml extrinsics/intrinsics, mapped into the aligned cage frame
    with alignment.toml exactly as cage_scene.load_session does), then cross-
    fades into that camera's video for the SAME data frames. The pinhole render
    cannot reproduce the lens's radial distortion, so the VIDEO is undistorted
    with the session's own K + distortion (cv2) and contrast-stretched for
    display (the IR video is very dark); both are stated on screen.
  * Tail keypoints that triangulated outside the cage walls are clamped inside
    (cage_scene.clamp_tails), as in the stills -- visualization only.

v2 (2026-10-03, after the author's review of the 1:55 v1 cut):
  * PACING: data at 2x (DATA_STEP), plain combos 3.5 s, cut-ins 6 s, cards
    2-3 s, 6-frame dissolves -> ~62 s. Lower thirds appear immediately.
  * FRAME PICKING (`pick` -> renders/movie3/frame_picks.json, read by
    combo_info): v1 used enrichment_scene.COMBOS' single-still frames, and in
    the enriched sessions the mice spend most of the session standing ON the
    real cubes/dome (lowest dorsal body keypoint 50-150 mm above the floor,
    95th pct 150-300 mm; rating-0 sessions: median ~36 mm, 99th pct <= 60 mm),
    which are not in the scene -- the "floating mouse". `pick` scans each
    session for a window of the segment's full data span where (i) every
    animal's lowest body keypoint stays <= FLOOR_MAX_MM above the rendered
    floor and (ii) after greedy placement every keypoint keeps >= CLEAR_MIN_MM
    xy clearance from every object footprint (cube half-diagonal / dome ring);
    it falls back to the other master_sheet sessions of the same condition.
    (2,4) and (2,5) moved to other sessions; (3,4) has one session and no
    window clears 20 mm -- its best (9 mm, conservative) window is used and
    flagged in frame_picks.json.
  * BLACK MEMBRANES: v1 frames in a2_o0 (frames 262-296) and a4_o0 (200-201,
    267-291) rendered one or two tracks' translucent body membranes solid
    black (max channel < 12) for runs of frames. Ruled OUT by re-rendering the
    identical frames: tail clamping / pose geometry (same trx -> clean),
    transparent-bounce exhaustion (128 vs 1024 identical), denoiser / sample
    count (off / 128 identical), persistent data on vs off, several scenes per
    process (3 scenes then a4_o0 at full res: clean). The identical frame is
    clean in a fresh process every time, and the fault only appeared in v1's
    long-running workers (3-4 scenes, ~1,000 frames each, two workers per GPU),
    always on a whole material's faces -- i.e. a stale/corrupt render-session
    state, not a scene or setting. Fix: `all` now runs ONE PROCESS PER COMBO,
    and `qc` scans every output frame for black pixels (`black_px`, threshold
    50) and re-renders offenders in a fresh process. Settings are unchanged.

Stages (each resumable; frames that exist are skipped):

  P=bpyenv/bin/python
  $P slap2m_movie.py pick   [--combo 1 0]          # frame_picks.json (floor + clearance)
  $P slap2m_movie.py check  --combo 1 0            # alignment overlay still(s)
  $P slap2m_movie.py render --combo 1 0 [--res 960 540 --samples 24]
  $P slap2m_movie.py video  --combo 1 0            # decode+undistort real frames
  $P slap2m_movie.py compose [--res 960 540] [--only 1 0] [--out x.mp4]
  $P slap2m_movie.py all --part 0/2                # render a share of all combos
  $P slap2m_movie.py qc [--fix]                    # black-frame scan (+ re-render)

Outputs under renders/movie3/: frames_<VERSION>_<W>x<H>/a<N>_o<R>/f####.png
(Blender), video_<VERSION>/a<N>_o0/v####.png (real camera, undistorted),
check/ (overlay stills), supp_movie3_slap2m_<VERSION>.mp4 (final; v1 files
have no version tag and are kept).

ENVIRONMENT NOTE (2026-10-03): this box lacked the X/GL shared libraries
bpy links against (libXrender, libXfixes, libXi, libxkbcommon, libSM, libICE,
libGL). They were supplied from a conda-forge env via LD_LIBRARY_PATH, e.g.
  conda create -p ./xlibs -c conda-forge xorg-libxrender xorg-libxfixes \\
      xorg-libxi libxkbcommon xorg-libsm xorg-libice libgl
  (symlink just those .so files into one dir and point LD_LIBRARY_PATH at it)
ffmpeg: /root/vast/eric/luc3d-bench/video_env/bin/ffmpeg.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUTROOT = os.path.join(HERE, "renders", "movie3")
FFMPEG = "/root/vast/eric/luc3d-bench/video_env/bin/ffmpeg"
FONTS = os.path.join(HERE, "..", "src", "fonts")
GRID = os.path.join(HERE, "renders", "enrichment_grid.png")
SLAP2M = "/root/talmolab-smb/eric/slap_2m"
FPS = 30
OUT_W, OUT_H = 1920, 1080

# combos in storyboard order: animals, then rating
ORDER = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
         (2, 0), (2, 2), (2, 3), (2, 4), (2, 5),
         (3, 0), (3, 4), (4, 0)]
#: real camera used for each rating-0 cut-in. 'mid' looks in over the open
#: top of the cage at ~67 deg elevation, so no acrylic wall sits between it and
#: the animals and the whole floor is in view (checked on all four sessions).
CUTIN_CAM = {(1, 0): "mid", (2, 0): "mid", (3, 0): "mid", (4, 0): "mid"}
#: sessions per (animals, rating) in master_sheet.xlsx (74 rows, read 2026-10-03)
N_SESSIONS = {(1, 0): 12, (1, 1): 3, (1, 2): 6, (1, 3): 3, (1, 4): 7, (1, 5): 1,
              (2, 0): 10, (2, 2): 7, (2, 3): 3, (2, 4): 3, (2, 5): 12,
              (3, 0): 3, (3, 4): 1, (4, 0): 3}

# ---- segment timing (output frames at 30 fps) ------------------------------
#: v2 pacing (the v1 cut was 1:55; the author asked for ~60-75 s): the data
#: plays at 2x (DATA_STEP data frames per output frame), plain combos run 3.5 s,
#: cut-ins 6 s, the cards 2-3 s, dissolves 6 frames.
DATA_STEP = 2                                   # step for a 30 fps session (= SPEED)
SPEED = 2                                       # playback speed vs real time, every segment
SLAP2M_60FPS_DATES = ("2022-10-19",)


def step_for(session):
    """Data frames per output frame for a session path, so it plays at SPEED x."""
    fps = 60 if any(d in session for d in SLAP2M_60FPS_DATES) else 30
    return SPEED * fps // FPS
PLAIN_N = 105                                   # 3.5 s orbit
ORBIT_A, FLY, FADE, HOLD, ORBIT_B = 45, 24, 9, 42, 27    # cut-in segment, 6.0 s
CUT_N = ORBIT_A + FLY + FADE + HOLD + FADE + FLY + ORBIT_B
TITLE_N, GRID_N, END_N = 2 * 30, 3 * 30, 2 * 30
PLAIN_AZ, PLAIN_SWEEP, ORBIT_EL = 205.0, 50.0, 22.0       # tiles look from az 205
ORBIT_DIST = 4.0                                # m; near-orthographic (cage_scene's dist)
MARGIN = 1.45                                   # cage-fit margin in the 16:9 frame
LIFT = 0.04                                     # aim drop (x ortho): cage clears the lower third
XFADE = 6                                       # frames, dissolve between segments
PICKS = os.path.join(OUTROOT, "frame_picks.json")   # merged view, written by `pick`
PICK_DIR = os.path.join(OUTROOT, "picks")          # one json per combo (parallel-safe)
VERSION = "v4"                                  # frames/video/mp4 of this cut; v1 files are kept
#: v4 (2026-11-12): every segment plays at SPEED x real time. The 2022-10-19 SLAP-2M
#: sessions were acquired at 60 fps (master sheet frames/duration and the mp4s agree),
#: the rest at 30 fps; v2/v3 used DATA_STEP = 2 everywhere, so those four segments
#: played at 1x while the lower third said 2x. step_for() makes the step per session.
# Caption drawn on every enriched 3D segment. v1/v2 carried "Enrichment objects
# illustrative (positions not recorded)"; dropped for v3 at the author's request
# (2026-10-04). The object meshes are the author's; placement is still
# place_objects() (see the HONESTY RULES above), and the paper's Fig 4A caption and
# Video 3 legend carry the caveat instead of the video.
OBJECT_CAVEAT = ""


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def seg_len(combo):
    return CUT_N if combo in CUTIN_CAM else PLAIN_N


def schedule(combo, i):
    """What output frame i of a combo shows: (kind, param, video_alpha).
    kind 'orbit' -> param = azimuth offset (deg) from the segment's anchor az;
    kind 'fly'   -> param = s in [0,1], 0 = orbit pose, 1 = real camera;
    video_alpha is the cross-fade weight of the real video (0 outside cut-ins).
    The data frame is always rep_frame + DATA_STEP * i."""
    if combo not in CUTIN_CAM:
        return "orbit", -PLAIN_SWEEP / 2 + PLAIN_SWEEP * i / (PLAIN_N - 1), 0.0
    # cut-in: orbit toward the camera's side (az offsets relative to camera az)
    t = i
    if t < ORBIT_A:
        return "orbit", 75.0 - 50.0 * t / ORBIT_A, 0.0
    t -= ORBIT_A
    if t < FLY:
        return "fly", smooth((t + 1) / FLY), 0.0
    t -= FLY
    if t < FADE:
        return "fly", 1.0, smooth((t + 1) / (FADE + 1))
    t -= FADE
    if t < HOLD:
        return "fly", 1.0, 1.0
    t -= HOLD
    if t < FADE:
        return "fly", 1.0, 1.0 - smooth((t + 1) / (FADE + 1))
    t -= FADE
    if t < FLY:
        return "fly", 1.0 - smooth((t + 1) / FLY), 0.0
    t -= FLY
    return "orbit", 25.0 - 30.0 * t / max(1, ORBIT_B - 1), 0.0


def data_span(combo):
    """Number of DATA frames a combo's segment consumes."""
    return step_for(combo_info(combo)[0]) * seg_len(combo)


def combo_info(combo):
    """(session_path, first data frame). enrichment_scene.COMBOS by default;
    overridden per combo by renders/movie3/frame_picks.json when `pick` has
    found a cleaner window (see cmd_pick)."""
    from enrichment_scene import COMBOS          # noqa: imports bpy; fine in bpyenv
    sp, fr = COMBOS[combo]
    pf = os.path.join(PICK_DIR, f"a{combo[0]}_o{combo[1]}.json")
    if os.path.exists(pf):
        pk = json.load(open(pf))
        sp, fr = pk["session"], pk["frame"]
    return sp, fr


def merged_picks():
    out = {}
    for f in sorted(glob.glob(os.path.join(PICK_DIR, "a*_o*.json"))):
        m_ = re.match(r"a(\d)_o(\d)\.json", os.path.basename(f))
        out[f"{m_.group(1)},{m_.group(2)}"] = json.load(open(f))
    return out


def frames_dir(combo, res):
    return os.path.join(OUTROOT, f"frames_{VERSION}_{res[0]}x{res[1]}", f"a{combo[0]}_o{combo[1]}")


# ============================================================================
# real-camera geometry (shared by render, check, video)
# ============================================================================
def real_camera(session, name):
    """Intrinsics + aligned-frame pose of one calibrated camera, built exactly
    as cage_scene.load_session builds the rig (Y = Ra (X - ta), mm -> m)."""
    import toml
    import cage_scene as cs
    cal = toml.load(f"{session}/calibration.toml")
    v = next(v for k, v in cal.items() if k.startswith("cam_") and v["name"] == name)
    ali = toml.load(f"{session}/alignment.toml")
    Ra = np.array(ali["rotation"], dtype=float)
    ta = np.array(ali["translation"], dtype=float)
    R = cs.rodrigues(v["rotation"])
    t = np.array(v["translation"], dtype=float)
    return {"name": name, "K": np.array(v["matrix"], float),
            "dist": np.array(v["distortions"], float),
            "rvec": np.array(v["rotation"], float), "tvec": t,
            "size": tuple(v["size"]), "Ra": Ra, "ta": ta,
            "C": Ra @ (-R.T @ t - ta) * cs.MM, "R_c2w": (R @ Ra.T).T}


def video_path(session, name):
    return glob.glob(f"{session}/{name}/{name}-*_CRF12_denoised.mp4")[0]


def video_window(res):
    """Where the 1280x1024 camera image sits in the output frame: full height,
    centred (pillarboxed). Returns (x0, y0, w, h) and the scale."""
    W, H = res
    sc = H / 1024.0
    w = int(round(1280 * sc))
    return (W - w) // 2, 0, w, H, sc


def frame_pts(vid):
    """Presentation timestamps of every frame, sorted (= display order). The
    containers are NOT consistent: some tick 1/30 s per frame, others 1/60 s
    (r_frame_rate 500000/8333) whatever the acquisition rate (30 fps, or 60 fps
    for the 2022-10-19 sessions; master_sheet frames/duration). Seeking by an assumed rate therefore
    lands on the wrong frame (a 2x error found by the overlay check); the h5
    frame index is the DISPLAY index, so look its pts up from the packets."""
    out = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error",
                          "-select_streams", "v", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", vid], check=True, capture_output=True, text=True)
    return np.array(sorted(float(x) for x in out.stdout.split() if x not in ("", "N/A")))


def decode_frames(vid, start, n, outdir):
    """Decode display frames start..start+n-1 of vid to outdir/raw####.png."""
    os.makedirs(outdir, exist_ok=True)
    pts = frame_pts(vid)
    dt = float(np.median(np.diff(pts)))
    t0 = pts[start] - dt / 2              # accurate seek lands ON frame `start`
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{t0:.6f}", "-i", vid,
                    "-frames:v", str(n), "-start_number", "0",
                    os.path.join(outdir, "raw%04d.png")], check=True)
    with open(os.path.join(outdir, "pts.json"), "w") as f:
        json.dump({"video": vid, "n_frames": len(pts), "frame_dt": dt,
                   "start": start, "seek_s": t0}, f)


# ============================================================================
# Blender side
# ============================================================================
class Scene:
    """One combo's world: enrichment_scene's tile scene, animated."""

    def __init__(self, combo, res, samples):
        import bpy
        import cage_scene as cs
        import enrichment_scene as es
        self.bpy, self.cs, self.es = bpy, cs, es
        self.combo = combo
        sp, f0 = combo_info(combo)
        self.session = f"{cs.SLAP2M}/{sp}"
        self.f0 = f0
        self.res = res
        n = seg_len(combo)

        bpy.ops.wm.read_factory_settings(use_empty=True)
        coll = bpy.context.scene.collection
        cage_nodes, cage, mouse_nodes, ali, cams = cs.load_session(self.session)
        self.cage_nodes, self.cage, self.mouse_nodes, self.cams = cage_nodes, cage, mouse_nodes, cams
        self.ctr = cage.mean(axis=0)
        focus = tuple(self.ctr * [1, 1, 0.7])

        M = {k: cs.pbr_mat(k, *v) for k, v in cs.PBR.items()}
        M["cage_wall"] = cs.flat_translucent("cage_wall", cs.PBR["cage_wall"][0],
                                             cs.PBR["cage_wall"][3])
        es.object_mats(M, es.OBJ_GREEN, es.OBJ_EDGE_ALPHA, es.OBJ_FACE_ALPHA)
        self.M = M
        # all fills built; the camera-facing one is hidden per frame (the
        # stills' open_toward, made view-dependent for the orbit)
        cs.build_cage(cage_nodes, cage, M, coll, open_toward=None)
        self.walls = {name: bpy.data.objects[f"cage_{name}"] for name in cs.CAGE_SURFACES}

        import h5py
        with h5py.File(f"{self.session}/aligned_points3d.h5") as f:
            st = step_for(self.session)
            trx = f["tracks"][f0:f0 + st * n:st] * cs.MM
        if trx.shape[0] < n:
            raise SystemExit(f"{combo}: session ends before frame {f0 + st * n}")
        if np.isnan(trx).any():
            raise SystemExit(f"{combo}: NaN keypoints in frames {f0}..{f0 + st * n}; "
                             "pick another representative frame")
        cs.clamp_tails(trx, cs.cage_planes(cage_nodes, cage))
        self.trx = trx
        self.animals = {tr: cs.build_animal(tr, trx[0, tr], mouse_nodes, M, coll)
                        for tr in range(trx.shape[1])}

        # synthetic objects: clear the whole segment's trajectory, not one frame
        corners, loop = es.floor_frame(cage_nodes, cage)
        traj = trx[::3, :, :, :2].reshape(-1, 2)
        self.objs = es.place_objects(combo[1], corners, loop, list(traj))
        if self.objs:
            print(f"  {combo}: object clearance over the segment = "
                  f"{1000 * object_clearance(self.objs, trx):.0f} mm")
        for name, opts, edges, faces in self.objs:
            es.build_object(name, opts, edges, faces, M, coll)

        zmin = float(cage[:, 2].min())
        # only the cut-in's camera is built (it is where the view flies to);
        # the rest of the rig would stand in front of the cage mid-orbit
        self.cam_objs = {}
        for c in cams:
            if combo not in CUTIN_CAM or c["name"] != CUTIN_CAM[combo]:
                continue
            before = set(bpy.data.objects.keys())
            cs.build_camera_unit(c["name"], c["C"], c["R_cam2world"], M, coll, focus)
            cs.build_camera_support(c["name"], c["C"], M, coll, zmin - 0.012)
            self.cam_objs[c["name"]] = [bpy.data.objects[k] for k in
                                        set(bpy.data.objects.keys()) - before]
        floor = cs.box("room_floor", (6.0, 6.0, 0.02), (0, 0, 0), M["floor"], coll)
        floor.location = (focus[0], focus[1], zmin - 0.012)
        floor.scale = (8.0, 8.0, 1.0)   # the pillar-extended real-camera view sees
                                        # past cage_scene's 6 m floor at its corners

        cs.setup_lighting(focus)
        for ob in bpy.data.objects:      # area lights must not show up when the
            if ob.type == "LIGHT":       # camera flies in close (grey wedge)
                ob.visible_camera = False
        cs.setup_cycles(samples, res)
        bpy.context.scene.cycles.transparent_max_bounces = 128

        cd = bpy.data.cameras.new("MovieCam")
        cd.type = "PERSP"
        cd.sensor_fit = "HORIZONTAL"
        cd.sensor_width = 36.0
        cd.clip_start = 0.01
        cd.clip_end = 100.0
        self.cam = cs.link(bpy.data.objects.new("MovieCam", cd), coll)
        self.cam.rotation_mode = "QUATERNION"
        bpy.context.scene.camera = self.cam

        self.aspect = res[0] / res[1]
        self.anchor_az = PLAIN_AZ
        self.real = None
        if combo in CUTIN_CAM:
            self.real = real_camera(self.session, CUTIN_CAM[combo])
            d = self.real["C"] - self.ctr
            self.anchor_az = math.degrees(math.atan2(d[1], d[0]))

    # ---- camera poses: (location, quaternion, focal/width) -------------------
    def orbit_pose(self, az, el=ORBIT_EL):
        from mathutils import Vector
        ortho, aim = self.es.fit_view(self.cage, az, el, self.aspect, MARGIN)
        d = Vector(self.cs.camera_dir(az, el))
        up = d.cross(Vector((0, 0, 1))).cross(d).normalized()
        aim = Vector(aim) - LIFT * ortho * up          # view centre drops -> cage rises
        loc = aim + ORBIT_DIST * d
        q = (aim - loc).to_track_quat("-Z", "Y")
        return loc, q, ORBIT_DIST / ortho, aim

    def real_pose(self):
        from mathutils import Matrix, Vector
        Rb = self.real["R_c2w"] @ np.diag([1.0, -1.0, -1.0])   # OpenCV -> Blender cam
        q = Matrix(Rb.tolist()).to_quaternion()
        # full output HEIGHT = the camera's 1024 rows; width is pillar-extended
        fx = self.real["K"][0, 0]
        fpx = fx * self.res[1] / 1024.0
        return Vector(self.real["C"]), q, fpx / self.res[0]

    def fly_pose(self, s):
        """s=0: orbit pose at the anchor's +25 deg; s=1: the real camera. Moves on
        a sphere around the aim point (az/el linear, distance log-linear), looks
        at the aim with the residual rotation to the real camera's true
        orientation slerped in, and sets focal so apparent size interpolates
        log-linearly -- no shrink-then-grow mid-flight."""
        from mathutils import Quaternion, Vector
        l0, q0, k0, aim = self.orbit_pose(self.anchor_az + 25.0)
        l1, q1, k1 = self.real_pose()
        if s <= 0:
            return l0, q0, k0
        if s >= 1:
            return l1, q1, k1

        def sph(p):
            d = p - aim
            r = d.length
            return math.atan2(d.y, d.x), math.asin(d.z / r), r
        a0, e0, r0 = sph(l0)
        a1, e1, r1 = sph(l1)
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        a, e = a0 + da * s, e0 + (e1 - e0) * s
        r = math.exp(math.log(r0) + (math.log(r1) - math.log(r0)) * s)
        loc = aim + r * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        look = (aim - loc).to_track_quat("-Z", "Y")
        look1 = (aim - l1).to_track_quat("-Z", "Y")
        resid = q1 @ look1.inverted()              # look-at -> true real orientation
        q = Quaternion((1, 0, 0, 0)).slerp(resid, s) @ look
        g0, g1 = k0 / r0, k1 / r1                  # focal per unit distance
        k = r * math.exp(math.log(g0) + (math.log(g1) - math.log(g0)) * s)
        return loc, q, k

    def pose_at(self, i):
        kind, p, _ = schedule(self.combo, i)
        if kind == "orbit":
            loc, q, k, _ = self.orbit_pose(self.anchor_az + p)
            return loc, q, k
        return self.fly_pose(p)

    # ---- per frame ----------------------------------------------------------
    def set_frame(self, i):
        loc, q, k = self.pose_at(i)
        self.cam.location = loc
        self.cam.rotation_quaternion = q
        self.cam.data.lens = k * 36.0
        toward = np.array(loc) - self.ctr
        toward /= np.linalg.norm(toward)
        open_ = set(self.cs.facing_surfaces(self.cage_nodes, self.cage, toward))
        for name, ob in self.walls.items():
            ob.hide_render = bool(name in open_)
        if self.real is not None:   # don't fly INTO the target camera's body
            near = bool(np.linalg.norm(np.array(loc) - self.real["C"]) < 0.30)
            for ob in self.cam_objs[self.real["name"]]:
                ob.hide_render = near
        for tr, h in self.animals.items():
            self.cs.update_animal(h, self.trx[i, tr])

    def render(self, outdir, frames):
        os.makedirs(outdir, exist_ok=True)
        for i in frames:
            out = os.path.join(outdir, f"f{i:04d}.png")
            if os.path.exists(out):
                continue
            t = time.time()
            self.set_frame(i)
            self.cs.render_to(out)
            print(f"  {self.combo} frame {i} ({time.time() - t:.1f}s)", flush=True)


def cmd_render(args):
    combo = tuple(args.combo)
    t = time.time()
    sc = Scene(combo, args.res, args.samples)
    n = seg_len(combo)
    frames = args.frames if args.frames else range(0, n, args.step)
    sc.render(frames_dir(combo, args.res), frames)
    print(f"RENDERED {combo} {len(list(frames))} frames in {time.time() - t:.0f}s", flush=True)


def cmd_all(args):
    """Render this worker's share (--part k/N) of all combos, ONE SUBPROCESS PER
    COMBO. v1 built the scenes back to back in one process and that is what
    blacked out the body membranes (see the BLACK MEMBRANES note in the
    docstring): with `use_persistent_data` the second scene's NEW materials
    rendered black. A fresh process per combo is the fix, not a render setting."""
    k, N = map(int, args.part.split("/"))
    log = os.path.join(OUTROOT, f"render_times_{VERSION}_{args.res[0]}x{args.res[1]}.jsonl")
    for j, combo in enumerate(ORDER):
        if j % N != k:
            continue
        t = time.time()
        subprocess.run([sys.executable, os.path.abspath(__file__), "render",
                        "--combo", str(combo[0]), str(combo[1]),
                        "--res", str(args.res[0]), str(args.res[1]),
                        "--samples", str(args.samples)], check=True)
        dt = time.time() - t
        print(f"RENDERED {combo} in {dt:.0f}s", flush=True)
        with open(log, "a") as f:
            f.write(json.dumps({"combo": combo, "frames": seg_len(combo),
                                "seconds": round(dt, 1), "res": args.res,
                                "samples": args.samples}) + "\n")


def black_px(path):
    """Pixels that are essentially black (every channel < 12). The cage ink
    renders at 20-30, so a clean frame scores ~0."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"))
    return int((a.max(axis=2) < 12).sum())


def cmd_qc(args):
    """Scan every rendered frame for the v1 black-membrane fault and re-render
    offenders in a fresh process (--fix). Writes qc_<res>.json."""
    res = tuple(args.res)
    bad = {}
    for combo in ORDER:
        d = frames_dir(combo, res)
        for i in range(seg_len(combo)):
            p = os.path.join(d, f"f{i:04d}.png")
            if not os.path.exists(p):
                bad.setdefault(combo, []).append((i, -1))
                continue
            c = black_px(p)
            if c > args.threshold:
                bad.setdefault(combo, []).append((i, c))
    with open(os.path.join(OUTROOT, f"qc_{VERSION}_{res[0]}x{res[1]}.json"), "w") as f:
        json.dump({f"{a},{r}": v for (a, r), v in bad.items()}, f, indent=1)
    for combo, v in bad.items():
        print(f"QC {combo}: {len(v)} frames flagged: {v[:12]}{'...' if len(v) > 12 else ''}")
    if not bad:
        print(f"QC clean: no frame above {args.threshold} black px")
    if args.fix and bad:
        for combo, v in bad.items():
            for i, _ in v:
                p = os.path.join(frames_dir(combo, res), f"f{i:04d}.png")
                if os.path.exists(p):
                    os.remove(p)
            subprocess.run([sys.executable, os.path.abspath(__file__), "render",
                            "--combo", str(combo[0]), str(combo[1]), "--res", str(res[0]),
                            str(res[1]), "--samples", str(args.samples),
                            "--frames", *[str(i) for i, _ in v]], check=True)
        print("re-rendered; run qc again to confirm")


# ============================================================================
# frame picking: a window where the animals stay on the floor and clear the
# (to-be-placed) enrichment objects for the WHOLE segment
# ============================================================================
BODY_NODES = ("Neck", "Trunk", "TTI", "Shoulder_left", "Shoulder_right",
              "Haunch_left", "Haunch_right")
#: Lowest BODY keypoint may sit at most this far above the rendered floor. The
#: keypoints are dorsal, so a mouse standing on the floor reads 25-50 mm here
#: (rating-0 sessions: median ~36 mm, 99th pct 50-60 mm, rearing included). In the
#: enriched sessions the median is 50-150 mm and the 95th pct 150-300 mm: the mice
#: spend most of the session ON the real cubes/dome, which are not in the scene --
#: that is the "floating mouse" the author saw. 60 mm keeps everyone on the floor
#: (or on another mouse) without rejecting rearing.
FLOOR_MAX_MM = 60.0
CLEAR_MIN_MM = 20.0      #: every keypoint keeps this xy clearance to every object footprint
CLEAR_GOOD_MM = 45.0     #: beyond this, prefer the window closest to the original frame
PICK_STRIDE = 60         #: candidate window starts, data frames


def object_clearance(objs, trx):
    """Smallest xy gap (m) between any keypoint in trx (T, A, 15, 3) and any
    placed object's footprint circle. inf when there are no objects."""
    xy = trx[..., :2].reshape(-1, 2)
    best = np.inf
    for _, pts, _, _ in objs:
        c = pts[:, :2].mean(axis=0)
        r = float(np.linalg.norm(pts[:, :2] - c, axis=1).max())
        best = min(best, float(np.linalg.norm(xy - c, axis=1).min()) - r)
    return best


def load_master_sheet():
    """master_sheet.xlsx rows as dicts (stdlib only: no openpyxl in bpyenv)."""
    import re
    import zipfile
    import xml.etree.ElementTree as ET
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(f"{SLAP2M}/master_sheet.xlsx")
    T = "{%s}t" % ns["m"]
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(t.text or "" for t in si.iter(T))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter("{%s}row" % ns["m"]):
        row = {}
        for c in r.findall("m:c", ns):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find("m:v", ns)
            val = None if v is None else v.text
            if c.get("t") == "s" and val is not None:
                val = ss[int(val)]
            isv = c.find("m:is", ns)                      # inline strings (this sheet)
            if isv is not None:
                val = "".join(t.text or "" for t in isv.iter(T))
            row[col] = val
        rows.append(row)
    hdr = rows[0]
    return [{hdr[k]: v for k, v in r.items() if k in hdr} for r in rows[1:] if any(r.values())]


def scan_session(session, level, span, mouse_nodes, cage_nodes, cage, stride, want=None):
    """Every window [s, s+span) of the session that passes the floor test,
    with its object clearance after placement. Returns list of dicts."""
    import h5py
    import cage_scene as cs
    import enrichment_scene as es
    with h5py.File(f"{session}/aligned_points3d.h5") as f:
        trx = f["tracks"][:] * cs.MM                      # (F, A, 15, 3)
    F, A = trx.shape[:2]
    idx = {n: i for i, n in enumerate(mouse_nodes)}
    body = [idx[n] for n in BODY_NODES]
    corners, loop = es.floor_frame(cage_nodes, cage)
    # height of the lowest body keypoint above the rendered floor, per frame/animal
    ctr = np.nanmean(trx[:, :, body, :2], axis=2)         # (F, A, 2)
    floor = np.full((F, A), np.nan)
    for a in range(A):
        for t in range(F):
            if not np.isnan(ctr[t, a]).any():
                floor[t, a] = es.floor_height(loop, ctr[t, a, 0], ctr[t, a, 1])
    lift = np.nanmin(trx[:, :, body, 2], axis=2) - floor  # (F, A)
    ok_frame = ~np.isnan(trx).any(axis=(1, 2, 3)) & (lift <= FLOOR_MAX_MM / 1000).all(axis=1)
    starts = list(range(0, F - span, stride))
    if want is not None and want not in starts and want + span <= F:
        starts.append(want)
    out = []
    for s in starts:
        if not ok_frame[s:s + span].all():
            continue
        w = trx[s:s + span]
        if level == 0:
            out.append({"frame": s, "clearance_mm": None, "max_lift_mm": 1000 * float(lift[s:s + span].max())})
            continue
        # place against the centroid track (fast), then measure against every keypoint
        traj = list(np.nanmean(w[::2, :, :, :2], axis=2).reshape(-1, 2))
        objs = es.place_objects(level, corners, loop, traj)
        if len(objs) < sum(len(u["cubes"]) + (1 if "dome" in u else 0) for u in es.LEVELS[level]):
            continue                                       # a unit did not fit
        out.append({"frame": s, "clearance_mm": 1000 * object_clearance(objs, w),
                    "max_lift_mm": 1000 * float(lift[s:s + span].max())})
    return out, F, A


def cmd_pick(args):
    """Choose each combo's (session, first frame) and write frame_picks.json.
    Keeps the COMBOS session when it has a passing window; otherwise walks the
    other master_sheet sessions recorded under the same (animals, rating)."""
    import cage_scene as cs
    from enrichment_scene import COMBOS
    combos = [tuple(args.combo)] if args.combo else ORDER
    os.makedirs(PICK_DIR, exist_ok=True)
    sheet = load_master_sheet()
    for combo in combos:
        a, lvl = combo
        sp0, f0 = COMBOS[combo]
        cands = [sp0] + [r["session_path"] for r in sheet
                         if int(r["animals"]) == a and int(r["obstacle_rating"]) == lvl
                         and r["session_path"] != sp0]
        chosen = None
        for sp in cands:
            session = f"{SLAP2M}/{sp}"
            span = step_for(sp) * seg_len(combo)
            try:
                cage_nodes, cage, mouse_nodes, _, _ = cs.load_session(session)
            except Exception as e:
                print(f"  {combo} {sp}: unreadable ({e}), skipped")
                continue
            wins, F, A = scan_session(session, lvl, span, mouse_nodes, cage_nodes, cage,
                                      args.stride, want=f0 if sp == sp0 else None)
            if A != a:
                print(f"  {combo} {sp}: h5 has {A} tracks, sheet says {a}; skipped")
                continue
            good = [w for w in wins if lvl == 0 or w["clearance_mm"] >= CLEAR_MIN_MM]
            print(f"  {combo} {sp}: {F} frames, {len(wins)} on-floor windows, "
                  f"{len(good)} clear", flush=True)
            if not good:
                continue
            ref = f0 if sp == sp0 else F // 2
            if lvl == 0:
                best = min(good, key=lambda w: abs(w["frame"] - ref))
            else:
                great = [w for w in good if w["clearance_mm"] >= CLEAR_GOOD_MM]
                best = (min(great, key=lambda w: abs(w["frame"] - ref)) if great
                        else max(good, key=lambda w: w["clearance_mm"]))
            chosen = dict(best, session=sp, span=span, original=[sp0, f0],
                          changed_session=sp != sp0, changed_frame=(sp != sp0 or best["frame"] != f0))
            break
        if chosen is None:
            print(f"  {combo}: NO clean window in any session; keeping COMBOS "
                  f"(flag for the author)")
            chosen = {"session": sp0, "frame": f0, "span": span, "original": [sp0, f0],
                      "changed_session": False, "changed_frame": False, "clearance_mm": None,
                      "unresolved": True}
        print(f"PICK {combo}: {chosen}", flush=True)
        with open(os.path.join(PICK_DIR, f"a{a}_o{lvl}.json"), "w") as f:
            json.dump(chosen, f, indent=1)
    # a merged, human-readable view; the renderer reads the per-combo files
    with open(PICKS, "w") as f:
        json.dump(merged_picks(), f, indent=1)


def cmd_check(args):
    """Alignment check for a cut-in: render the Blender scene from the real
    camera's pose at the cut-in's first video frame, at the camera's own
    1280x1024 / K, and blend it over the undistorted real frame; also draw the
    3D keypoints projected with cv2.projectPoints (incl. distortion) on the RAW
    frame, which must agree with reprojections.h5 to float precision."""
    import cv2
    import h5py
    combo = tuple(args.combo)
    cam_res = (1280, 1024)
    sc = Scene(combo, cam_res, args.samples)
    i = ORBIT_A + FLY                      # first cross-fade frame
    df = sc.f0 + step_for(sc.session) * i
    sc.set_frame(i)
    loc, q, _ = sc.real_pose()
    sc.cam.location, sc.cam.rotation_quaternion = loc, q
    sc.cam.data.lens = sc.real["K"][0, 0] / 1280 * 36.0
    outd = os.path.join(OUTROOT, "check")
    os.makedirs(outd, exist_ok=True)
    tag = f"{VERSION}_a{combo[0]}_o{combo[1]}_{sc.real['name']}_f{df}"
    rpath = os.path.join(outd, f"{tag}_blender.png")
    sc.cs.render_to(rpath)

    tmp = os.path.join(outd, f"_raw_{tag}")
    decode_frames(video_path(sc.session, sc.real["name"]), df, 1, tmp)
    raw = cv2.imread(os.path.join(tmp, "raw0000.png"))
    K, d = sc.real["K"], sc.real["dist"]
    und = cv2.undistort(raw, K, d)
    und = stretch(und, *stretch_levels(und))
    ren = cv2.imread(rpath)
    # only the rendered CONTENT (wireframe, animals, camera bodies -- anything
    # clearly darker or more saturated than the white room) is laid over the video
    rf = ren.astype(np.float32)
    ink = (rf.min(axis=2) < 170) | (rf.max(axis=2) - rf.min(axis=2) > 40)
    blend = und.copy()
    blend[ink] = (0.3 * und[ink] + 0.7 * rf[ink]).astype(np.uint8)
    # projected keypoints on the raw (distorted) frame
    with h5py.File(f"{sc.session}/aligned_points3d.h5") as f:
        Y = f["tracks"][df].reshape(-1, 3)
    X = (sc.real["Ra"].T @ Y.T).T + sc.real["ta"]
    p, _ = cv2.projectPoints(X, sc.real["rvec"], sc.real["tvec"], K, d)
    with h5py.File(f"{sc.session}/reprojections.h5") as f:
        rep = f[sc.real["name"]][df].reshape(-1, 2)
    err = float(np.nanmax(np.abs(p.reshape(-1, 2) - rep)))
    rawv = stretch(raw, *stretch_levels(raw))
    for (x, y) in p.reshape(-1, 2):
        cv2.circle(rawv, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
    # undistorted-pinhole projection on the blend (what the Blender camera does)
    pu, _ = cv2.projectPoints(X, sc.real["rvec"], sc.real["tvec"], K, None)
    for (x, y) in pu.reshape(-1, 2):
        cv2.circle(blend, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(outd, f"{tag}_overlay.png"), blend)
    cv2.imwrite(os.path.join(outd, f"{tag}_rawproj.png"), rawv)
    with open(os.path.join(outd, f"{tag}.json"), "w") as f:
        json.dump({"combo": combo, "session": sc.session, "camera": sc.real["name"],
                   "data_frame": df, "max_px_vs_reprojections_h5": err}, f, indent=1)
    print("check", tag, "max |projectPoints - reprojections.h5| =", err, "px")


# ============================================================================
# real video: decode, undistort, contrast-stretch, scale into the window
# ============================================================================
def stretch_levels(img):
    g = img if img.ndim == 2 else img.mean(axis=2)
    return float(np.percentile(g, 1.0)), float(np.percentile(g, 99.7))


def stretch(img, lo, hi, gamma=0.8):
    x = np.clip((img.astype(np.float32) - lo) / max(1.0, hi - lo), 0, 1) ** gamma
    return (x * 255 + 0.5).astype(np.uint8)


def cmd_video(args):
    import cv2
    combo = tuple(args.combo)
    sp, f0 = combo_info(combo)
    session = f"{SLAP2M}/{sp}"
    rc = real_camera(session, CUTIN_CAM[combo])
    first = ORBIT_A + FLY
    n = FADE + HOLD + FADE
    outd = os.path.join(OUTROOT, f"video_{VERSION}", f"a{combo[0]}_o{combo[1]}")
    raw = os.path.join(outd, "raw")
    # the video plays at the same DATA_STEP as the 3D: decode the whole span and
    # keep every DATA_STEP-th frame
    st = step_for(sp)
    nraw = st * n
    if not os.path.exists(os.path.join(raw, f"raw{nraw - 1:04d}.png")):
        decode_frames(video_path(session, rc["name"]), f0 + st * first, nraw, raw)
    K, d = rc["K"], rc["dist"]
    m1, m2 = cv2.initUndistortRectifyMap(K, d, None, K, (1280, 1024), cv2.CV_32FC1)
    # k1-only radial model folds over beyond r_max = sqrt(-1/(3 k1)); pixels whose
    # undistorted radius exceeds it map to garbage -- black them out
    k1 = d[0]
    yy, xx = np.mgrid[0:1024, 0:1280].astype(np.float32)
    ru = np.hypot((xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1])
    valid = ru < (math.sqrt(-1 / (3 * k1)) if k1 < 0 else 1e9)
    first_img = cv2.imread(os.path.join(raw, "raw0000.png"))
    lo, hi = stretch_levels(cv2.remap(first_img, m1, m2, cv2.INTER_LINEAR))
    for j in range(n):
        im = cv2.imread(os.path.join(raw, f"raw{st * j:04d}.png"))
        u = cv2.remap(im, m1, m2, cv2.INTER_LINEAR)
        u[~valid] = 0
        cv2.imwrite(os.path.join(outd, f"v{j:04d}.png"), stretch(u, lo, hi))
    with open(os.path.join(outd, "meta.json"), "w") as f:
        json.dump({"session": sp, "camera": rc["name"],
                   "first_data_frame": f0 + st * first, "data_step": st,
                   "n": n, "levels": [lo, hi], "gamma": 0.8}, f)
    print("video", combo, rc["name"], "first data frame", f0 + st * first, "n", n)


# ============================================================================
# compose
# ============================================================================
def font(size, bold=False, italic=False):
    from PIL import ImageFont
    name = "LiberationSans-" + ({(0, 0): "Regular", (1, 0): "Bold", (0, 1): "Italic",
                                 (1, 1): "BoldItalic"}[(int(bold), int(italic))]) + ".ttf"
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def combo_label(combo):
    a, r = combo
    return f"{a} mouse" if a == 1 else f"{a} mice", f"enrichment rating {r}"


class Composer:
    def __init__(self, res):
        from PIL import Image
        self.W, self.H = res
        self.s = self.W / OUT_W
        g = Image.open(GRID).convert("RGB")
        self.grid = g

    def px(self, v):
        return max(1, int(round(v * self.s)))

    def blank(self, c=(255, 255, 255)):
        from PIL import Image
        return Image.new("RGB", (self.W, self.H), c)

    def title(self, k, n):
        from PIL import ImageDraw
        im = self.blank()
        d = ImageDraw.Draw(im)
        d.text((self.W / 2, self.H * 0.44), "SLAP-2M", fill=(25, 25, 25),
               font=font(self.px(96), bold=True), anchor="mm")
        d.text((self.W / 2, self.H * 0.56),
               "74 sessions, 1–4 mice, graded environmental enrichment",
               fill=(60, 60, 60), font=font(self.px(46)), anchor="mm")
        d.text((self.W / 2, self.H * 0.66),
               "real tracked 3D poses · one representative session per condition",
               fill=(120, 120, 120), font=font(self.px(30), italic=True), anchor="mm")
        return fade(im, k, n, self, fin=10, fout=0)

    def grid_card(self, k, n, zoom0, zoom1, fin, fout, note=None):
        from PIL import Image, ImageDraw
        u = k / max(1, n - 1)
        z = zoom0 + (zoom1 - zoom0) * smooth(u)
        g = self.grid
        fit = min(self.W * 0.92 / g.width, self.H * 0.86 / g.height) * z
        gw, gh = int(g.width * fit), int(g.height * fit)
        gi = g.resize((gw, gh), Image.LANCZOS)
        im = self.blank()
        im.paste(gi, ((self.W - gw) // 2, (self.H - gh) // 2 - self.px(10)))
        if note:
            ImageDraw.Draw(im).text((self.W / 2, self.H - self.px(28)), note,
                                    fill=(110, 110, 110), font=font(self.px(26), italic=True),
                                    anchor="mm")
        return fade(im, k, n, self, fin=fin, fout=fout)

    def combo_frame(self, combo, i, res, video_dir, meta):
        from PIL import Image, ImageDraw
        im = Image.open(os.path.join(frames_dir(combo, res), f"f{i:04d}.png")).convert("RGB")
        if im.size != (self.W, self.H):
            im = im.resize((self.W, self.H), Image.LANCZOS)
        _, _, va = schedule(combo, i)
        x0, y0, ww, wh, _ = video_window((self.W, self.H))
        if va > 0:
            j = i - (ORBIT_A + FLY)
            v = Image.open(os.path.join(video_dir, f"v{j:04d}.png")).convert("RGB")
            v = v.resize((ww, wh), Image.LANCZOS)
            region = im.crop((x0, y0, x0 + ww, y0 + wh))
            im.paste(Image.blend(region, v, va), (x0, y0))
            d = ImageDraw.Draw(im, "RGBA")
            a = int(255 * va)
            d.rectangle([x0, y0, x0 + ww - 1, y0 + wh - 1], outline=(255, 255, 255, a),
                        width=self.px(3))
            sp, f0 = combo_info(combo)
            txt = (f"real camera ‘{meta['camera']}’ · frame {f0 + step_for(sp) * i} "
                   f"· lens-undistorted, contrast-stretched")
            f = font(self.px(26))
            bb = d.textbbox((0, 0), txt, font=f)
            tx, ty = x0 + self.px(18), y0 + self.px(18)
            d.rounded_rectangle([tx - self.px(10), ty - self.px(8),
                                 tx + bb[2] + self.px(10), ty + bb[3] + self.px(10)],
                                radius=self.px(8), fill=(0, 0, 0, int(150 * va)))
            d.text((tx, ty), txt, fill=(255, 255, 255, a), font=f)
        self.lower_third(im, combo, i)
        return im

    def lower_third(self, im, combo, i):
        from PIL import ImageDraw
        d = ImageDraw.Draw(im, "RGBA")
        sp, f0 = combo_info(combo)
        a, r = combo_label(combo)
        big, small = font(self.px(44), bold=True), font(self.px(24))
        l1 = f"{a} · {r}"
        ns = N_SESSIONS[combo]
        l2 = (f"session {sp.split('/')[1]} · frames {f0}–{f0 + data_span(combo) - 1} "
              f"at {SPEED}x "
              f"· 1 of {ns} session{'s' if ns > 1 else ''} in this condition")
        x, y = self.px(48), self.H - self.px(140)
        w1 = d.textbbox((0, 0), l1, font=big)[2]
        w2 = d.textbbox((0, 0), l2, font=small)[2]
        d.rounded_rectangle([x - self.px(20), y - self.px(16), x + max(w1, w2) + self.px(22),
                             y + self.px(100)], radius=self.px(10), fill=(255, 255, 255, 215))
        d.rectangle([x - self.px(20), y - self.px(16), x - self.px(13), y + self.px(100)],
                    fill=(30, 132, 73, 255) if combo[1] > 0 else (60, 60, 60, 255))
        d.text((x, y), l1, fill=(20, 20, 20), font=big)
        d.text((x, y + self.px(60)), l2, fill=(80, 80, 80), font=small)
        if combo[1] > 0 and OBJECT_CAVEAT:
            cap = OBJECT_CAVEAT
            f = font(self.px(26), italic=True)
            bb = d.textbbox((0, 0), cap, font=f)
            cx, cy = self.W - self.px(40) - bb[2], self.px(36)
            d.rounded_rectangle([cx - self.px(14), cy - self.px(10), cx + bb[2] + self.px(14),
                                 cy + bb[3] + self.px(12)], radius=self.px(8),
                                fill=(255, 255, 255, 215))
            d.text((cx, cy), cap, fill=(30, 110, 60), font=f)


def fade(im, k, n, C, fin=0, fout=0, color=(255, 255, 255)):
    from PIL import Image
    a = 1.0
    if fin and k < fin:
        a = min(a, (k + 1) / fin)
    if fout and k >= n - fout:
        a = min(a, (n - k) / fout)
    if a >= 1:
        return im
    return Image.blend(C.blank(color), im, a)


def cmd_compose(args):
    from PIL import Image
    res = tuple(args.res)
    C = Composer(res)
    combos = [tuple(args.only)] if args.only else ORDER
    # build the sequence of segment generators (each yields PIL frames)
    segs = []
    if not args.only:
        segs.append(("title", TITLE_N, lambda k, n: C.title(k, n)))
        segs.append(("grid", GRID_N, lambda k, n: C.grid_card(
            k, n, 1.0, 1.07, 12, 0,
            note="grid shows 10 of the 14 recorded conditions \u00b7 all 14 follow")))
    for combo in combos:
        vd = os.path.join(OUTROOT, f"video_{VERSION}", f"a{combo[0]}_o{combo[1]}")
        meta = json.load(open(os.path.join(vd, "meta.json"))) if combo in CUTIN_CAM else None
        segs.append((f"combo{combo}", seg_len(combo),
                     (lambda c, v, m: lambda k, n: C.combo_frame(c, k, res, v, m))(combo, vd, meta)))
    if not args.only:
        segs.append(("end", END_N, lambda k, n: C.grid_card(
            k, n, 1.07, 1.07, 0, 20,
            note="animal poses are the real tracked 3D")))
    out = args.out or os.path.join(OUTROOT, f"supp_movie3_slap2m_{VERSION}.mp4")
    ff = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{res[0]}x{res[1]}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    total = 0
    tail = []                          # last XFADE frames of the previous segment
    stills = set(args.stills or [])
    for si, (name, n, gen) in enumerate(segs):
        for k in range(n):
            im = gen(k, n)
            if si > 0 and k < XFADE and tail:
                # dissolve: overlap this segment's head with the previous tail
                im = Image.blend(tail[k], im, (k + 1) / (XFADE + 1))
            if si < len(segs) - 1 and k >= n - XFADE:
                if k == n - XFADE:
                    tail = []
                tail.append(im)
                continue               # emitted blended into the next segment
            ff.stdin.write(im.tobytes())
            if total in stills:
                im.save(os.path.join(OUTROOT, f"still_{VERSION}_{total:05d}_{name}.png"))
            total += 1
        print(f"composed {name}: {n} frames (running total {total})", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out}: {total} frames = {total / FPS:.1f} s")


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "check", "video", "all"):
        p = sub.add_parser(name)
        if name != "all":
            p.add_argument("--combo", type=int, nargs=2, required=True, metavar=("ANIMALS", "RATING"))
        p.add_argument("--res", type=int, nargs=2, default=[OUT_W, OUT_H])
        p.add_argument("--samples", type=int, default=24)
        if name == "render":
            p.add_argument("--step", type=int, default=1)
            p.add_argument("--frames", type=int, nargs="*")
        if name == "all":
            p.add_argument("--part", default="0/1")
    p = sub.add_parser("qc")
    p.add_argument("--res", type=int, nargs=2, default=[OUT_W, OUT_H])
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--threshold", type=int, default=50)
    p.add_argument("--fix", action="store_true")
    p = sub.add_parser("pick")
    p.add_argument("--combo", type=int, nargs=2, metavar=("ANIMALS", "RATING"))
    p.add_argument("--stride", type=int, default=PICK_STRIDE)
    p = sub.add_parser("compose")
    p.add_argument("--res", type=int, nargs=2, default=[OUT_W, OUT_H])
    p.add_argument("--only", type=int, nargs=2)
    p.add_argument("--out")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--stills", type=int, nargs="*", help="output frame numbers to save as PNG")
    args = ap.parse_args()
    {"render": cmd_render, "check": cmd_check, "video": cmd_video, "pick": cmd_pick, "qc": cmd_qc,
     "all": cmd_all, "compose": cmd_compose}[args.cmd](args)


if __name__ == "__main__":
    main()
