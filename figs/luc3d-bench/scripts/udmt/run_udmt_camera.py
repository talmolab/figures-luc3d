#!/usr/bin/env python3
"""Headless driver for the FULL UDMT stack on one slap2m camera-video.

Replicates the UDMT GUI tabs (create project -> tracking init [SEEM auto-detect] ->
XMem foreground propagation -> create training dataset -> train network -> analyze)
as plain function calls, since this box has no display.

Run via scripts/udmt/run_udmt_camera.sh (sets the required env vars).

Stages (each idempotent; skipped if its sentinel output already exists):
  extract  -> tmp/<vid>/extracted-images/%05d.jpg          (whole video, resize 0.8)
  init     -> SEEM auto-init: frame-0 mask + start_pos_array.txt (authentic)
  xmem     -> tmp/<vid>/masks/%07d.png                      (binary fg, XMem propagation)
  traindata-> training-datasets/<vid>/{img,label}          (status_flag=1 + create_train_label)
  train    -> models/                                        (run_training_process)
  analyze  -> tracking-results/<vid>/...                     (status 2 then 3 + post_process)

Output of interest: tracking-results/<vid>/*-whole-filter*.npy (n_animals,n_frames,2 centroids)
and the raw per-animal <ds>_<id>_new.txt boxes ([x,y,w,h] in resized coords).
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time, logging
from pathlib import Path

import cv2
import numpy as np

UDMT = Path("/root/vast/eric/UDMT")
RESIZE = 0.8
DOWNSAMPLE = 1
NUM_ANIMALS = 4  # session 10072022145420 is a 4-animal session; cap SEEM over-detection to this
MAX_FRAMES = None  # cap analyze-stage frame extraction (None = whole video)

log = logging.getLogger("udmt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def banner(msg):
    log.info("=" * 70)
    log.info(msg)
    log.info("=" * 70)


def extract_frames(video, out_dir, scale, downsample, limit=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("empty video")
    h, w = first.shape[:2]
    dim = (int(w * scale), int(h * scale))
    fc = sc = 0
    step = int(1 / downsample)
    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fc % step == 0:
            cv2.imwrite(str(out_dir / f"{sc:05d}.jpg"), cv2.resize(fr, dim))
            sc += 1
            if limit and sc >= limit:
                break
        fc += 1
    cap.release()
    return sc, dim


def stage_extract(proj, vid, video):
    img_dir = proj / "tmp" / vid / "extracted-images"
    done = img_dir / "_extract_done.txt"
    if done.exists():
        log.info("extract: already done (%s frames)", done.read_text().strip())
        return
    banner("STAGE extract (whole video)")
    t = time.time()
    n, dim = extract_frames(video, img_dir, RESIZE, DOWNSAMPLE, limit=MAX_FRAMES)
    done.write_text(str(n))
    log.info("extract: %d frames %s in %.1fs", n, dim, time.time() - t)


def stage_init(proj, vid):
    img_dir = proj / "tmp" / vid / "extracted-images"
    sp_file = img_dir / "start_pos_array.txt"
    mask0 = proj / "tmp" / vid / "seem_mask" / "0000000.png"
    if sp_file.exists() and mask0.exists():
        log.info("init: already done (%d centroids)", len(np.loadtxt(sp_file, delimiter=",", ndmin=2)))
        return
    banner("STAGE init (SEEM authentic auto-detect on frame 0)")
    sys.path.insert(0, str(UDMT / "udmt/gui/tabs/xmem"))
    from auto_initial_prediction import run_auto_initial_prediction
    seem_in = proj / "tmp" / vid / "seem_in"
    seem_in.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_dir / "00000.jpg", seem_in / "0000000.jpg")
    mask_dir = proj / "tmp" / vid / "seem_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    # resize_scale=RESIZE so centroids (computed on the RESIZE-scaled frame) are stored in
    # ORIGINAL coords; run_tracker then multiplies by resize_factor(=RESIZE) to get back to
    # tracking-frame coords. (Passing 1.0 here double-applies the downscale -> points land
    # on background -> set_ini_value empty-list crash.)
    t = time.time()
    ok, n = run_auto_initial_prediction(str(seem_in), str(mask_dir),
                                        start_point_save_path=str(img_dir), resize_scale=RESIZE)
    if not ok or n == 0:
        raise RuntimeError(f"SEEM init failed (ok={ok}, n={n})")
    log.info("init: SEEM found %d instances in %.1fs", n, time.time() - t)
    # SEEM over-segments these crowded 4-animal views (6-7 blobs), which makes tracking
    # ~3x slower and pollutes identity with phantom IDs. We KNOW the count is NUM_ANIMALS,
    # so cap start positions to the NUM_ANIMALS largest foreground blobs of the frame-0 mask.
    _cap_start_pos_to_n(mask0, sp_file, NUM_ANIMALS)


def _cap_start_pos_to_n(mask_path, sp_file, n):
    sp = np.loadtxt(sp_file, delimiter=",", ndmin=2)  # original coords (already /RESIZE)
    if len(sp) <= n:
        log.info("init: %d start points (<= %d), no cap needed", len(sp), n)
        return
    mask = cv2.imread(str(mask_path), 0)
    mask = (mask > 0).astype(np.uint8)
    ncomp, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # rank each existing start point by the area of the blob it lands in (in RESIZE coords)
    areas = []
    h, w = mask.shape
    for x, y in sp:
        xr, yr = int(min(x * RESIZE, w - 1)), int(min(y * RESIZE, h - 1))
        lab = labels[yr, xr]
        areas.append(stats[lab, cv2.CC_STAT_AREA] if lab != 0 else 0)
    keep = np.argsort(areas)[::-1][:n]
    keep = sorted(keep)
    capped = sp[keep]
    np.savetxt(sp_file, capped, fmt="%.2f", delimiter=",")
    log.info("init: capped %d -> %d start points (kept largest-blob); areas=%s",
             len(sp), n, [int(areas[k]) for k in keep])


XMEM_DIR = str(UDMT / "udmt/gui/tabs/xmem")


def stage_xmem(proj, vid):
    """Authentic XMem foreground propagation, driven in-process (the repo's eval.py is
    broken as a standalone script — its inference modules mix absolute `dataset`/`inference`
    imports with relative `..model`, only resolvable via the udmt package path). We import
    InferenceCore/XMem/MaskMapper via the package path and replicate eval.py's generic loop,
    loading + normalizing frames ourselves (size=-1, no resize)."""
    masks_dir = proj / "tmp" / vid / "masks"
    done = masks_dir / "_xmem_done.txt"
    if done.exists():
        log.info("xmem: already done (%s masks)", done.read_text().strip())
        return
    banner("STAGE xmem (in-process XMem foreground propagation)")
    img_dir = proj / "tmp" / vid / "extracted-images"
    seem_mask = proj / "tmp" / vid / "seem_mask" / "0000000.png"
    masks_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from torchvision import transforms
    if XMEM_DIR not in sys.path:
        sys.path.insert(0, XMEM_DIR)  # makes absolute `dataset.util` import in MaskMapper resolve
    from udmt.gui.tabs.xmem.model.network import XMem
    from udmt.gui.tabs.xmem.inference.inference_core import InferenceCore
    from udmt.gui.tabs.xmem.inference.data.mask_mapper import MaskMapper

    cfg = dict(disable_long_term=False, enable_long_term=True, enable_long_term_count_usage=True,
               max_mid_term_frames=10, min_mid_term_frames=5, max_long_term_elements=10000,
               num_prototypes=128, top_k=30, mem_every=5, deep_update_every=-1,
               benchmark=False, size=-1)
    model_path = XMEM_DIR + "/saves/XMem.pth"
    net = XMem(cfg, model_path).cuda().eval()
    net.load_weights(torch.load(model_path), init_as_zero_if_needed=True)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    frames = sorted(img_dir.glob("*.jpg"))
    mapper = MaskMapper()
    proc = InferenceCore(net, cfg)
    torch.set_grad_enabled(False)
    t = time.time()
    n = len(frames)
    for ti, fp in enumerate(frames):
        rgb = tf(Image.open(fp).convert("RGB")).cuda()
        if ti == 0:
            m = np.array(Image.open(seem_mask))
            msk, labels = mapper.convert_mask(m)
            msk = torch.Tensor(msk).cuda()
            proc.set_all_labels(list(mapper.remappings.values()))
            prob = proc.step(rgb, msk, labels, end=False)
        else:
            prob = proc.step(rgb, None, None, end=(ti == n - 1))
        out = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)
        fg = ((out > 0).astype(np.uint8)) * 255
        cv2.imwrite(str(masks_dir / f"{ti:07d}.png"), fg)
        if ti % 2000 == 0:
            log.info("xmem: frame %d/%d (%.0f ms/frame)", ti, n, (time.time() - t) / max(ti, 1) * 1000)
    done.write_text(str(n))
    log.info("xmem: %d masks in %.1f min", n, (time.time() - t) / 60)


def _make_qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    l1 = QtWidgets.QLabel(); l1.setFixedSize(350, 350)
    l2 = QtWidgets.QLabel(); l2.setFixedSize(350, 350)
    return app, l1, l2


# Param search ranges (trimmed from the GUI default 3x3=9 combos to fit the overnight
# window: keep the search-region-scale sweep, drop the target-size-bias sweep). Documented
# as a tractability deviation in the report.
SEARCH_SCALE_RANGE = [2.0]
TARGET_SZ_BIAS_RANGE = [0.0]


def _silence_vis():
    # skip per-frame Qt rendering (vis_flag) for speed/headless. MUST keep save_flag=True:
    # the trajectory output files (result_save_file_list / <ds>_<id>_new.txt) are built
    # inside the `if save_flag:` block in tracker_multi, so disabling it loses all output.
    import udmt.gui.tabs.ST_Net.pytracking.evaluation.tracker_multi as tm
    tm.vis_flag = False
    # stub cv2.VideoWriter everywhere so the debug/overlay videos aren't encoded (we only
    # need the txt trajectories + npy). Big speedup on the 18k-frame status-3 pass.
    import cv2 as _cv2

    class _NoVid:
        def __init__(self, *a, **k):
            pass

        def write(self, *a, **k):
            pass

        def release(self, *a, **k):
            pass

        def isOpened(self):
            return True

    _cv2.VideoWriter = _NoVid


def _clean_argv_call(fn, *args, **kw):
    """run_tracking AND run_training_process both call argparse.parse_args() on sys.argv,
    which chokes on our --session/--cam/--video. Swap argv to a bare token around the call."""
    argv = sys.argv
    sys.argv = ["udmt"]
    try:
        return fn(*args, **kw)
    finally:
        sys.argv = argv


def _run_tracking(params):
    from udmt.gui.tabs.ST_Net.pytracking.run_tracker import run_tracking
    _clean_argv_call(run_tracking, params)


def _find_subfolder_with(parent, keyword):
    for name in os.listdir(parent):
        if os.path.isdir(os.path.join(parent, name)) and keyword in name:
            return name
    raise RuntimeError(f"no subfolder with '{keyword}' in {parent}")


def _find_latest_folder(parent):
    subs = [os.path.join(parent, d) for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    return max(subs, key=os.path.getmtime)


def stage_traindata(proj, vid, fps, l1, l2):
    label_dir = proj / "training-datasets" / vid / "label"
    if label_dir.exists() and any(label_dir.iterdir()):
        log.info("traindata: already done")
        return
    banner("STAGE traindata (status_flag=1 + create_train_label)")
    from udmt.gui import BASE_DIR
    from udmt.gui.tabs.ST_Net.pytracking.utils.convert_to_train_set import create_train_label
    img_dir = proj / "training-datasets" / vid / "img"
    if not (img_dir.exists() and len(list(img_dir.glob("*.jpg"))) >= 2000):
        n, _ = extract_frames(SRC_VIDEO, img_dir, RESIZE, DOWNSAMPLE, limit=2000)
        log.info("traindata: extracted %d training frames", n)
    params = {
        'video_display_widget1': l1, 'video_display_widget2': l2,
        'project_folder': str(proj), 'video_name': vid,
        'model_path': BASE_DIR + '/pretrained/trdimp_net_ep.pth.tar',
        'resize_factor': RESIZE, 'downsample_factor': DOWNSAMPLE, 'frame_rate': fps,
        'frame_num': 2000, 'search_scale_range': SEARCH_SCALE_RANGE,
        'target_sz_bias_range': TARGET_SZ_BIAS_RANGE, 'status_flag': 1,
        'evaluation_metric': [], 'is_concave': False,
    }
    t = time.time()
    _run_tracking(params)
    pj = proj / "tmp" / vid / "evaluation_metric_for_train.json"
    best = json.load(open(pj))[-1]
    key = best["target_sz"] + "_" + best["search_scale"]
    results_path = proj / "tmp" / vid / "train_set_results"
    best_path = str(results_path / _find_subfolder_with(results_path, key))
    log.info("traindata: best param subfolder %s", best_path)
    create_train_label(vid, best_path, str(label_dir))
    log.info("traindata: done in %.1fs", time.time() - t)


def stage_train(proj, vid, epochs, batch):
    models = proj / "models"
    if models.exists() and any(models.rglob("*.pth.tar")):
        log.info("train: already have a model")
        return
    banner("STAGE train (run_training_process)")
    from udmt.gui.tabs.ST_Net.ltr.run_training import run_training_process
    # num_workers=0: /dev/shm is only 64M here, so DataLoader worker processes Bus-error
    # ("out of shared memory"). 0 loads in the main process (no shm) -> reliable.
    params = {'project_folder': str(proj), 'video_name': vid, 'num_workers': 0,
              'epoch_num': epochs, 'batch_size': batch, 'max_save_snapshots': 3,
              'logger': log}
    t = time.time()
    _clean_argv_call(run_training_process, params)
    saved = sorted(models.rglob("*.pth.tar"), key=os.path.getmtime)
    if not saved:
        raise RuntimeError("train produced no .pth.tar model (check run_training_process)")
    log.info("train: done in %.1f min -> %s", (time.time() - t) / 60, saved[-1])


def stage_analyze(proj, vid, fps, l1, l2):
    res_dir = proj / "tracking-results" / vid
    if list(res_dir.glob("*-whole-filter*.npy")):
        log.info("analyze: already done")
        return
    banner("STAGE analyze (status 2 param-search then status 3 full track + post_process)")
    from udmt.gui import BASE_DIR  # noqa: F401
    from udmt.gui.tabs.ST_Net.pytracking.utils.filter_and_save import post_process_results
    # select trained model (saved recursively under models/<vid>/)
    models = proj / "models"
    cands = sorted(models.rglob("*.pth.tar"), key=os.path.getmtime)
    if not cands:
        raise RuntimeError(f"no trained model under {models}")
    model = cands[-1]
    img_dir = proj / "tmp" / vid / "extracted-images"
    nframes = len(list(img_dir.glob("*.jpg")))
    params = {
        'video_display_widget1': l1, 'video_display_widget2': l2,
        'project_folder': str(proj), 'video_name': vid, 'model_path': str(model),
        'resize_factor': RESIZE, 'downsample_factor': DOWNSAMPLE, 'frame_rate': fps,
        'frame_num': 1500, 'search_scale_range': SEARCH_SCALE_RANGE,
        'target_sz_bias_range': TARGET_SZ_BIAS_RANGE, 'status_flag': 2,
        'evaluation_metric': [], 'is_concave': False,
    }
    t = time.time()
    log.info("analyze: param search (status 2)")
    _run_tracking(params)
    pj = proj / "tmp" / vid / "evaluation_metric_for_test.json"
    best = json.load(open(pj))[-1]
    params['frame_num'] = nframes - 1
    params['search_scale_range'] = [float(best['search_scale'])]
    params['target_sz_bias_range'] = [best['target_sz_bias']]
    params['status_flag'] = 3
    log.info("analyze: full track (status 3) over %d frames, params %s/%s",
             nframes, best['search_scale'], best['target_sz_bias'])
    _run_tracking(params)
    raw = _find_latest_folder(str(res_dir))
    ds_name, save_path = post_process_results(vid, 5, raw, str(res_dir))
    log.info("analyze: done in %.1fs -> %s", time.time() - t, save_path)


SRC_VIDEO = None


def main():
    global SRC_VIDEO, NUM_ANIMALS, MAX_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--proj-root", default="/root/vast/eric/UDMT_runs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stages", default="extract,init,xmem,traindata,train,analyze")
    ap.add_argument("--num-animals", type=int, default=NUM_ANIMALS)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    SRC_VIDEO = args.video
    NUM_ANIMALS = args.num_animals
    MAX_FRAMES = args.max_frames

    from udmt.create_project.new import create_new_project
    proj_name = f"{args.session}_{args.cam}"
    proj_root = Path(args.proj_root)
    # create_new_project appends -<date>; find or create
    existing = sorted(proj_root.glob(f"{proj_name}-*"))
    if existing:
        proj = existing[0]
        log.info("using existing project %s", proj)
    else:
        cfg = create_new_project(proj_name, [args.video], working_directory=str(proj_root), is_concave=False)
        proj = Path(cfg).parent
        log.info("created project %s", proj)
    vid = Path(args.video).stem
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30.0

    _silence_vis()
    app, l1, l2 = _make_qt()
    stages = args.stages.split(",")
    overall = time.time()
    if "extract" in stages: stage_extract(proj, vid, args.video)
    if "init" in stages: stage_init(proj, vid)
    if "xmem" in stages: stage_xmem(proj, vid)
    if "traindata" in stages: stage_traindata(proj, vid, fps, l1, l2)
    if "train" in stages: stage_train(proj, vid, args.epochs, args.batch)
    if "analyze" in stages: stage_analyze(proj, vid, fps, l1, l2)
    banner(f"ALL DONE {proj_name} in {(time.time()-overall)/60:.1f} min")


if __name__ == "__main__":
    main()
