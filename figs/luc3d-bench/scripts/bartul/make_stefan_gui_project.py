#!/usr/bin/env python
"""Build a .slp project file referencing all runnable BMimica videos + the
15-node skeleton, so Stefan's optimized models
(/root/vast/eric/stefan-bartul-model/OptimizedModels/) can be selected and run
interactively from the SLEAP GUI's Predict dialog (Runs > Run Inference,
pointing at the centroid + centered-instance model directories).

No labeled frames -- just skeleton + video references, so this is a lightweight
"open project, pick models, hit predict" starting point, not a labels export.

Output: /root/vast/eric/stefan-bartul-model/bartul_gui_project.slp
"""
import glob
import os

import sleap_io as sio

ROOT = "/root/vast/eric/BMimica"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]
OUT = "/root/vast/eric/stefan-bartul-model/bartul_gui_project.slp"
SKELETON_SRC = f"{ROOT}/20250827_141755/21241563/21241563-20250827141804.slp"


def runnable_sessions():
    out = []
    for s in sorted(glob.glob(ROOT + "/*")):
        sid = os.path.basename(s)
        if not (os.path.isdir(s) and sid[0].isdigit()):
            continue
        if not glob.glob(s + "/calibration/*_calibration.toml"):
            continue
        if not glob.glob(s + "/*points3d*.h5"):
            continue
        if sum(1 for c in SER if glob.glob(f"{s}/{c}/*.mp4")) != 5:
            continue
        out.append(sid)
    return out


def main():
    skeleton = sio.load_slp(SKELETON_SRC, open_videos=False).skeletons[0]
    print(f"skeleton: {len(skeleton.nodes)} nodes from {SKELETON_SRC}")

    sids = runnable_sessions()
    videos = []
    for sid in sids:
        for cam in SER:
            matches = glob.glob(f"{ROOT}/{sid}/{cam}/*.mp4")
            if not matches:
                continue
            videos.append(sio.Video.from_filename(matches[0], grayscale=True, keep_open=False))
    print(f"videos: {len(videos)} from {len(sids)} sessions x {len(SER)} cams")

    labels = sio.Labels(videos=videos, skeletons=[skeleton], labeled_frames=[])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    labels.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
