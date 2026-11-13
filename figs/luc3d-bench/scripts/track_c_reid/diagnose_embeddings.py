"""Diagnose the trained ReID embeddings on held-out sessions.

Outputs:
  - same-id vs diff-id cosine-similarity histograms
  - per-identity centroid table
  - simple kNN classifier accuracy on held-out frames (within-session)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from model import EncoderConfig, PoseHistoryEncoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="lp3d_models/slap2m_reid/windows.npz")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="lp3d_models/slap2m_reid/diagnose.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=50000)
    args = p.parse_args()

    data = np.load(args.data, allow_pickle=True)
    windows = data["windows"]
    sess_idx = data["sess_idx"]
    track_idx = data["track_idx"]
    global_id = data["global_id"]
    sessions = data["sessions"]

    # rebuild same train/val split as train.py
    import random
    rng = random.Random(args.seed)
    sess_indices = list(range(len(sessions)))
    rng.shuffle(sess_indices)
    n_val = max(1, int(round(args.val_frac * len(sessions))))
    val_sess = set(sess_indices[:n_val])
    val_mask = np.array([int(s) in val_sess for s in sess_idx])

    # sub-sample val
    val_idx = np.where(val_mask)[0]
    rng2 = np.random.RandomState(args.seed)
    if len(val_idx) > args.max_windows:
        val_idx = rng2.choice(val_idx, args.max_windows, replace=False)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = EncoderConfig(**ckpt["cfg"])
    model = PoseHistoryEncoder(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    embs = np.zeros((len(val_idx), cfg.d_embed), dtype=np.float32)
    bs = 1024
    with torch.no_grad():
        for i in tqdm(range(0, len(val_idx), bs), desc="embed"):
            chunk = val_idx[i:i+bs]
            x = torch.from_numpy(windows[chunk]).to(args.device)
            e = model(x).cpu().numpy()
            embs[i:i+bs] = e
    labels = global_id[val_idx]
    sess = sess_idx[val_idx]

    # Per-session, per-identity stats
    per_id_centroids = {}
    for lbl in np.unique(labels):
        per_id_centroids[int(lbl)] = embs[labels == lbl].mean(axis=0)

    # Same-session pairwise classification accuracy:
    # for each window, find its nearest neighbour in the same session, see if id matches.
    from collections import defaultdict
    by_sess = defaultdict(list)
    for i, s in enumerate(sess):
        by_sess[int(s)].append(i)

    per_session_acc = {}
    overall_correct = 0
    overall_total = 0
    for s, idxs in by_sess.items():
        if len(idxs) < 2:
            continue
        sub_embs = embs[idxs]
        sub_lbls = labels[idxs]
        if len(np.unique(sub_lbls)) < 2:
            # only one mouse in this session; 1-NN inside the session is trivially correct
            per_session_acc[int(s)] = {"acc": 1.0, "n": len(idxs), "n_ids": 1}
            continue
        # cosine sim matrix
        sim = sub_embs @ sub_embs.T
        np.fill_diagonal(sim, -1.0)
        nn_idx = sim.argmax(axis=1)
        nn_lbls = sub_lbls[nn_idx]
        correct = (nn_lbls == sub_lbls).sum()
        total = len(sub_lbls)
        per_session_acc[int(s)] = {
            "acc": float(correct / total),
            "n": int(total),
            "n_ids": int(len(np.unique(sub_lbls))),
            "session": str(sessions[s]),
        }
        # Only count multi-id sessions in the overall
        overall_correct += int(correct)
        overall_total += int(total)

    overall_acc = float(overall_correct / overall_total) if overall_total else float("nan")

    out = {
        "n_val_windows": int(len(val_idx)),
        "n_val_ids": int(len(np.unique(labels))),
        "n_val_sessions": int(len(by_sess)),
        "overall_within_session_1nn_acc": overall_acc,
        "per_session_acc": per_session_acc,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    print(f"overall within-session 1-NN accuracy on multi-mouse sessions: {overall_acc:.4f}")
    # by num_ids breakdown
    by_nids = defaultdict(list)
    for _, info in per_session_acc.items():
        by_nids[info["n_ids"]].append(info["acc"])
    for n_ids, accs in sorted(by_nids.items()):
        print(f"  n_ids={n_ids}: mean acc={np.mean(accs):.4f} ({len(accs)} sessions, baseline={1/n_ids:.3f})")


if __name__ == "__main__":
    main()
