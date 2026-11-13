"""Train the pose-history ReID encoder with SupCon-style contrastive loss.

Splits sessions into train/val. Each batch is a P-K sampler: P identities
each contributing K windows. Train with SupCon (intra-batch all-vs-all).
Validate by computing same-id vs diff-id cosine similarity distributions
on held-out sessions and reporting AUROC and ROC@FAR=10%.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from model import EncoderConfig, PoseHistoryEncoder, supcon_loss


class WindowDataset(Dataset):
    def __init__(self, windows, global_ids, sess_idx, track_idx):
        self.windows = windows
        self.global_ids = global_ids
        self.sess_idx = sess_idx
        self.track_idx = track_idx

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            int(self.global_ids[idx]),
            int(self.sess_idx[idx]),
            int(self.track_idx[idx]),
        )


class PKSampler(Sampler):
    """Sample P identities x K windows per batch."""

    def __init__(self, global_ids: np.ndarray, P: int, K: int, num_batches: int, seed: int = 0):
        self.global_ids = global_ids
        self.P = P
        self.K = K
        self.num_batches = num_batches
        # group indices by identity
        self.by_id = {}
        for i, g in enumerate(global_ids):
            self.by_id.setdefault(int(g), []).append(i)
        self.ids = [g for g, idxs in self.by_id.items() if len(idxs) >= 2]
        if len(self.ids) < P:
            print(f"  WARN: only {len(self.ids)} ids with >=2 windows; reducing P to {len(self.ids)}")
            self.P = len(self.ids)
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_ids = self.rng.sample(self.ids, self.P)
            batch = []
            for gid in chosen_ids:
                idxs = self.by_id[gid]
                if len(idxs) >= self.K:
                    pick = self.rng.sample(idxs, self.K)
                else:
                    pick = [self.rng.choice(idxs) for _ in range(self.K)]
                batch.extend(pick)
            yield batch

    def __len__(self):
        return self.num_batches


def split_sessions(num_sessions: int, val_frac: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    sess_indices = list(range(num_sessions))
    rng.shuffle(sess_indices)
    n_val = max(1, int(round(val_frac * num_sessions)))
    val = set(sess_indices[:n_val])
    train = set(sess_indices[n_val:])
    return train, val


def evaluate(model, val_loader, device, max_pairs: int = 200_000):
    model.eval()
    embs = []
    labels = []
    sess_idxs = []
    with torch.no_grad():
        for batch in val_loader:
            win, gid, si, _ = batch
            win = win.to(device, non_blocking=True)
            emb = model(win)
            embs.append(emb.cpu().numpy())
            labels.append(np.array(gid))
            sess_idxs.append(np.array(si))
    embs = np.concatenate(embs)
    labels = np.concatenate(labels)
    sess_idxs = np.concatenate(sess_idxs)

    rng = np.random.RandomState(0)
    N = len(embs)
    same_sims = []
    diff_sims_within = []   # different identity, SAME session
    diff_sims_across = []   # different identity, ANY session
    target = max_pairs

    # pos pairs
    by_label = {}
    for i, lbl in enumerate(labels):
        by_label.setdefault(int(lbl), []).append(i)
    pos_count = 0
    for lbl, idxs in by_label.items():
        if len(idxs) < 2:
            continue
        # sample up to ~target/len(by_label) pairs from each id
        n_pairs = min(len(idxs), max(2, target // max(1, len(by_label))))
        a = rng.choice(idxs, n_pairs)
        b = rng.choice(idxs, n_pairs)
        keep = a != b
        a = a[keep]; b = b[keep]
        s = (embs[a] * embs[b]).sum(axis=1)
        same_sims.append(s)
        pos_count += len(s)
        if pos_count >= target:
            break
    same_sims = np.concatenate(same_sims) if same_sims else np.array([])

    # negative pairs: split by session
    n_target_neg = max(1, len(same_sims))
    n_within = 0
    n_across = 0
    while n_within < n_target_neg and len(diff_sims_within) < n_target_neg:
        a = rng.randint(0, N)
        b = rng.randint(0, N)
        if labels[a] == labels[b]:
            continue
        if sess_idxs[a] == sess_idxs[b]:
            diff_sims_within.append(float(np.dot(embs[a], embs[b])))
            n_within += 1
        if len(diff_sims_within) >= n_target_neg:
            break
    while n_across < n_target_neg:
        a = rng.randint(0, N)
        b = rng.randint(0, N)
        if labels[a] == labels[b]:
            continue
        if sess_idxs[a] != sess_idxs[b]:
            diff_sims_across.append(float(np.dot(embs[a], embs[b])))
            n_across += 1
        if len(diff_sims_across) >= n_target_neg:
            break

    diff_sims_within = np.array(diff_sims_within)
    diff_sims_across = np.array(diff_sims_across)

    def auroc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        scores = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        # rank-based AUROC
        order = np.argsort(-scores)
        y = y[order]
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        tp = 0
        fp = 0
        prev_tpr = 0.0
        prev_fpr = 0.0
        auc = 0.0
        for v in y:
            if v == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_tpr = tpr
            prev_fpr = fpr
        return float(auc)

    return {
        "n_val_windows": int(N),
        "n_val_ids": int(len(set(labels.tolist()))),
        "same_mean": float(same_sims.mean()) if len(same_sims) else float("nan"),
        "same_std": float(same_sims.std()) if len(same_sims) else float("nan"),
        "diff_within_mean": float(diff_sims_within.mean()) if len(diff_sims_within) else float("nan"),
        "diff_within_std": float(diff_sims_within.std()) if len(diff_sims_within) else float("nan"),
        "diff_across_mean": float(diff_sims_across.mean()) if len(diff_sims_across) else float("nan"),
        "diff_across_std": float(diff_sims_across.std()) if len(diff_sims_across) else float("nan"),
        "auroc_within": auroc(same_sims, diff_sims_within),
        "auroc_across": auroc(same_sims, diff_sims_across),
        "n_same_pairs": int(len(same_sims)),
        "n_diff_within": int(len(diff_sims_within)),
        "n_diff_across": int(len(diff_sims_across)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="lp3d_models/slap2m_reid/windows.npz")
    p.add_argument("--out", default="lp3d_models/slap2m_reid")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batches-per-epoch", type=int, default=200)
    p.add_argument("--P", type=int, default=16, help="ids per batch")
    p.add_argument("--K", type=int, default=8, help="windows per id")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--d-embed", type=int, default=128)
    p.add_argument("--n-block", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--velocity", action="store_true", default=True)
    p.add_argument("--no-velocity", dest="velocity", action="store_false")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    windows = data["windows"]            # (N, T, K, 3)
    sess_idx = data["sess_idx"]
    track_idx = data["track_idx"]
    global_id = data["global_id"]
    sessions = data["sessions"]
    num_animals = data["num_animals"]
    print(f"loaded {windows.shape[0]} windows over {len(sessions)} sessions, {len(np.unique(global_id))} ids")

    # session-level train/val split
    train_sess, val_sess = split_sessions(len(sessions), args.val_frac, seed=args.seed)
    train_mask = np.array([int(s) in train_sess for s in sess_idx])
    val_mask = ~train_mask

    # also: train only on identities with >=2 windows
    train_ids = global_id[train_mask]
    print(f"train sessions: {len(train_sess)} ({train_mask.sum()} windows, {len(np.unique(train_ids))} ids)")
    print(f"val   sessions: {len(val_sess)} ({val_mask.sum()} windows, {len(np.unique(global_id[val_mask]))} ids)")

    train_set = WindowDataset(
        windows[train_mask], global_id[train_mask], sess_idx[train_mask], track_idx[train_mask]
    )
    val_set = WindowDataset(
        windows[val_mask], global_id[val_mask], sess_idx[val_mask], track_idx[val_mask]
    )

    pk_sampler = PKSampler(
        global_ids=global_id[train_mask], P=args.P, K=args.K,
        num_batches=args.batches_per_epoch, seed=args.seed,
    )
    train_loader = DataLoader(
        train_set, batch_sampler=pk_sampler, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
    )

    cfg = EncoderConfig(
        window_len=windows.shape[1],
        n_keypoints=windows.shape[2],
        in_dim=windows.shape[3],
        d_embed=args.d_embed,
        n_block=args.n_block,
        n_head=args.n_head,
        dropout=args.dropout,
        use_velocity_channel=args.velocity,
    )
    model = PoseHistoryEncoder(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.batches_per_epoch)

    # baseline: random model
    print("baseline (random init) val metrics:")
    base_metrics = evaluate(model, val_loader, args.device)
    for k, v in base_metrics.items():
        print(f"  {k}: {v}")

    history = {"baseline": base_metrics, "epochs": []}
    best_auroc = -1.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            win, gid, _, _ = batch
            win = win.to(args.device, non_blocking=True)
            labels = torch.tensor(gid, dtype=torch.long, device=args.device)
            emb = model(win)
            loss = supcon_loss(emb, labels, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))

        ep_metrics = evaluate(model, val_loader, args.device)
        ep_metrics["epoch"] = ep
        ep_metrics["train_loss"] = mean_loss
        history["epochs"].append(ep_metrics)
        print(
            f"ep {ep:02d} loss={mean_loss:.4f} same_mu={ep_metrics['same_mean']:.3f} "
            f"diff_within_mu={ep_metrics['diff_within_mean']:.3f} "
            f"diff_across_mu={ep_metrics['diff_across_mean']:.3f} "
            f"auroc_within={ep_metrics['auroc_within']:.3f} "
            f"auroc_across={ep_metrics['auroc_across']:.3f}"
        )

        if ep_metrics["auroc_within"] > best_auroc:
            best_auroc = ep_metrics["auroc_within"]
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "metrics": ep_metrics,
                "args": vars(args),
            }, out_dir / "best.pt")

    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "metrics": history["epochs"][-1] if history["epochs"] else base_metrics,
        "args": vars(args),
    }, out_dir / "last.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"done. best auroc_within = {best_auroc:.3f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
