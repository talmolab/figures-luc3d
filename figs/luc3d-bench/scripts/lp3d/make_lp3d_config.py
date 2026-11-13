"""Generate an LP3D multi-view config YAML for our SLAP-2M dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

BENCH_CAMERAS = ("back", "backL", "mid", "midL", "top", "topL")
KEYPOINT_NAMES = ["Nose", "Ear_R", "Ear_L", "TTI", "TailTip", "Head", "Trunk",
                  "Tail_0", "Tail_1", "Tail_2", "Shoulder_left", "Shoulder_right",
                  "Haunch_left", "Haunch_right", "Neck"]


def make_config(data_dir: Path, image_size: int = 256, max_epochs: int = 100,
                train_batch_size: int = 8, num_gpus: int = 1) -> dict:
    cfg = {
        "data": {
            "image_resize_dims": {"height": image_size, "width": image_size},
            "data_dir": str(data_dir),
            "video_dir": str(data_dir / "videos"),
            "csv_file": [f"CollectedData_{c}.csv" for c in BENCH_CAMERAS],
            # No bbox file (whole image used)
            "view_names": list(BENCH_CAMERAS),
            "camera_params_file": None,  # auto-discovers per-session calibrations/<session>.toml
            "num_keypoints": len(KEYPOINT_NAMES),
            "keypoint_names": KEYPOINT_NAMES,
            "mirrored_column_matches": None,
            "columns_for_singleview_pca": None,
        },
        "training": {
            "imgaug": "dlc",
            "imgaug_3d": True,
            "train_batch_size": train_batch_size,
            "val_batch_size": 16,
            "test_batch_size": 16,
            "train_prob": 0.9,
            "val_prob": 0.05,
            "train_frames": 1,
            "num_gpus": num_gpus,
            "unfreezing_epoch": 5,
            "min_epochs": max_epochs,
            "max_epochs": max_epochs,
            "log_every_n_steps": 5,
            "check_val_every_n_epoch": 5,
            "ckpt_every_n_epochs": None,
            "rng_seed_data_pt": 0,
            "rng_seed_model_pt": 0,
            "optimizer": "AdamW",
            "optimizer_params": {"learning_rate": 5e-5},
            "lr_scheduler": "multisteplr",
            "lr_scheduler_params": {
                "multisteplr": {
                    "milestones": [max_epochs // 2, int(max_epochs * 0.75), int(max_epochs * 0.9)],
                    "gamma": 0.5,
                },
            },
            "uniform_heatmaps_for_nan_keypoints": True,
            "patch_mask": {
                "init_epoch": max(2, max_epochs // 5),
                "final_epoch": max_epochs,
                "init_ratio": 0.0,
                "final_ratio": 0.3,
            },
        },
        "model": {
            "losses_to_use": [],
            "backbone": "vits_dino",
            "model_type": "heatmap_multiview_transformer",
            "heatmap_loss_type": "mse",
            "model_name": "slap2m_mvt",
            "checkpoint": None,
        },
        "dali": {
            "base": {
                "train": {"sequence_length": 16},
                "predict": {"sequence_length": 32},
            },
            "context": {
                "train": {"batch_size": 8},
                "predict": {"sequence_length": 32},
            },
        },
        "losses": {
            "pca_multiview": {"log_weight": 5.0, "components_to_keep": 3, "epsilon": None},
            "pca_singleview": {"log_weight": 5.0, "components_to_keep": 0.99, "epsilon": None},
            "temporal": {"log_weight": 5.0, "epsilon": 20.0, "prob_threshold": 0.05},
            "supervised_reprojection_heatmap_mse": {"log_weight": 3.0},
        },
        "eval": {
            "predict_vids_after_training": False,
            "test_videos_directory": str(data_dir / "videos"),
            "save_vids_after_training": False,
            "colormap": "cool",
            "confidence_thresh_for_vid": 0.90,
        },
        "callbacks": {
            "anneal_weight": {
                "attr_name": "total_unsupervised_importance",
                "init_val": 0.0,
                "increase_factor": 0.01,
                "final_val": 1.0,
                "freeze_until_epoch": max_epochs // 2,
            },
        },
        "hydra": {
            "run": {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"},
            "sweep": {"dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}", "subdir": "${hydra.job.num}"},
        },
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--train-batch-size", type=int, default=8)
    ap.add_argument("--num-gpus", type=int, default=1)
    args = ap.parse_args()
    cfg = make_config(args.data_dir, args.image_size, args.max_epochs,
                      args.train_batch_size, args.num_gpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote config to {args.out}")


if __name__ == "__main__":
    main()
