"""Load the OSNet appearance ("color") encoder via boxmot, with API-drift fallback.

The person-reid tutorial accesses `StrongSort(reid_weights=..., device=..., half=...).encoder`.
boxmot has renamed this across versions, so encoder access is isolated here: a future
rename is a one-line fix. Returns a callable that maps a batch tensor (B,3,256,128) →
(B, 512) appearance features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models/reid/osnet_x0_25_imagenet.pt"

# ImageNet normalization, standard person-ReID input size (H, W) = (256, 128).
_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_encoder(weights: Path, device: torch.device, half: bool):
    """Return a model whose forward maps a normalized (B,3,H,W) tensor → (B, D).

    Tries the encoder backends across boxmot versions (layout drifts between releases).
    """
    errors = []
    # boxmot ~10.x: the ReID MultiBackend nn.Module (forward takes normalized crops).
    try:
        from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
        return ReIDDetectMultiBackend(weights=weights, device=device, fp16=half)
    except Exception as e:  # noqa: BLE001
        errors.append(f"ReIDDetectMultiBackend: {e!r}")
    # Newer boxmot: appearance auto-backend.
    try:
        from boxmot.appearance.reid_auto_backend import ReidAutoBackend
        rab = ReidAutoBackend(weights=weights, device=device, half=half)
        return getattr(rab, "model", rab)
    except Exception as e:  # noqa: BLE001
        errors.append(f"ReidAutoBackend: {e!r}")
    # Tutorial path: StrongSort.encoder.
    try:
        from boxmot.trackers.strongsort.strongsort import StrongSort
        ss = StrongSort(reid_weights=weights, device=device, half=half)
        enc = getattr(ss, "encoder", None) or getattr(ss, "model", None)
        if enc is not None:
            return enc
    except Exception as e:  # noqa: BLE001
        errors.append(f"StrongSort: {e!r}")
    raise RuntimeError("could not load boxmot ReID encoder:\n  " + "\n  ".join(errors))


class ReidEncoder:
    """Thin wrapper: numpy RGB crops → L2-normalized appearance embeddings."""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str | None = None, half: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = half and self.device.type == "cuda"
        self.encoder = _load_encoder(Path(weights), self.device, self.half)

    @torch.inference_mode()
    def embed(self, crops_rgb: list[np.ndarray], batch_size: int = 256) -> np.ndarray:
        """crops_rgb: list of (256,128,3) uint8 RGB → (N, D) float32, L2-normalized."""
        if not crops_rgb:
            return np.zeros((0, 512), dtype=np.float32)
        feats = []
        for i in range(0, len(crops_rgb), batch_size):
            batch = torch.stack([_TF(c) for c in crops_rgb[i:i + batch_size]]).to(self.device)
            if self.half:
                batch = batch.half()
            out = self.encoder(batch)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                out = out.float().cpu().numpy()
            feats.append(np.asarray(out, dtype=np.float32))
        f = np.vstack(feats).astype(np.float32)
        f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
        return f
