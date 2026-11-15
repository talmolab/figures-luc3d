"""Candidate triangulators, to see whether anything beats aniposelib's plain DLT.

Where the 0.011 px gap probably comes from, and what each candidate does about it:

  dlt_px      LUC3D's current linear DLT: algebraic error, rows built in UNDISTORTED
              PIXEL coordinates. The algebraic residual each row contributes is
              scaled by that camera's focal length, so cameras are implicitly
              weighted by focal. On BMimica the focals differ by <1% so this is a
              small effect, but it is a bias with no justification.
  dlt_norm    The same DLT with rows built in NORMALISED camera coordinates
              (K^-1 applied first), which equalises the cameras. This is what
              aniposelib does, and it is the single cheapest hypothesis for why it
              is ahead.
  irls        Iteratively reweighted linear triangulation. Plain DLT minimises an
              ALGEBRAIC error whose per-camera weight is the point's depth in that
              camera; dividing each camera's rows by its current depth and
              re-solving converges to the GEOMETRIC minimiser. Two or three
              iterations is usually enough. Still closed-form per iteration.
  gn          Gauss-Newton on SQUARED reprojection error in pixels -- the maximum
              likelihood estimate under i.i.d. isotropic Gaussian pixel noise, which
              is exactly the noise model a held-out-view reprojection metric
              rewards. NOTE this is the one thing LUC3D's shipped refinement is not:
              `triangulatePointBA` minimises a soft-L1 loss with an L1 polish, which
              is deliberately ROBUST and therefore deliberately not the least-squares
              optimum. On data where only 1.6% of keypoints have any view off by
              >=10 px (Fig 4c), robustness costs more than it buys.
  gn_conf     Gauss-Newton weighted by the detector's own per-point confidence.
              Every method above throws that information away. A view whose
              detection the network was unsure about should pull the solution less
              than one it was certain about, and SLEAP hands us that number for free
              in `point_scores`.
  huber       Gauss-Newton with a Huber loss, as the control for the claim above:
              if robustness is what costs `luc3d_refined` its accuracy, huber should
              land between gn and the shipped refinement.

Everything is vectorised over all N points at once; these run over millions of
keypoints, so a per-point Python loop is not an option.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
def _solve_dlt(A: np.ndarray) -> np.ndarray:
    """A: (N, 2C, 4) -> (N, 3). Null space via SVD, batched."""
    # Guard rows that are all-NaN (missing views) by zeroing them: a zero row
    # contributes nothing to the null space, which is exactly "this view abstains".
    A = np.nan_to_num(A, nan=0.0)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    X = Vt[:, -1, :]
    w = X[:, 3:4]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = X[:, :3] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def _rows(obs: np.ndarray, P: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """obs (C,N,2), P (C,3,4) -> A (N, 2C, 4), optionally row-weighted by (C,N)."""
    C, N, _ = obs.shape
    A = np.empty((N, 2 * C, 4))
    for c in range(C):
        x = obs[c, :, 0][:, None]
        y = obs[c, :, 1][:, None]
        A[:, 2 * c] = x * P[c, 2][None, :] - P[c, 0][None, :]
        A[:, 2 * c + 1] = y * P[c, 2][None, :] - P[c, 1][None, :]
        if weights is not None:
            w = weights[c][:, None]
            A[:, 2 * c] *= w
            A[:, 2 * c + 1] *= w
    return A


def dlt(obs: np.ndarray, P: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    return _solve_dlt(_rows(obs, P, None))


def dlt_conf(obs: np.ndarray, P: np.ndarray, conf: np.ndarray) -> np.ndarray:
    return _solve_dlt(_rows(obs, P, conf))


def irls(obs: np.ndarray, P: np.ndarray, conf: np.ndarray | None = None,
         iters: int = 5) -> np.ndarray:
    """Reweight each camera's rows by 1/depth and re-solve. Converges from the
    algebraic optimum to the geometric one."""
    X = _solve_dlt(_rows(obs, P, conf))
    C = obs.shape[0]
    for _ in range(iters):
        ok = np.all(np.isfinite(X), axis=1)
        Xh = np.concatenate([np.where(ok[:, None], X, 0.0),
                             np.ones((len(X), 1))], axis=1)          # (N,4)
        depth = np.stack([Xh @ P[c, 2] for c in range(C)], axis=0)    # (C,N)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / depth
        w[~np.isfinite(w)] = 0.0
        if conf is not None:
            w = w * conf
        Xn = _solve_dlt(_rows(obs, P, np.abs(w)))
        X = np.where(np.isfinite(Xn), Xn, X)
    return X


# --------------------------------------------------------------------------- #
def _project_batch(X: np.ndarray, P: np.ndarray):
    """X (N,3), P (3,4) -> uv (N,2), and the (N,2,3) Jacobian d(uv)/dX."""
    Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    num = Xh @ P.T                                        # (N,3)
    z = num[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = num[:, 0] / z
        v = num[:, 1] / z
        J = np.empty((len(X), 2, 3))
        for k in range(3):
            J[:, 0, k] = (P[0, k] - u * P[2, k]) / z
            J[:, 1, k] = (P[1, k] - v * P[2, k]) / z
    return np.stack([u, v], axis=1), J


def gauss_newton(obs: np.ndarray, P: np.ndarray, conf: np.ndarray | None = None,
                 iters: int = 12, loss: str = "l2", huber_delta: float = 4.0,
                 seed: np.ndarray | None = None) -> np.ndarray:
    """Minimise sum_c w_c * rho(||proj_c(X) - x_c||^2) over X, in PIXELS."""
    C, N, _ = obs.shape
    X = irls(obs, P, conf) if seed is None else seed.copy()
    valid = np.isfinite(obs).all(axis=2)                              # (C,N)

    for _ in range(iters):
        H = np.zeros((N, 3, 3))
        g = np.zeros((N, 3))
        good = np.all(np.isfinite(X), axis=1)
        Xs = np.where(good[:, None], X, 0.0)
        for c in range(C):
            uv, J = _project_batch(Xs, P[c])
            r = uv - obs[c]                                           # (N,2)
            m = valid[c] & good & np.isfinite(r).all(axis=1) & np.isfinite(J).all(axis=(1, 2))
            if not m.any():
                continue
            w = np.ones(N)
            if loss == "huber":
                nr = np.linalg.norm(r, axis=1)
                w = np.where(nr > huber_delta, huber_delta / np.maximum(nr, 1e-9), 1.0)
            if conf is not None:
                w = w * conf[c]
            w = np.where(m, w, 0.0)
            Jw = J * w[:, None, None]
            H += np.einsum("nij,nik->njk", Jw, J)
            g += np.einsum("nij,ni->nj", Jw, r)
        # Levenberg damping keeps a rank-deficient point (2 views, near-degenerate
        # geometry) from exploding instead of simply not improving.
        H += np.eye(3)[None] * 1e-9
        try:
            step = np.linalg.solve(H, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        step = np.where(np.isfinite(step), step, 0.0)
        X = X - step
    return X


# --------------------------------------------------------------------------- #
VARIANTS = {
    "v_dlt":        lambda o, P, c: dlt(o, P),
    "v_dlt_conf":   lambda o, P, c: dlt_conf(o, P, c),
    "v_irls":       lambda o, P, c: irls(o, P),
    "v_irls_conf":  lambda o, P, c: irls(o, P, c),
    "v_gn":         lambda o, P, c: gauss_newton(o, P, None),
    "v_gn_conf":    lambda o, P, c: gauss_newton(o, P, c),
    "v_huber":      lambda o, P, c: gauss_newton(o, P, None, loss="huber"),
}
