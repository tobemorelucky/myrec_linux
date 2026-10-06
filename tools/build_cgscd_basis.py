# -*- coding: UTF-8 -*-
"""
Build the CGSCD shared subspace basis from LLM and collaborative item views.

Chapter 3 (Architecture-First): the shared/complementary split must come from
cross-view association, not from an arbitrary PCA-variance cut.

Inputs
  Z = frozen LLM item embedding   (llm_table_pca1536.pkl, n_items rows, row 0 = 0)
  C = collaborative item embed    (itm_emb_pomrec.pkl, n_items-1 rows, no row 0)

Two estimation methods (neither is declared the winner a priori):

  crosscov : M = Zc^T Cc / N  ->  SVD  ->  U[:, :r]      (orthonormal, no whitening)
  cca      : regularized CCA on (Zc, Cc); canonical directions in Z space are
             A = W_z @ U[:, :r] with W_z the Z-whitening operator, then
             QR-orthonormalized so the basis spans the same subspace but is
             Euclidean-orthonormal (required for a proper orthogonal residual).

Output: data/<dataset>/handled/cgscd_basis_<method>_r<rank>.pkl
  {
    'U_r'            : float32 [d_llm, r]   orthonormal basis of the shared subspace
    'singular_values': float32 [r]          singular values / canonical correlations
    'z_mean'         : float32 [d_llm]
    'c_mean'         : float32 [d_cf]
    'method'         : 'crosscov' | 'cca'
    'rank'           : int
    'ridge'          : float (cca only)
    'source'         : {'z_path','z_sha1','c_path','c_sha1','n_items'}
    'diagnostics'    : {...}
  }

Usage:
  python tools/build_cgscd_basis.py --dataset beauty --method crosscov --rank 32
  python tools/build_cgscd_basis.py --dataset beauty --method cca --rank 32
"""

import argparse
import hashlib
import os
import pickle
import time

import numpy as np


# =========================
#  IO helpers
# =========================

def sha1_of(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_views(dataset, llm_path="", cf_path="", verbose=True):
    """Load and align (Z, C). Returns Zc [N, d_llm], Cc [N, d_cf], z_mean, c_mean."""
    llm_path = llm_path or f"./data/{dataset}/handled/llm_table_pca1536.pkl"
    cf_path = cf_path or f"./data/{dataset}/handled/itm_emb_pomrec.pkl"

    for p in (llm_path, cf_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing view file: {p}")

    Z = np.asarray(pickle.load(open(llm_path, "rb")), dtype=np.float32)
    C = np.asarray(pickle.load(open(cf_path, "rb")), dtype=np.float32)

    if Z.ndim != 2 or C.ndim != 2:
        raise ValueError(f"views must be 2D, got Z{Z.shape} C{C.shape}")

    # Align: Z row 0 is padding; C has no padding row.
    if Z.shape[0] == C.shape[0] + 1:
        if not np.allclose(Z[0], 0.0):
            raise ValueError("Z has n_items-1+1 rows but row 0 is not zero padding")
        Zv = Z[1:]
    elif Z.shape[0] == C.shape[0]:
        Zv = Z
        if verbose:
            print("[warn] Z and C have the same row count; assuming no padding row")
    else:
        raise ValueError(
            f"row mismatch: Z {Z.shape[0]} vs C {C.shape[0]} "
            f"(expected Z = C + 1 with zero padding row)"
        )

    if not np.isfinite(Zv).all():
        raise ValueError(f"Z contains NaN/Inf ({int((~np.isfinite(Zv)).sum())} entries)")
    if not np.isfinite(C).all():
        raise ValueError(f"C contains NaN/Inf ({int((~np.isfinite(C)).sum())} entries)")

    z_mean = Zv.mean(axis=0)
    c_mean = C.mean(axis=0)
    Zc = Zv - z_mean
    Cc = C - c_mean

    if verbose:
        print(f"  Z: {Z.shape} -> view {Zv.shape}   C: {C.shape}")
        print(f"  ||Zc||^2={float((Zc.astype(np.float64)**2).sum()):.4e}  "
              f"||Cc||^2={float((Cc.astype(np.float64)**2).sum()):.4e}")
    return Zc, Cc, z_mean.astype(np.float32), c_mean.astype(np.float32), llm_path, cf_path


# =========================
#  Core estimators
# =========================

def _top_r_svd(M, r):
    """Truncated SVD of a [d_llm, d_cf] matrix. d_cf is small, full SVD is fine."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, S.shape[0])
    return U[:, :r], S, Vt


def _cca_whitener(S, ridge_rel, name):
    """Regularized whitening operator W = V diag(1/sqrt(w)) for covariance S.

    lambda = ridge_rel * trace(S)/d  (scale-free); eigenvalues are floored.
    """
    d = S.shape[0]
    lam = ridge_rel * float(np.trace(S)) / d
    S_reg = S + lam * np.eye(d)
    w, V = np.linalg.eigh(S_reg)          # ascending
    w = np.maximum(w, 1e-12)
    W = V * (1.0 / np.sqrt(w))            # [d, d]  (column scaling)
    return W, lam, float(w.min()), float(w.max())


def estimate_basis(Zc, Cc, method, rank, cca_ridge=1e-2, verbose=True, seed=42):
    """Return (U_r [d_llm,r], singular_values [r], info dict)."""
    N, d_llm = Zc.shape
    d_cf = Cc.shape[1]

    Zc64 = Zc.astype(np.float64)
    Cc64 = Cc.astype(np.float64)
    M0 = (Zc64.T @ Cc64) / N                       # cross-covariance [d_llm, d_cf]

    info = {}
    if method == "null":
        # CONTROL: zero basis => shared branch receives a constant, giving a
        # "complement branch only" model with the same parameter budget.
        U_r = np.zeros((d_llm, rank), dtype=np.float64)
        S = np.zeros(rank, dtype=np.float64)
        info["control"] = "no shared information reaches the shared branch"
    elif method == "random":
        # CONTROL: a fixed random orthonormal subspace of the same rank, i.e.
        # "would any 32 directions work as well as the cross-view ones?"
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d_llm, rank)))
        U_r = Q
        S = np.zeros(rank, dtype=np.float64)
        info["control"] = f"random orthonormal subspace (seed={seed})"
    elif method == "crosscov":
        U_r, S, _ = _top_r_svd(M0, rank)
        info["whitened"] = False
    elif method == "cca":
        Sz = (Zc64.T @ Zc64) / N
        Sc = (Cc64.T @ Cc64) / N
        Wz, lam_z, zmin, zmax = _cca_whitener(Sz, cca_ridge, "Z")
        Wc, lam_c, cmin, cmax = _cca_whitener(Sc, cca_ridge, "C")
        M = Wz.T @ M0 @ Wc                          # [d_llm, d_cf], whitened cross-cov
        U, S, _ = _top_r_svd(M, rank)
        A = Wz @ U[:, :rank]                        # canonical directions in Z space
        # Orthonormalize so the residual projection is well defined.
        Q, R = np.linalg.qr(A)
        rdiag = np.abs(np.diag(R))
        info.update(
            whitened=True, cca_ridge_rel=cca_ridge,
            cca_lambda_z=lam_z, cca_lambda_c=lam_c,
            cca_cond_z=zmax / max(zmin, 1e-300), cca_cond_c=cmax / max(cmin, 1e-300),
            qr_r_min=float(rdiag.min()), qr_r_max=float(rdiag.max()),
            qr_rank_deficient=bool(rdiag.min() < 1e-8 * max(rdiag.max(), 1e-30)),
            canonical_direction_norms=float(np.linalg.norm(A, axis=0).mean()),
        )
        if info["qr_rank_deficient"] and verbose:
            print("[warn] CCA canonical directions are near rank-deficient; "
                  "the orthonormalized basis is not uniquely determined. "
                  "Increase --cca_ridge or lower --rank.")
        U_r = Q
    else:
        raise ValueError(f"unknown method: {method}")

    return U_r.astype(np.float64), np.asarray(S[:rank], dtype=np.float64), info


# =========================
#  Diagnostics
# =========================

def _var_share(Zc64, U_r):
    """Fraction of total Z variance that lies in span(U_r)."""
    tot = float((Zc64 ** 2).sum())
    if tot <= 0:
        return float("nan")
    proj = Zc64 @ U_r                       # [N, r]
    return float((proj ** 2).sum() / tot)


def _ridge_r2(Zc64, Cc64, ridge_rel=1e-2):
    """Total linear predictability of Z from C (ridge regression R^2)."""
    N, d_cf = Cc64.shape
    Sc = (Cc64.T @ Cc64) / N
    lam = ridge_rel * float(np.trace(Sc)) / d_cf
    G = Cc64.T @ Cc64 + lam * N * np.eye(d_cf)
    W = np.linalg.solve(G, Cc64.T @ Zc64)   # [d_cf, d_llm]
    Zhat = Cc64 @ W
    tot = float((Zc64 ** 2).sum())
    return float((Zhat ** 2).sum() / tot), Zhat


def compute_diagnostics(Zc, Cc, U_r, S, verbose=True):
    Zc64 = Zc.astype(np.float64)
    Cc64 = Cc.astype(np.float64)
    N, d_llm = Zc64.shape

    diag = {}
    # --- spectrum ---
    s2 = S ** 2
    diag["singular_values"] = S.astype(np.float32)
    diag["sigma_energy_cum"] = (np.cumsum(s2) / max(s2.sum(), 1e-300)).astype(np.float32)

    # --- how much of Z lives in the shared subspace ---
    diag["z_var_share_r"] = _var_share(Zc64, U_r)
    diag["z_var_share_private"] = 1.0 - diag["z_var_share_r"]

    # --- overall linear predictability of Z from C ---
    r2_total, Zhat = _ridge_r2(Zc64, Cc64)
    diag["cf_r2_total"] = r2_total

    # --- does the shared subspace capture where the CF information lives? ---
    tot = float((Zc64 ** 2).sum())
    pred_tot = float((Zhat ** 2).sum())
    pred_proj = Zhat @ U_r
    diag["cf_r2_in_shared"] = float((pred_proj ** 2).sum() / max(tot, 1e-300))
    diag["pred_share_in_shared"] = float(
        (pred_proj ** 2).sum() / max(pred_tot, 1e-300)
    )

    # --- how much of C is explained by the shared Z coordinates ---
    # Least-squares map from the r shared coordinates to C; R^2 tells us whether
    # the shared subspace really carries the collaborative signal.
    Zs = Zc64 @ U_r                                        # [N, r]
    Gram = Zs.T @ Zs + 1e-8 * np.eye(U_r.shape[1])
    B = np.linalg.solve(Gram, Zs.T @ Cc64)                 # [r, d_cf]
    Chat = Zs @ B
    diag["c_var_share_r"] = float((Chat ** 2).sum() / max((Cc64 ** 2).sum(), 1e-300))

    # --- baseline comparison: the PCA first-512 slice that ASPCF assumes ---
    for cut in (32, 512):
        if cut <= d_llm:
            U_cut = np.zeros((d_llm, cut), dtype=np.float64)
            U_cut[np.arange(cut), np.arange(cut)] = 1.0
            share = _var_share(Zc64, U_cut)
            r2c, Zhatc = _ridge_r2(Zc64[:, :cut], Cc64)
            diag[f"pca{cut}_var_share"] = share
            diag[f"pca{cut}_cf_r2"] = r2c

    # --- alignment between the shared subspace and the PCA-512 slice ---
    U512 = np.zeros((d_llm, min(512, d_llm)), dtype=np.float64)
    U512[np.arange(U512.shape[1]), np.arange(U512.shape[1])] = 1.0
    overlap = U_r.T @ U512
    diag["overlap_shared_with_pca512"] = float(
        (overlap ** 2).sum() / U_r.shape[1]
    )  # mean squared cosine of shared directions against the PCA-512 subspace

    if verbose:
        print("  --- diagnostics ---")
        print(f"    z_var_share_r          = {diag['z_var_share_r']:.4f}")
        print(f"    cf_r2_total            = {diag['cf_r2_total']:.4f}")
        print(f"    cf_r2_in_shared        = {diag['cf_r2_in_shared']:.4f}")
        print(f"    pred_share_in_shared   = {diag['pred_share_in_shared']:.4f}")
        print(f"    c_var_share_r          = {diag['c_var_share_r']:.4f}")
        print(f"    pca512_var_share       = {diag.get('pca512_var_share', float('nan')):.4f}")
        print(f"    pca512_cf_r2           = {diag.get('pca512_cf_r2', float('nan')):.4f}")
        print(f"    overlap(shared,pca512) = {diag['overlap_shared_with_pca512']:.4f}")
    return diag


# =========================
#  Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--method", type=str, default="crosscov",
                    choices=["crosscov", "cca", "null", "random"],
                    help="null/random are controls: no-shared-information and "
                         "arbitrary-orthonormal-subspace")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--cca_ridge", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42, help="for --method random")
    ap.add_argument("--llm_emb_path", type=str, default="")
    ap.add_argument("--cf_emb_path", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--no_diagnostics", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[cgscd-basis] dataset={args.dataset} method={args.method} rank={args.rank}")

    Zc, Cc, z_mean, c_mean, zp, cp = load_views(
        args.dataset, args.llm_emb_path, args.cf_emb_path)
    d_llm, d_cf = Zc.shape[1], Cc.shape[1]

    if args.rank > d_cf:
        print(f"[warn] rank {args.rank} > d_cf {d_cf}; clamping to {d_cf} "
              f"(rank(M) <= min(d_llm, d_cf))")
    rank = min(args.rank, d_cf)

    U_r, S, info = estimate_basis(Zc, Cc, args.method, rank,
                                  cca_ridge=args.cca_ridge, seed=args.seed)

    diag = {} if args.no_diagnostics else compute_diagnostics(Zc, Cc, U_r, S)
    diag["fit_seconds"] = float(time.time() - t0)
    diag["d_llm"] = int(d_llm)
    diag["d_cf"] = int(d_cf)
    diag["n_items"] = int(Zc.shape[0])

    out = args.out or (
        f"./data/{args.dataset}/handled/cgscd_basis_{args.method}_r{rank}.pkl"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "U_r": U_r.astype(np.float32),
        "singular_values": S.astype(np.float32),
        "z_mean": z_mean,
        "c_mean": c_mean,
        "method": args.method,
        "rank": int(rank),
        "ridge": float(args.cca_ridge) if args.method == "cca" else None,
        "source": {
            "z_path": zp, "z_sha1": sha1_of(zp),
            "c_path": cp, "c_sha1": sha1_of(cp),
            "n_items": int(Zc.shape[0]),
        },
        "diagnostics": diag,
        **info,
    }
    pickle.dump(payload, open(out, "wb"))

    print(f"[cgscd-basis] saved: {out}")
    print(f"  U_r {payload['U_r'].shape}  sigma[:5]={np.round(S[:5], 4).tolist()}")
    print(f"  z_sha1={payload['source']['z_sha1'][:12]}  "
          f"c_sha1={payload['source']['c_sha1'][:12]}")
    print(f"  elapsed {diag['fit_seconds']:.1f}s")


if __name__ == "__main__":
    main()
