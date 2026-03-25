"""
cp_pipeline.py
==============
Cross-patient sEEG tensor decomposition pipeline.

Built on top of the supervisor's sparsify.py logic. The core sparsification
loop is kept identical — same variable names, same structure. Our additions
are clearly marked with  # ── ADDED  comments so the team can see exactly
what we extended vs what was inherited.

Pipeline steps
--------------
  1. Load the electrode atlas (elecs.npz) — provided by supervisor
  2. For each patient: map channels onto 1000 shared brain clusters
     (supervisor's logic) and build a binary observation mask (our addition)
  3. Exclude patients who fail quality checks — wrong length, NaN signal,
     no electrode matched any cluster, or known data quality issues.
     Print a full exclusion report.
  4. Stack all patients into a 3-mode tensor  X[time x clusters x patients]
  5. Robust normalisation — median + IQR per patient per cluster
  6. CP decomposition via iterative imputation + randomised block sampling
  7. Print component summaries and save factor matrices to cp_factors.npz

Usage
-----
  python cp_pipeline.py --data_dir /path/to/data --rank 10

Authors
-------
  Supervisor: sparsify.py (steps 1-2 core logic)
  Team:       tensor construction, normalisation, decomposition (steps 3-7)
"""

import os
import time
import argparse
import numpy as np
import tensorly as tl
from tensorly.decomposition import randomised_parafac
from scipy.stats import iqr


# Only patients with exactly this many time samples are included.
REQUIRED_T = 6000

# Patients excluded due to known data quality issues.
# kh003: signal saturation — 10+ channels all hitting ~69.5-70 uV ceiling,
#        consistent with hardware clipping artifact, not real brain activity.
# Add future known-bad patients here with a reason string.
EXCLUDE_PATIENTS = {
    'kh003': 'signal saturation (hardware clipping ~70 uV)'
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SPARSIFY ONE PATIENT
# ══════════════════════════════════════════════════════════════════════════════
#
# Direct extension of supervisor's sparsify.py.
# Same variable names (pElecs, clusters, cNr, c) and same loop structure.
#
# Additions:
#   - NaN / Inf detection before any processing
#   - Binary mask W tracking which clusters have a real electrode
#   - Metadata return for the exclusion report

def sparsify_patient(p, elecs, data_path):
    """
    Parameters
    ----------
    p         : str   patient ID e.g. 'kh001'
    elecs     : dict  loaded elecs.npz
    data_path : str   path to patient .npz file

    Returns
    -------
    sparse        : ndarray (T, 1000) or None
    mask          : ndarray (T, 1000) or None
    obs_clusters  : list[int]  or None
    n_channels    : int        or None
    unique_regions: list[str]  or None
    err           : str or None  — exclusion reason, None if patient is fine
    """
    if not os.path.exists(data_path):
        return None, None, None, None, None, 'file not found'

    data   = np.load(data_path)
    signal = data['data']           # shape: (T, n_channels)

    # ── ADDED: quality check for NaN / Inf ───────────────────────────────────
    # NaN propagates through every matrix operation and silently corrupts
    # the entire decomposition. Reject before doing anything else.
    n_nan = int(np.isnan(signal).sum())
    n_inf = int(np.isinf(signal).sum())
    if n_nan > 0 or n_inf > 0:
        return None, None, None, int(signal.shape[1]), None, \
               f'{n_nan} NaN + {n_inf} Inf values in signal'
    # ─────────────────────────────────────────────────────────────────────────

    # Supervisor's variable names kept as-is
    pElecs   = elecs['pts'][elecs['pts'][:, 0] == p, 1]
    clusters = elecs['cluster'][elecs['pts'][:, 0] == p]
    regions  = elecs['pts'][elecs['pts'][:, 0] == p, 2]

    # Supervisor's sparse matrix allocation
    sparse = np.zeros((signal.shape[0], np.max(elecs['cluster']) + 1),
                      dtype=np.float32)

    # ── ADDED: binary observation mask ───────────────────────────────────────
    # 1 where a real electrode is present, 0 everywhere else.
    # Used in the CP loss function to ignore the ~94% of empty clusters.
    mask = np.zeros_like(sparse)
    # ─────────────────────────────────────────────────────────────────────────

    obs_clusters = []
    obs_regions  = []

    # ── Supervisor's core loop — kept identical ───────────────────────────────
    for cNr, c in enumerate(data['chNames'].flatten()):
        if c in pElecs:
            idx = np.where(pElecs == c)
            sparse[:, clusters[idx]] = signal[:, cNr][:, None]

            # ── ADDED: mark observed clusters ────────────────────────────────
            mask[:, clusters[idx]] = 1.0
            obs_clusters.extend(clusters[idx].tolist())
            obs_regions.extend(regions[idx].tolist())
            # ─────────────────────────────────────────────────────────────────

    unique_regions = sorted(set(obs_regions))
    return sparse, mask, list(set(obs_clusters)), \
           len(data['chNames'].flatten()), unique_regions, None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD THE FULL TENSOR
# ══════════════════════════════════════════════════════════════════════════════
#
# Iterates all patients in the atlas, applies four inclusion criteria,
# and stacks passing patients into a 3-mode tensor (T x C x P).
#
# Exclusion criteria (checked in order):
#   1. Patient in EXCLUDE_PATIENTS — known data quality issue
#   2. File not found
#   3. Signal contains NaN or Inf
#   4. Recording length != REQUIRED_T
#   5. No electrode matched any cluster in the atlas

def build_tensor(data_dir, elecs):
    """
    Returns
    -------
    X    : ndarray (T, C, P)   time x clusters x patients
    W    : ndarray (T, C, P)   binary observation mask
    pids : list[str]           patient IDs in tensor order
    """
    pts = np.unique(elecs['pts'][:, 0])   # same variable name as supervisor

    slices, masks, pids = [], [], []
    excluded = []

    for p in pts:

        # ── Exclusion 1: known bad patient ───────────────────────────────────
        if p in EXCLUDE_PATIENTS:
            excluded.append({'id': p,
                             'reason': EXCLUDE_PATIENTS[p],
                             'T': None, 'channels': None,
                             'active_clusters': None, 'regions': []})
            continue
        # ─────────────────────────────────────────────────────────────────────

        data_path = os.path.join(data_dir, '%s.npz' % p)
        sparse, mask, obs_clusters, n_channels, regions, err = \
            sparsify_patient(p, elecs, data_path)

        # ── Exclusion 2+3: file not found or NaN/Inf ─────────────────────────
        if sparse is None:
            reason = err if err else 'file not found'
            excluded.append({'id': p, 'reason': reason,
                             'T': None, 'channels': n_channels,
                             'active_clusters': None, 'regions': []})
            continue

        T           = sparse.shape[0]
        n_act_clust = len(obs_clusters)

        # ── Exclusion 4: wrong recording length ──────────────────────────────
        if T != REQUIRED_T:
            excluded.append({'id': p,
                             'reason': f'T={T} (required {REQUIRED_T})',
                             'T': T, 'channels': n_channels,
                             'active_clusters': n_act_clust,
                             'regions': regions})
            continue

        # ── Exclusion 5: no electrode matched any cluster ────────────────────
        if n_act_clust == 0:
            excluded.append({'id': p,
                             'reason': 'no electrode matched any cluster',
                             'T': T, 'channels': n_channels,
                             'active_clusters': 0, 'regions': []})
            continue

        slices.append(sparse)
        masks.append(mask)
        pids.append(p)
        print(f'  Loaded {p}  T={T}  channels={n_channels}  '
              f'active_clusters={n_act_clust}  regions={len(regions)}')

    # ── Exclusion report ──────────────────────────────────────────────────────
    print(f'\n{"─"*62}')
    print(f'  Included : {len(pids)} patients')
    print(f'  Excluded : {len(excluded)} patients')
    print(f'{"─"*62}')
    if excluded:
        reasons = {}
        for e in excluded:
            reasons[e['reason']] = reasons.get(e['reason'], 0) + 1
        print('  Breakdown by reason:')
        for r, n in reasons.items():
            print(f'    {n:3d}  {r}')
        print()
        non_missing = [e for e in excluded
                       if e['reason'] != 'file not found']
        if non_missing:
            print(f'  {"Patient":<10} {"Reason":<38} {"T":>6} '
                  f'{"Ch":>4} {"Clust":>5}  Regions')
            print(f'  {"─"*9} {"─"*37} {"─"*6} '
                  f'{"─"*4} {"─"*5}  {"─"*20}')
            for e in non_missing:
                T_s  = str(e['T'])  if e['T']  is not None else 'N/A'
                ch_s = str(e['channels']) \
                       if e['channels'] is not None else 'N/A'
                cl_s = str(e['active_clusters']) \
                       if e['active_clusters'] is not None else 'N/A'
                reg_s = ', '.join(e['regions'][:3])
                if len(e['regions']) > 3:
                    reg_s += f' (+{len(e["regions"])-3})'
                print(f'  {e["id"]:<10} {e["reason"]:<38} {T_s:>6} '
                      f'{ch_s:>4} {cl_s:>5}  {reg_s}')
    print(f'{"─"*62}\n')

    if len(pids) == 0:
        raise ValueError('No patients passed the inclusion criteria.')

    X = np.stack(slices, axis=2).astype(np.float32)
    W = np.stack(masks,  axis=2).astype(np.float32)
    return X, W, pids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ROBUST NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
#
# Per-patient per-cluster robust scaling using median and IQR.
# Preferred over z-score because mean and std are pulled by outlier amplitudes.
# Values beyond +-10 IQR are clipped as they are almost certainly artifacts.

def normalise(X, W):
    X_norm = X.copy()
    T, C, P = X.shape
    for p in range(P):
        for c in range(C):
            # Only normalise where real data exists for this patient+cluster
            if W[:, c, p].sum() > 1:
                col   = X[:, c, p]
                obs   = col[W[:, c, p] > 0]
                med   = np.median(obs)
                scale = iqr(obs) + 1e-8     # +epsilon avoids division by zero
                normed = (col - med) / scale
                normed = np.clip(normed, -10, 10)   # clip artifact spikes
                X_norm[:, c, p] = normed * (W[:, c, p] > 0)
    return X_norm


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CP DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
#
# Approximates X as a sum of R rank-1 components:
#   X  ≈  sum_r  a_r ∘ b_r ∘ c_r
#
# Two challenges solved:
#   1. Missing data (94% empty) — iterative imputation: fill blanks with
#      current reconstruction, update decomposition, refill, repeat.
#      Loss is measured only over observed entries (W > 0).
#   2. Scale (6000x1000x92) — randomised_parafac samples 30% of the time
#      dimension per update instead of the full tensor.
#
# Multiple random initialisations reduce the risk of getting stuck in a
# local minimum (ALS is not guaranteed to find the global optimum).

def cp_decompose(X, W, rank, n_iter=200, tol=1e-5, n_inits=3,
                 n_outer=10, random_state=0):
    best_factors, best_loss = None, np.inf
    T         = X.shape[0]
    n_samples = max(10, int(T * 0.3))   # 30% of time dimension per update

    for i in range(n_inits):
        print(f'  Init {i+1}/{n_inits}')

        # Fill missing entries with zero as starting point.
        # Zeros are neutral for signals centred around zero after normalisation.
        X_filled  = X.copy()
        X_filled[W == 0] = 0
        prev_loss = np.inf
        cp        = None

        for outer in range(n_outer):

            # SVD init on first outer loop only — gives smarter starting point.
            # Random init on all subsequent loops because the filled tensor
            # has changed so SVD would be stale.
            cp = randomised_parafac(
                tl.tensor(X_filled),
                rank=rank,
                n_samples=n_samples,
                n_iter_max=max(10, n_iter // n_outer),
                init='svd' if (outer == 0 and i == 0) else 'random',
                random_state=random_state + i + outer,
            )

            # Reconstruct and re-impute ONLY the missing entries.
            # Observed entries are never overwritten — real data stays real.
            X_hat            = np.array(tl.cp_to_tensor(cp))
            X_filled         = X.copy()
            X_filled[W == 0] = X_hat[W == 0]

            # Loss over observed entries only
            loss = float(np.sum((W > 0) * (X - X_hat) ** 2))
            print(f'    outer {outer+1}/{n_outer}  loss={loss:.2f}')

            if abs(prev_loss - loss) / (prev_loss + 1e-12) < tol:
                print(f'    Converged at outer iter {outer+1}')
                break
            prev_loss = loss

        print(f'  Init {i+1} done — loss={loss:.2f}')
        if loss < best_loss:
            best_loss    = loss
            best_factors = [np.array(f) for f in cp.factors]

    print(f'\nBest loss: {best_loss:.2f}')
    return best_factors, best_loss


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INTERPRET COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def interpret(factors, pids, elecs, top_k=5):
    A, B, C_mat = factors   # A=(TxR), B=(CxR), C_mat=(PxR)
    R         = A.shape[1]
    centroids = elecs['centroids']  # MNI xyz of each cluster centroid

    print('\n=== CP Component Summary ===')
    for r in range(R):
        spatial  = B[:, r]
        temporal = A[:, r]
        patient  = C_mat[:, r]

        # Rank by absolute loading — catches both activation (+) and
        # suppression (-) as strongly involved
        top_clusters = np.argsort(np.abs(spatial))[-top_k:][::-1]
        top_mni      = centroids[top_clusters]
        peak_t       = int(np.argmax(np.abs(temporal)))

        print(f'\n--- Component {r+1} ---')
        print(f'  Peak time sample : {peak_t}')
        print(f'  Top {top_k} clusters (MNI xyz):')
        for ci, mni in zip(top_clusters[:top_k], top_mni[:top_k]):
            print(f'    cluster {ci:4d}  [{mni[0]:+.1f}, '
                  f'{mni[1]:+.1f}, {mni[2]:+.1f}]  '
                  f'loading={spatial[ci]:.3f}')
        top_pts = np.argsort(np.abs(patient))[-3:][::-1]
        print(f'  Strongest patients: {[str(pids[j]) for j in top_pts]}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()

    parser = argparse.ArgumentParser(
        description='CP decomposition pipeline for cross-patient sEEG data')
    parser.add_argument('--data_dir', default='.',
                        help='Folder with elecs.npz and patient .npz files')
    parser.add_argument('--rank',    type=int, default=10,
                        help='Number of CP components R (default: 10)')
    parser.add_argument('--n_inits', type=int, default=3,
                        help='Random initialisations (default: 3)')
    parser.add_argument('--n_iter',  type=int, default=200,
                        help='Total iteration budget per init (default: 200)')
    parser.add_argument('--n_outer', type=int, default=10,
                        help='Imputation outer loops per init (default: 10)')
    args = parser.parse_args()

    print('Loading electrode atlas ...')
    elecs = np.load(os.path.join(args.data_dir, 'elecs.npz'))
    print(f'  Atlas: {elecs["pts"].shape[0]} electrodes  '
          f'{int(elecs["cluster"].max())+1} clusters  '
          f'{len(np.unique(elecs["pts"][:,0]))} patients in atlas\n')

    print('Building tensor ...')
    X, W, pids = build_tensor(args.data_dir, elecs)
    print(f'Tensor shape: {X.shape}  (time x clusters x patients)')
    print(f'Density: {100*W.mean():.2f}%  '
          f'({int(W.sum())} observed / {W.size} total entries)\n')

    print('Normalising ...')
    X = normalise(X, W)

    print(f'Running CP decomposition  rank={args.rank} ...')
    factors, loss = cp_decompose(
        X, W,
        rank=args.rank,
        n_iter=args.n_iter,
        n_inits=args.n_inits,
        n_outer=args.n_outer,
    )

    interpret(factors, pids, elecs)

    # Save factor matrices for downstream use
    # A = temporal factors  (T x R)
    # B = spatial factors   (C x R)  — the brain atlas
    # C = patient factors   (P x R)  — patient fingerprints
    out = os.path.join(args.data_dir, 'cp_factors.npz')
    np.savez(out,
             A=factors[0],
             B=factors[1],
             C=factors[2],
             pids=np.array(pids))
    print(f'\nFactors saved to {out}')

    elapsed = time.time() - start
    print(f'Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)')


if __name__ == '__main__':
    main()
