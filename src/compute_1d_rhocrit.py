"""
Compute 1D rho_crit for 0PA and 1PA using shift-aggregated Fisher matrices
evaluated at the bias-optimised (recovered) parameter points.

For each of the 25 EMRI grid points, 9 Fisher matrices are loaded from the
shift directories (3x3 in (a, e0)) computed by:
    fisher_grid_optimized_shifts_0pa_mle.py  →  fishers/mle/EMRI_0pa/
    fisher_grid_optimized_shifts_1pa_mle.py  →  fishers/mle/EMRI_1pa/

Each Fisher_transformed (log-mass-transformed K×K matrix) is inverted via SVD.
The median of σ_i across the 9 shifts gives a numerically robust uncertainty
estimate for each intrinsic parameter i.

Parameter sets (intrinsic only, no extrinsic):
    0PA : K=7  ['m1','m2','a','p0','e0','Phi_phi0','Phi_r0']
    1PA : K=8  ['m1','m2','a','p0','e0','Phi_phi0','Phi_r0','chi2']

Bias vector at SNR=20:
    Δλ_i = recovered_i - injected_i

1D criterion:
    ρ_crit^{1D} = SNR0 × r_{1,0.99} / max_i( |Δλ_i| / σ_i(SNR0) )

where r_{1,0.99} = norm.ppf(0.995) ≈ 2.5758.

Outputs (in bias_inference_emri/data/):
    sigmas/sigmas_0pa_K7_all.npy     (25, 9, 7)  — σ per shift per param
    sigmas/sigmas_0pa_K7_median.npy  (25, 7)     — median across shifts
    sigmas/sigmas_1pa_K8_all.npy     (25, 9, 8)
    sigmas/sigmas_1pa_K8_median.npy  (25, 8)
    rhocrit/rhocrit_1d_summary.txt               — human-readable table
    rhocrit/rhocrit_1d_0pa_K7.npy   (25,)
    rhocrit/rhocrit_1d_1pa_K8.npy   (25,)

Usage
-----
    python compute_1d_rhocrit.py
"""
import os
import argparse
import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SRC_DIR, '..', 'data')
SIGMA_DIR    = os.path.join(DATA_DIR, 'sigmas')
RHOCRIT_DIR  = os.path.join(DATA_DIR, 'rhocrit')

FISHER_BASE  = '/scratch/josh.mat/plot_scripts/data/fishers'
FISHER_0PA   = os.path.join(FISHER_BASE, 'mle', 'EMRI_0pa')
FISHER_1PA   = os.path.join(FISHER_BASE, 'mle', 'EMRI_1pa')

SIGNAL_FILE  = os.path.join(DATA_DIR, 'signal_parameter_array_EMRI.npy')
REC_0PA_FILE = '/scratch/josh.mat/plot_scripts/data/recovered/mle/recovered_parameter_array_EMRI_0pa.npy'
REC_1PA_FILE = '/scratch/josh.mat/plot_scripts/data/recovered/mle/recovered_parameter_array_EMRI_1pa.npy'

# Signal array column indices for the relevant parameters
# (same ordering as recovered arrays)
# Col: m1=0, m2=1, a=2, p0=3, e0=4, Phi_phi0=11, Phi_r0=13, chi2=16
SIG_COLS_0PA = [0, 1, 2, 3, 4, 11, 13]       # K=7
SIG_COLS_1PA = [0, 1, 2, 3, 4, 11, 13, 16]   # K=8
REC_COLS_0PA = [0, 1, 2, 3, 4, 11, 13]        # same indexing in rec array
REC_COLS_1PA = [0, 1, 2, 3, 4, 11, 13, 16]

PARAM_NAMES_0PA = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0']
PARAM_NAMES_1PA = ['m1', 'm2', 'a', 'p0', 'e0', 'Phi_phi0', 'Phi_r0', 'chi2']

N_SHIFTS = 9
N_GRID   = 25
SNR0     = 20.0
R1_99    = 2.5758   # norm.ppf(0.995)


# ---------------------------------------------------------------------------
# Fisher inversion
# ---------------------------------------------------------------------------
def invert_log_fisher(Fisher_transformed, m1, m2):
    """
    Invert a log-mass-transformed Fisher matrix and return physical-space sigmas.

    Fisher_transformed = J^T F J  where J = diag(m1, m2, 1, ..., 1).
    Uses np.linalg.inv directly (same as the chi2 uncertainty plots).
    Returns None if the matrix is not positive definite.

    Returns
    -------
    sigmas : (K,) array, or None if Fisher is not positive definite
    cond   : condition number (max/min eigenvalue)
    """
    K = Fisher_transformed.shape[0]
    J = np.eye(K)
    J[0, 0] = m1
    J[1, 1] = m2

    evals = np.linalg.eigvalsh(Fisher_transformed)
    cond  = evals.max() / max(evals.min(), 1e-300)
    if not (evals > 0).all():
        return None, cond

    C_log  = np.linalg.inv(Fisher_transformed)   # covariance in log-mass space
    C_phys = J @ C_log @ J                        # back to physical space
    sigmas = np.sqrt(np.diag(C_phys))
    return sigmas, cond


# ---------------------------------------------------------------------------
# Load all shifts for one grid point
# ---------------------------------------------------------------------------
def load_sigmas_for_point(fisher_base, idx, param_names, m1, m2):
    """
    Load Fisher_transformed from each shift directory and return sigma arrays.

    Returns
    -------
    sigma_all : (N_SHIFTS, K) array — NaN if file missing or not positive definite
    """
    K = len(param_names)
    sigma_all = np.full((N_SHIFTS, K), np.nan)

    for s in range(N_SHIFTS):
        fpath = os.path.join(fisher_base, f'shift_{s}', 'Fishers', f'Fisher_{idx}.h5')
        if not os.path.exists(fpath):
            continue
        try:
            with h5py.File(fpath, 'r') as f:
                Ft = f['Fisher_transformed'][:]
            sigmas, cond = invert_log_fisher(Ft, m1, m2)
            if sigmas is None:
                print(f'    [SKIP] shift_{s} idx={idx}: not positive definite  cond={cond:.2e}')
                continue
            sigma_all[s] = sigmas
            if s == 4:   # shift_4 is the nominal (da=0, de=0)
                print(f'    nominal shift: cond={cond:.2e}  '
                      f'σ_m1={sigmas[0]:.3e}  σ_a={sigmas[2]:.3e}  '
                      f'σ_Phi_phi0={sigmas[5]:.3e}')
        except Exception as exc:
            print(f'    [WARN] shift_{s} idx={idx}: {exc}')

    return sigma_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    os.makedirs(SIGMA_DIR,   exist_ok=True)
    os.makedirs(RHOCRIT_DIR, exist_ok=True)

    signal  = np.load(SIGNAL_FILE)    # (25, 17+)
    rec_0pa = np.load(REC_0PA_FILE)   # (25, 18)
    rec_1pa = np.load(REC_1PA_FILE)   # (25, 18)

    K7 = len(PARAM_NAMES_0PA)
    K8 = len(PARAM_NAMES_1PA)

    # (25, 9, K)  — σ per grid point per shift per parameter
    sigmas_0pa_all = np.full((N_GRID, N_SHIFTS, K7), np.nan)
    sigmas_1pa_all = np.full((N_GRID, N_SHIFTS, K8), np.nan)

    for i in range(N_GRID):
        m1_0pa = float(rec_0pa[i, REC_COLS_0PA[0]])
        m2_0pa = float(rec_0pa[i, REC_COLS_0PA[1]])
        m1_1pa = float(rec_1pa[i, REC_COLS_1PA[0]])
        m2_1pa = float(rec_1pa[i, REC_COLS_1PA[1]])

        a_inj  = float(signal[i, 2])
        e0_inj = float(signal[i, 4])
        print(f'\n[{i:2d}/24]  a_inj={a_inj:+.2f}  e0_inj={e0_inj:.2f}')

        print('  0PA:')
        sigmas_0pa_all[i] = load_sigmas_for_point(
            FISHER_0PA, i, PARAM_NAMES_0PA, m1_0pa, m2_0pa)

        print('  1PA:')
        sigmas_1pa_all[i] = load_sigmas_for_point(
            FISHER_1PA, i, PARAM_NAMES_1PA, m1_1pa, m2_1pa)

    # Median across all 9 shifts; nominal = shift index 4 (da=0, de=0)
    NOMINAL_SHIFT = 4
    sigmas_0pa_med = np.nanmedian(sigmas_0pa_all, axis=1)        # (25, 7)
    sigmas_1pa_med = np.nanmedian(sigmas_1pa_all, axis=1)        # (25, 8)
    sigmas_0pa_nom = sigmas_0pa_all[:, NOMINAL_SHIFT, :]          # (25, 7)
    sigmas_1pa_nom = sigmas_1pa_all[:, NOMINAL_SHIFT, :]          # (25, 8)

    np.save(os.path.join(SIGMA_DIR, 'sigmas_0pa_K7_all.npy'),     sigmas_0pa_all)
    np.save(os.path.join(SIGMA_DIR, 'sigmas_0pa_K7_median.npy'),  sigmas_0pa_med)
    np.save(os.path.join(SIGMA_DIR, 'sigmas_0pa_K7_nominal.npy'), sigmas_0pa_nom)
    np.save(os.path.join(SIGMA_DIR, 'sigmas_1pa_K8_all.npy'),     sigmas_1pa_all)
    np.save(os.path.join(SIGMA_DIR, 'sigmas_1pa_K8_median.npy'),  sigmas_1pa_med)
    np.save(os.path.join(SIGMA_DIR, 'sigmas_1pa_K8_nominal.npy'), sigmas_1pa_nom)
    print(f'\nSaved sigma arrays to {SIGMA_DIR}/ (median + nominal shift={NOMINAL_SHIFT})')

    # -----------------------------------------------------------------------
    # 1D rho_crit
    # -----------------------------------------------------------------------
    rhocrit_0pa = np.full(N_GRID, np.nan)
    rhocrit_1pa = np.full(N_GRID, np.nan)

    summary_path = os.path.join(RHOCRIT_DIR, 'rhocrit_1d_summary.txt')
    with open(summary_path, 'w') as fout:
        header = (
            f'1D rho_crit — intrinsic-only Fisher at MLE recovery point\n'
            f'Shift aggregation: {N_SHIFTS} shifts in (a,e0), median sigma\n'
            f'Inversion: np.linalg.inv (positive-definite check; NaN if not PD)\n'
            f'r_{{1,0.99}} = {R1_99:.4f},  SNR0 = {SNR0}\n'
            f'0PA K=7: {PARAM_NAMES_0PA}\n'
            f'1PA K=8: {PARAM_NAMES_1PA}\n'
        )
        fout.write(header)
        print('\n' + header)

        # Column headers
        def pn_abbrev(names):
            abbrev = {'m1': 'm1', 'm2': 'm2', 'a': 'a', 'p0': 'p0', 'e0': 'e0',
                      'Phi_phi0': 'Φφ', 'Phi_r0': 'Φr', 'chi2': 'χ2'}
            return [abbrev.get(n, n) for n in names]

        hdr_0pa = '  '.join(f'd_{p:4s}' for p in pn_abbrev(PARAM_NAMES_0PA))
        hdr_1pa = '  '.join(f'd_{p:4s}' for p in pn_abbrev(PARAM_NAMES_1PA))
        hdr = (f"\n{'idx':>3}  {'a':>6}  {'e0':>5}  ||  "
               f"{hdr_0pa}  {'max_d':>8}  {'ρcrit':>8}  ||  "
               f"{hdr_1pa}  {'max_d':>8}  {'ρcrit':>8}\n")
        sep = '-' * len(hdr.split('\n')[1])
        fout.write(hdr + sep + '\n')
        print(hdr + sep)

        for i in range(N_GRID):
            a_inj  = float(signal[i, 2])
            e0_inj = float(signal[i, 4])

            # Bias vectors (recovered − injected) in physical units
            delta_0pa = np.array([
                rec_0pa[i, REC_COLS_0PA[k]] - signal[i, SIG_COLS_0PA[k]]
                for k in range(K7)
            ])
            delta_1pa = np.array([
                rec_1pa[i, REC_COLS_1PA[k]] - signal[i, SIG_COLS_1PA[k]]
                for k in range(K8)
            ])

            s7 = sigmas_0pa_med[i]   # (7,) at SNR=20
            s8 = sigmas_1pa_med[i]   # (8,) at SNR=20

            # d_i = |Δλ_i| / σ_i  (dimensionless; SNR-independent at fixed SNR0)
            d_0pa = np.where(s7 > 0, np.abs(delta_0pa) / s7, np.nan)
            d_1pa = np.where(s8 > 0, np.abs(delta_1pa) / s8, np.nan)

            d0_max = np.nanmax(d_0pa)
            d1_max = np.nanmax(d_1pa)

            rc_0 = SNR0 * R1_99 / d0_max if np.isfinite(d0_max) and d0_max > 0 else np.nan
            rc_1 = SNR0 * R1_99 / d1_max if np.isfinite(d1_max) and d1_max > 0 else np.nan

            rhocrit_0pa[i] = rc_0
            rhocrit_1pa[i] = rc_1

            def fmt(v, w=8):
                return f'{v:{w}.3f}' if np.isfinite(v) else ' ' * (w-3) + 'nan'

            d0_str = '  '.join(fmt(v) for v in d_0pa)
            d1_str = '  '.join(fmt(v) for v in d_1pa)
            line = (f'{i:3d}  {a_inj:+5.2f}  {e0_inj:.2f}  ||  '
                    f'{d0_str}  {fmt(d0_max)}  {fmt(rc_0)}  ||  '
                    f'{d1_str}  {fmt(d1_max)}  {fmt(rc_1)}\n')
            fout.write(line)
            print(line, end='')

    np.save(os.path.join(RHOCRIT_DIR, 'rhocrit_1d_0pa_K7.npy'), rhocrit_0pa)
    np.save(os.path.join(RHOCRIT_DIR, 'rhocrit_1d_1pa_K8.npy'), rhocrit_1pa)

    print(f'\nSaved ρ_crit arrays to {RHOCRIT_DIR}/')
    print(f'  0PA: min={np.nanmin(rhocrit_0pa):.2f}  '
          f'max={np.nanmax(rhocrit_0pa):.2f}  '
          f'median={np.nanmedian(rhocrit_0pa):.2f}')
    print(f'  1PA: min={np.nanmin(rhocrit_1pa):.3e}  '
          f'max={np.nanmax(rhocrit_1pa):.3e}  '
          f'median={np.nanmedian(rhocrit_1pa):.3e}')
    print(f'\nSummary table written to {summary_path}')


if __name__ == '__main__':
    main()
